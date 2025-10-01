"""
Adaptive Headway Controller for CARLA
------------------------------------
Implements the adaptive active inference controller for headway maintenance
in the CARLA simulation environment using PyMDP.
"""

import math
from collections import deque

import carla
import numpy as np
# Import PyMDP components
from pymdp import utils
from pymdp.agent import Agent


class AdaptiveHeadwayController:
    def __init__(self, ego_vehicle, lead_vehicle, target_headway=15.0, 
                 adaptive=True, time_step=0.05, policy_len=1):
        """
        Initialize the adaptive headway controller for CARLA
        
        Args:
            ego_vehicle: CARLA actor representing the ego vehicle
            lead_vehicle: CARLA actor representing the lead vehicle
            target_headway: Target following distance in meters
            adaptive: Whether to use adaptive control
            time_step: Simulation time step in seconds
            policy_len: Length of policies to consider
        """
        self.ego_vehicle = ego_vehicle
        self.lead_vehicle = lead_vehicle
        self.target_headway = target_headway
        self.use_adaptive = adaptive
        self.time_step = time_step
        self.state_values = np.linspace(-5, 100, 45)
        
        # Current state
        self.relative_dist = None
        self.ego_velocity = 0.0
        self.lead_velocity = 0.0
        self.ego_acceleration = 0.0
        self.lead_acceleration = 0.0
        self.previous_lead_velocity = 0.0
        self.previous_ego_velocity = 0.0
        self.previous_acc = 0.0
        self.headway_error_history = deque(maxlen=100)
        self.velocity_change_history = deque(maxlen=100)
        self.lead_behavior_history = deque(maxlen=20)
        
        # Adaptation parameters
        self.smoothing_factor = 0.5  # Initial value
        self.dip_factor = 1.0        # Initial value
        self.acc_scaling = 0.5      # Initial acceleration scaling

        # Dual mode control
        self.catchup_mode = False
        self.catchup_threshold = 3.0  # Enter catchup mode if headway error > 3.0m
        
        # Initialize by calculating current headway
        self.update_state()
        
        # Set up active inference model
        self.setup_ai_model(policy_len)
        
        # Performance history
        
        
        
    def setup_ai_model(self, policy_len):
        """Set up the active inference model with PyMDP"""
        # Model parameters
        num_states = 51
        relative_dist_min = -10
        relative_dist_max = 30
        noise_var = 0.05
        
        # Create state space
        self.state_values = np.linspace(relative_dist_min, relative_dist_max, num_states)
        self.num_states = num_states
        self.noise_var = noise_var
        
        # Number of actions
        self.num_actions = 9
        
        # Create acceleration map with finer gradations for more control
        self.acceleration_map = {}
        for i in range(self.num_actions):
            self.acceleration_map[i] = -2.5 + 0.7 * i
        
        # Find current state index
        self.current_state_idx = self._find_closest_state_idx(self.relative_dist)
        
        # Set up A matrix (observation model)
        self.A = utils.obj_array(1)
        self.A[0] = np.zeros((self.num_states, self.num_states))
        
        # Fill A matrix with Gaussian-like probabilities
        for s in range(self.num_states):
            for o in range(self.num_states):
                # Distance between state and observation
                dist = self.state_values[s] - self.state_values[o]
                # Gaussian probability density
                self.A[0][o, s] = np.exp(-(dist**2) / (2 * self.noise_var))
            
            # Normalize
            self.A[0][:, s] = utils.norm_dist(self.A[0][:, s])
        
        # Set up B matrix (transition model)
        self.B = utils.obj_array(1)
        self.B[0] = np.zeros((self.num_states, self.num_states, self.num_actions))
        
        # Define strength of each action
        action_strengths = [5, 4, 2, 1, 0, -1, -2, -4, -5]  # Relative position change
        
        # Fill B matrix with transition probabilities
        for a, strength in enumerate(action_strengths):
            for s in range(self.num_states):
                # Target idx changes based on action strength
                target_idx = max(0, min(s + strength, self.num_states - 1))
                
                # Create a probability distribution
                probs = np.zeros(self.num_states)
                
                # Create a focused distribution
                width = 0.8  # Controls how focused the distribution is
                for i in range(self.num_states):
                    dist = abs(i - target_idx)
                    probs[i] = np.exp(-(dist**2) / (2 * width))
                
                # Normalize and assign
                self.B[0][:, s, a] = utils.norm_dist(probs)
        
        # Set up C matrix (preferences)
        self.C = utils.obj_array(1)
        self.C[0] = np.zeros(self.num_states)
        
        # Set up D matrix (prior beliefs)
        self.D = utils.obj_array(1)
        self.D[0] = np.zeros(self.num_states)
        self.D[0][self.current_state_idx] = 1.0  # Start with certainty about current state
        
        # Set preferred headway
        self.set_preferred_headway(self.target_headway, self.dip_factor)
        
        # Create PyMDP agent
        self.agent = Agent(
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            control_fac_idx=[0],  # Only one controllable factor
            policy_len=policy_len,  # Use specified policy length
            inference_algo='VANILLA',  # Use basic algorithm
            action_selection='deterministic'
        )
    
    def _find_closest_state_idx(self, value):
        """Find closest state index for a given value"""
        return np.argmin(np.abs(self.state_values - value))
    
    def set_preferred_headway(self, target_headway, dip_factor=1.5):
        """Set preferred headway distance with adaptive dipping"""
        # Find closest state index
        target_idx = self._find_closest_state_idx(target_headway)
        
        # Create preference distribution with peak at target headway
        preferences = np.zeros(self.num_states)
        
        # Adjust gradient based on mode
        gradient_factor = 2.0
        if self.catchup_mode:
            gradient_factor = 1.5  # Less steep gradient for more focus on closing gap
        
        # Create asymmetric preference gradient - penalize being too far more than too close
        for i in range(self.num_states):
            dist = self.state_values[i] - self.state_values[target_idx]
            
            # Asymmetric quadratic preference
            if dist > 0:  # Too far away - stronger penalty
                preferences[i] = -gradient_factor * 3.0 * (dist**2)
            else:  # Too close - normal penalty
                preferences[i] = -gradient_factor * 2.0 * (dist**2)
        
        # Safety: More gradual penalties for getting too close
        for i in range(self.num_states):
            if self.state_values[i] < (target_headway - 1.0):
                # Extra penalty for getting dangerously close
                distance_below = target_headway - 1.0 - self.state_values[i]
                preferences[i] -= 10.0 * (distance_below**2)
        
        # Create a dip with adaptive strength based on mode
        current_idx = self._find_closest_state_idx(self.relative_dist)
        
        # Only apply dip if we're farther away than target
        if current_idx > target_idx:
            # Create a smoother "valley" of enhanced preferences
            min_allowed_idx = self._find_closest_state_idx(max(target_headway * 0.7, 3.0))
            
            # Adjust dip shape based on mode
            dip_width = 2.5
            dip_height = 8.0
            
            if self.catchup_mode:
                # In catchup mode, create wider, stronger dip for more aggressive approach
                dip_width = 3.5
                dip_height = 12.0
            
            for i in range(self.num_states):
                # If this state is between target and current (with conservative limit)
                if (i < current_idx and i >= min_allowed_idx):
                    # Calculate how far into "dip zone" we are
                    progress = 1.0 - (i - min_allowed_idx) / max(1, (current_idx - min_allowed_idx))
                    
                    # Create parabolic dip with adaptive strength
                    dip_value = dip_factor * progress * (1.0 - progress) * dip_width
                    
                    # Add the dip value to preferences (making these states more attractive)
                    preferences[i] += dip_value * dip_height
        
        # Set as preference (C matrix)
        self.C[0] = preferences
        if hasattr(self, 'agent'):
            self.agent.C = self.C
    
    def update_state(self):
        """Update vehicle states from CARLA"""
        # Get vehicle transforms and velocities
        ego_transform = self.ego_vehicle.get_transform()
        lead_transform = self.lead_vehicle.get_transform()
        
        # Get velocity vectors
        ego_velocity_vec = self.ego_vehicle.get_velocity()
        lead_velocity_vec = self.lead_vehicle.get_velocity()
        
        # Calculate scalar velocities (magnitude of velocity vectors)
        self.previous_ego_velocity = self.ego_velocity
        self.previous_lead_velocity = self.lead_velocity
        
        self.ego_velocity = math.sqrt(
            ego_velocity_vec.x**2 + ego_velocity_vec.y**2 + ego_velocity_vec.z**2)
        self.lead_velocity = math.sqrt(
            lead_velocity_vec.x**2 + lead_velocity_vec.y**2 + lead_velocity_vec.z**2)
        
        # Calculate accelerations
        self.ego_acceleration = (self.ego_velocity - self.previous_ego_velocity) / self.time_step
        self.lead_acceleration = (self.lead_velocity - self.previous_lead_velocity) / self.time_step
        
        # Track lead vehicle behavior
        self.lead_behavior_history.append(self.lead_acceleration)

        # Calculate headway distance (distance between vehicles in the direction of travel)
        ego_loc = ego_transform.location
        lead_loc = lead_transform.location
        ego_forward = ego_transform.get_forward_vector()
        
        # Vector from ego to lead
        vector_to_lead = carla.Vector3D(
            lead_loc.x - ego_loc.x,
            lead_loc.y - ego_loc.y,
            lead_loc.z - ego_loc.z
        )
        
        # Project onto forward direction (dot product)
        # This gives distance in the direction of travel
        self.relative_dist = (vector_to_lead.x * ego_forward.x + 
                            vector_to_lead.y * ego_forward.y + 
                            vector_to_lead.z * ego_forward.z)
        
        # Make sure distance is always positive
        self.relative_dist = max(0.1, self.relative_dist)
        # Update current state index
        self.current_state_idx = self._find_closest_state_idx(self.relative_dist)
        
        # Track headway error
        headway_error = self.relative_dist - self.target_headway
        self.headway_error_history.append(headway_error)
    
    def adapt_parameters(self):
        """Adapt control parameters based on performance"""
        if len(self.headway_error_history) < 10:
            return  # Need sufficient history
        
        # Calculate recent performance metrics
        recent_errors = list(self.headway_error_history)
        mean_error = np.mean(np.abs(recent_errors))
        error_variance = np.var(recent_errors)
        
        # Detect if we need to enter catchup mode
        current_error = recent_errors[-1]
        previous_errors = recent_errors[-min(10, len(recent_errors)):-1]
        
        if len(previous_errors) > 1:
            error_increasing = np.mean(np.diff(previous_errors)) > 0
        else:
            error_increasing = False
        
        # Calculate lead vehicle acceleration trend
        lead_accel_trend = 0
        if len(self.lead_behavior_history) > 5:
            recent_lead_accels = list(self.lead_behavior_history)
            if len(recent_lead_accels) > 1:
                lead_accel_trend = np.mean(np.diff(recent_lead_accels))
        
        # Determine mode based on error and lead vehicle behavior
        old_mode = self.catchup_mode
        if current_error > self.catchup_threshold and error_increasing and lead_accel_trend >= 0:
            self.catchup_mode = True
        elif current_error < self.catchup_threshold * 0.7:  # Hysteresis to prevent oscillation
            self.catchup_mode = False
        
        # If mode changed, update parameters
        if old_mode != self.catchup_mode:
            print(f"Switched to {'catchup' if self.catchup_mode else 'normal'} mode.")
            
            # Update policy length in agent
            policy_len = 4 if self.catchup_mode else 2
            self.agent.policy_len = policy_len
        
        if len(self.velocity_change_history) > 1:
            recent_velocity_changes = list(self.velocity_change_history)
            velocity_variance = np.var(recent_velocity_changes)
            
            # Adapt smoothing factor
            if self.catchup_mode:
                # In catchup mode, be more responsive (lower smoothing)
                target_smoothing = 0.3
                self.smoothing_factor = 0.9 * self.smoothing_factor + 0.1 * target_smoothing
            else:
                # In normal mode, adjust based on variance
                if velocity_variance > 1.0:
                    self.smoothing_factor = min(0.9, self.smoothing_factor + 0.02)
                else:
                    self.smoothing_factor = max(0.3, self.smoothing_factor - 0.01)
            
            # Adapt acceleration scaling
            if self.catchup_mode:
                # In catchup mode, use more aggressive acceleration
                target_scaling = 1.5
                self.acc_scaling = 0.8 * self.acc_scaling + 0.2 * target_scaling
            else:
                # In normal mode, adjust based on error and variance
                if error_variance > 1.0 and velocity_variance > 0.5:
                    self.acc_scaling = max(0.7, self.acc_scaling - 0.05)
                else:
                    self.acc_scaling = min(1.2, self.acc_scaling + 0.01)
        
        # Adapt dip factor based on mode and headway error
        if self.catchup_mode:
            # In catchup mode, use more aggressive dipping
            target_dip = 2.5
            self.dip_factor = 0.9 * self.dip_factor + 0.1 * target_dip
        else:
            # Normal mode adjustment
            if mean_error > 1.0:
                self.dip_factor = max(0.5, self.dip_factor - 0.1)
            else:
                self.dip_factor = min(2.0, self.dip_factor + 0.05)
        
        # Update preference function
        self.set_preferred_headway(self.target_headway, self.dip_factor)
    
    def step(self):
        """Perform one step of adaptive headway control"""
        # Update current state
        self.update_state()
        
        # Periodically adapt parameters if using adaptive control
        if self.use_adaptive and len(self.headway_error_history) % 5 == 0:
            self.adapt_parameters()
        
        # Generate observation for active inference
        obs_idx = self._find_closest_state_idx(self.relative_dist)
        obs = [obs_idx]
        
        # Update D with current state belief
        qs = self.agent.infer_states(obs)
        self.D[0] = qs[0]
        self.agent.D = self.D
        
        # Get policy and expected free energies
        q_pi, efe = self.agent.infer_policies()
        
        # Sample action using active inference
        action = self.agent.sample_action()
        action_idx = action[0]
        
        # Map action to acceleration with adaptive scaling
        base_acc = self.acceleration_map[action_idx]
        
        # Apply mode-specific acceleration scaling
        mode_scaling = self.acc_scaling
        if self.catchup_mode and action_idx > 4:  # For acceleration actions in catchup mode
            # Apply extra scaling to accelerations when in catchup mode
            mode_scaling = self.acc_scaling * 1.3
        
        desired_acc = base_acc * mode_scaling
        
        # Apply temporal smoothing to reduce rapid fluctuations
        current_smoothing = self.smoothing_factor
        
        # Reduce smoothing for large headway errors to improve responsiveness
        headway_error = abs(self.relative_dist - self.target_headway)
        if headway_error > self.catchup_threshold:
            error_factor = min(1.0, headway_error / (2 * self.catchup_threshold))
            current_smoothing *= (1.0 - 0.5 * error_factor)
        
        # Apply the adjusted smoothing
        ego_acceleration = current_smoothing * self.previous_acc + (1 - current_smoothing) * desired_acc
        
        # Save for next iteration
        self.previous_acc = ego_acceleration
        
        control = carla.VehicleControl()

        if ego_acceleration >= 0:
            # For acceleration, apply throttle (normalized to 0-1 range)
            # Assuming max acceleration of 3 m/s^2 for full throttle
            control.throttle = min(1.0, ego_acceleration / 3.0)
            control.brake = 0.0
        else:
            # For deceleration, apply brake (normalized to 0-1 range)
            # Assuming max deceleration of -5 m/s^2 for full brake
            control.throttle = 0.0
            control.brake = min(1.0, -ego_acceleration / 5.0)

        # Keep steering straight
        control.steer = 0.0
        control.hand_brake = False
        control.manual_gear_shift = False

        # Apply the control to the vehicle
        self.ego_vehicle.apply_control(control)
                
        # Calculate velocity change for adaptation
        ego_velocity_change = self.ego_velocity - self.previous_ego_velocity
        self.velocity_change_history.append(ego_velocity_change)
        
        # Return current state and adaptive parameters
        return {
            'relative_dist': self.relative_dist,
            'ego_position': self.ego_vehicle.get_location(),
            'lead_position': self.lead_vehicle.get_location(),
            'ego_velocity': self.ego_velocity,
            'lead_velocity': self.lead_velocity,
            'ego_acceleration': ego_acceleration,
            'lead_acceleration': self.lead_acceleration,
            'action': action_idx,
            'smoothing_factor': self.smoothing_factor if self.use_adaptive else None,
            'acc_scaling': self.acc_scaling if self.use_adaptive else None,
            'dip_factor': self.dip_factor if self.use_adaptive else None,
            'catchup_mode': self.catchup_mode if self.use_adaptive else None
        }