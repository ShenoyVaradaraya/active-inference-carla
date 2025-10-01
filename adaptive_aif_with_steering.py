"""
Adaptive Headway Controller for CARLA
------------------------------------
Implements the adaptive active inference controller for headway maintenance
in the CARLA simulation environment using PyMDP.
"""

import math
import random
import time
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
        self.heading_values = np.linspace(-math.pi, math.pi, 45)
        world = ego_vehicle.get_world()
        self.map = world.get_map()
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
        self.lateral_error_history = deque(maxlen=100)
        self.velocity_change_history = deque(maxlen=100)
        self.lead_behavior_history = deque(maxlen=20)
        self.relative_heading = 0.0  # Initialize to zero
        self.previous_steering = 0.0
        self.num_heading_states = 31
            
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

    def setup_ai_model(self, policy_len):
        """Set up the active inference model with PyMDP"""
        # Model parameters
        num_distance_states = 51
        num_heading_states = 31
        
        relative_dist_min = -10
        relative_dist_max = 30
        heading_min = -math.pi
        heading_max = math.pi
        noise_var = 0.05
        heading_noise_var = 0.02
        
        # Create state space values
        self.state_values = np.linspace(relative_dist_min, relative_dist_max, num_distance_states)
        self.heading_values = np.linspace(heading_min, heading_max, num_heading_states)
        self.noise_var = noise_var
        self.heading_noise_var = heading_noise_var
        
        # Define state factors
        self.num_states = [num_distance_states, num_heading_states]
        num_factors = len(self.num_states)
        
        # Define control factors
        self.num_actions = 9  # Acceleration actions
        self.num_steering_actions = 15  # Steering actions
        num_controls = [self.num_actions, self.num_steering_actions]
        
        # Create acceleration and steering action maps
        self.acceleration_map = {}
        for i in range(self.num_actions):
            self.acceleration_map[i] = -2.5 + 0.7 * i
        
        # Find current state indices
        self.current_state_idx = self._find_closest_state_idx(self.relative_dist)
        self.current_heading_idx = self._find_closest_heading_idx(self.relative_heading)
        
        # Set up A matrices (observation model)
        self.A = utils.obj_array(2)  # Two observation modalities
        
        # Distance observation modality
        self.A[0] = np.zeros((num_distance_states, num_distance_states, num_heading_states))
        
        # Fill with appropriate probabilities
        for s_dist in range(num_distance_states):
            for s_head in range(num_heading_states):
                for o_dist in range(num_distance_states):
                    # Distance between state and observation
                    dist = self.state_values[s_dist] - self.state_values[o_dist]
                    # Gaussian probability density
                    self.A[0][o_dist, s_dist, s_head] = np.exp(-(dist**2) / (2 * self.noise_var))
                
                # Normalize across observations
                self.A[0][:, s_dist, s_head] = utils.norm_dist(self.A[0][:, s_dist, s_head])
        
        # Heading observation modality
        self.A[1] = np.zeros((num_heading_states, num_distance_states, num_heading_states))
        
        # Fill with appropriate probabilities
        for s_dist in range(num_distance_states):
            for s_head in range(num_heading_states):
                for o_head in range(num_heading_states):
                    # Distance between state and observation
                    dist = self.heading_values[s_head] - self.heading_values[o_head]
                    # Gaussian probability density
                    self.A[1][o_head, s_dist, s_head] = np.exp(-(dist**2) / (2 * self.heading_noise_var))
                
                # Normalize across observations
                self.A[1][:, s_dist, s_head] = utils.norm_dist(self.A[1][:, s_dist, s_head])
        
        # Set up B matrices (transition model) - one for each state factor
        self.B = utils.obj_array(num_factors)
        
        # Transition model for distance factor
        # Format: B[f] has shape (num_states_f, num_states_f, num_control_states_f)
        # for each control factor f
        self.B[0] = np.zeros((num_distance_states, num_distance_states, self.num_actions))
        
        # Define strength of each acceleration action
        action_strengths = [5, 4, 2, 1, 0, -1, -2, -4, -5]  # Relative position change
        
        # Fill B matrix for distance transitions
        for a_acc in range(self.num_actions):
            strength = action_strengths[a_acc]
            
            for s_dist in range(num_distance_states):
                # Target idx changes based on action strength
                target_idx = max(0, min(s_dist + strength, num_distance_states - 1))
                
                # Create a probability distribution
                probs = np.zeros(num_distance_states)
                
                # Create a focused distribution
                width = 0.8  # Controls how focused the distribution is
                for i in range(num_distance_states):
                    dist = abs(i - target_idx)
                    probs[i] = np.exp(-(dist**2) / (2 * width))
                
                # Normalize and assign
                self.B[0][:, s_dist, a_acc] = utils.norm_dist(probs)
        
        # Transition model for heading factor
        self.B[1] = np.zeros((num_heading_states, num_heading_states, self.num_steering_actions))
        
        # Define base steering strengths
        base_steering_strengths = [-7, -5, -3, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 3, 5, 7]
        
        # Calculate path curvature
        curvature = self.calculate_path_curvature()
        
        # Adjust steering strengths based on curvature
        if abs(curvature) > 0.1:  # Significant curve
            # Amplify steering for curves
            steering_strengths = [s * 1.5 for s in base_steering_strengths]
        else:
            # Normal steering for straight sections
            steering_strengths = base_steering_strengths
        
        # Fill B matrix for heading transitions
        for a_steer in range(self.num_steering_actions):
            strength = steering_strengths[a_steer]
            
            for s_head in range(num_heading_states):
                # Target idx changes based on action strength
                target_idx = max(0, min(s_head + strength, num_heading_states - 1))
                
                # Create a probability distribution
                probs = np.zeros(num_heading_states)
                
                # Create a focused distribution
                width = 0.8  # Controls how focused the distribution is
                for i in range(num_heading_states):
                    dist = abs(i - target_idx)
                    probs[i] = np.exp(-(dist**2) / (2 * width))
                
                # Normalize and assign
                self.B[1][:, s_head, a_steer] = utils.norm_dist(probs)
        
        # Set up C matrix (preferences) - one for each observation modality
        self.C = utils.obj_array(2)
        self.C[0] = np.zeros(num_distance_states)  # For distance
        self.C[1] = np.zeros(num_heading_states)   # For heading
        
        # Set up D matrix (prior beliefs) - one for each state factor
        self.D = utils.obj_array(num_factors)
        self.D[0] = np.zeros(num_distance_states)
        self.D[0][self.current_state_idx] = 1.0
        self.D[1] = np.zeros(num_heading_states)
        self.D[1][self.current_heading_idx] = 1.0e-3
        self.D[1] = utils.norm_dist(self.D[1])  # Normalize heading belief
        
        # Set preferred headway and heading
        self.set_preferred_headway(self.target_headway, self.dip_factor)
        self.set_preferred_heading(0.0)  # Prefer aligned with lead vehicle
        
        # Create PyMDP agent with multiple control factors
        self.agent = Agent(
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            control_fac_idx=[0, 1],
            policy_len=policy_len,
            inference_algo='VANILLA',
            action_selection='deterministic'
        )

    def _find_closest_state_idx(self, value):
        """Find closest state index for a given value"""
        return np.argmin(np.abs(self.state_values - value))
    
    def _find_closest_heading_idx(self, heading_value):
        """Find closest heading index for a given value"""
        if heading_value is None:
            return self.num_heading_states // 2  # Return middle index as default
        
        return np.argmin(np.abs(self.heading_values - heading_value))
    
    def set_preferred_headway(self, target_headway, dip_factor=1.5):
        """Set preferred headway distance with adaptive dipping"""
        # Find closest state index
        target_idx = self._find_closest_state_idx(target_headway)
        
        # Create preference distribution with peak at target headway
        preferences = np.zeros(self.num_states[0])
        
        # Adjust gradient based on mode
        gradient_factor = 2.0
        if self.catchup_mode:
            gradient_factor = 1.5  # Less steep gradient for more focus on closing gap
        
        # Create asymmetric preference gradient - penalize being too far more than too close
        for i in range(self.num_states[0]):
            dist = self.state_values[i] - self.state_values[target_idx]
            
            # Asymmetric quadratic preference
            if dist > 0:  # Too far away - stronger penalty
                preferences[i] = -gradient_factor * 3.0 * (dist**2)
            else:  # Too close - normal penalty
                preferences[i] = -gradient_factor * 2.0 * (dist**2)
        
        # Safety: More gradual penalties for getting too close
        for i in range(self.num_states[0]):
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
            
            for i in range(self.num_states[0]):
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

    def set_preferred_heading(self, target_heading):
        """Set preferred heading to match lead vehicle with improved responsiveness"""
        # Find closest state index
        target_idx = self._find_closest_heading_idx(target_heading)
        
        # Create preference distribution with peak at target heading
        preferences = np.zeros(self.num_heading_states)
        
        # Create sharper preference gradient for better lead vehicle tracking
        scaling_factor = 10.0  # Increased from 2.0 for more responsive heading matching
        for i in range(self.num_heading_states):
            dist = self.heading_values[i] - self.heading_values[target_idx]
            preferences[i] = -scaling_factor * (dist**2)  # Quadratic preference
        
        # Add additional penalty for extreme heading differences
        for i in range(self.num_heading_states):
            if abs(self.heading_values[i]) > math.pi/3:  # For large heading differences
                preferences[i] -= 2.0  # Additional penalty
        
        # Set as preference
        self.C[1] = preferences
        if hasattr(self, 'agent'):
            self.agent.C = self.C
    
    def get_path_following_target(self, ego_loc, look_ahead_distance=10.0):
        """Get target point on path that follows curvature"""
        if len(self.path_points) < 2:
            # If we don't have enough path points yet, return the lead vehicle's position and heading
            lead_transform = self.lead_vehicle.get_transform()
            lead_loc = lead_transform.location
            lead_yaw = math.radians(lead_transform.rotation.yaw)
            return lead_loc, lead_yaw
        
        # Find closest point on path
        min_dist = float('inf')
        closest_idx = 0
        
        for i, waypoint in enumerate(self.path_points):
            dist = math.sqrt(
                (ego_loc.x - waypoint['position'].x) ** 2 + 
                (ego_loc.y - waypoint['position'].y) ** 2
            )
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Look ahead along the path
        target_idx = closest_idx
        accumulated_distance = 0.0
        
        # Start from the closest point and move forward along path
        for i in range(closest_idx, len(self.path_points) - 1):
            current = self.path_points[i]['position']
            next_point = self.path_points[i + 1]['position']
            
            # Calculate distance to next point
            segment_distance = math.sqrt(
                (next_point.x - current.x) ** 2 + 
                (next_point.y - current.y) ** 2
            )
            
            accumulated_distance += segment_distance
            
            if accumulated_distance >= look_ahead_distance:
                target_idx = i + 1
                break
        
        # Get target waypoint
        target_waypoint = self.path_points[target_idx]
        
        return target_waypoint['position'], target_waypoint['heading']

    def calculate_path_curvature(self, window_size=5):
        """Calculate local path curvature using recent waypoints"""
        if len(self.path_points) < window_size:
            return 0.0
        
        # Get recent waypoints
        recent_points = list(self.path_points)[-window_size:]
        
        # Calculate angles between consecutive segments
        angles = []
        for i in range(len(recent_points) - 2):
            p1 = recent_points[i]['position']
            p2 = recent_points[i + 1]['position']
            p3 = recent_points[i + 2]['position']
            
            # Calculate vectors
            v1 = carla.Vector3D(p2.x - p1.x, p2.y - p1.y, 0)
            v2 = carla.Vector3D(p3.x - p2.x, p3.y - p2.y, 0)
            
            # Calculate angle between vectors
            angle = math.atan2(v2.y, v2.x) - math.atan2(v1.y, v1.x)
            angle = (angle + math.pi) % (2 * math.pi) - math.pi
            angles.append(angle)
        
        # Return average curvature
        return np.mean(angles) if angles else 0.0

    def update_state(self):
        """Update vehicle states from CARLA"""
        # Get vehicle transforms and velocities
        ego_transform = self.ego_vehicle.get_transform()
        lead_transform = self.lead_vehicle.get_transform()
        
        # Get velocity vectors
        ego_velocity_vec = self.ego_vehicle.get_velocity()
        lead_velocity_vec = self.lead_vehicle.get_velocity()
        
        # Extract yaw angles (in radians)
        ego_yaw = math.radians(ego_transform.rotation.yaw)
        lead_yaw = math.radians(lead_transform.rotation.yaw)
        
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
        
        # Initialize path tracking if not already done
        if not hasattr(self, 'path_points'):
            self.path_points = deque(maxlen=100)  # Store recent lead vehicle positions
        
        # Store current position and heading as a waypoint
        current_waypoint = {
            'position': lead_loc,
            'heading': lead_yaw,
            'timestamp': time.time()
        }
        self.path_points.append(current_waypoint)
        
        # Get path-following target
        target_pos, target_heading = self.get_path_following_target(ego_loc)
        
        # Calculate desired heading based on path following
        if target_pos is not None:
            # Calculate vector to target point
            to_target = carla.Vector3D(
                target_pos.x - ego_loc.x,
                target_pos.y - ego_loc.y,
                0  # Ignore height
            )
            
            # Calculate desired heading from vector to target
            desired_heading = math.atan2(to_target.y, to_target.x)
            
            # Calculate heading error
            self.heading_error = (desired_heading - ego_yaw + math.pi) % (2 * math.pi) - math.pi
            
            # Use path heading for relative heading
            self.relative_heading = self.heading_error
        else:
            # Fallback to relative heading if no path target
            self.relative_heading = (lead_yaw - ego_yaw + math.pi) % (2 * math.pi) - math.pi
            self.heading_error = self.relative_heading
        
        # Update current state indices
        self.current_state_idx = self._find_closest_state_idx(self.relative_dist)
        self.current_heading_idx = self._find_closest_heading_idx(self.relative_heading)
        
        # Track headway error
        headway_error = self.relative_dist - self.target_headway
        self.headway_error_history.append(headway_error)
        
        # Get waypoint for lead vehicle's current position
        waypoint = self.map.get_waypoint(lead_loc)
        
        # Project a path from ego vehicle toward lead vehicle
        ego_waypoint = self.map.get_waypoint(ego_loc)
        
        # Calculate lane centering data
        lane_width = ego_waypoint.lane_width
        lane_center = ego_waypoint.transform.location
        
        # Vector from ego to lane center
        to_center = carla.Vector3D(
            lane_center.x - ego_loc.x,
            lane_center.y - ego_loc.y,
            0  # Ignore height differences
        )
        
        # Project onto right vector to get deviation from center
        ego_right = ego_transform.get_right_vector()
        center_deviation = to_center.x * ego_right.x + to_center.y * ego_right.y
        
        # Store center deviation history for smoother control
        if not hasattr(self, 'center_deviation_history'):
            self.center_deviation_history = deque(maxlen=10)
        self.center_deviation_history.append(center_deviation)
        
        # Calculate lateral deviation from lead vehicle's path
        lead_lane_id = waypoint.lane_id
        ego_lane_id = ego_waypoint.lane_id
        
        if lead_lane_id == ego_lane_id:
            # Same lane - calculate lateral offset from lane center
            ego_lateral_offset = center_deviation
            self.lateral_error = ego_lateral_offset
        else:
            # Different lanes - set target to move toward lead vehicle's lane
            # Calculate signed lateral distance to lead vehicle's lane
            direction_to_lead = carla.Vector3D(
                lead_loc.x - ego_loc.x,
                lead_loc.y - ego_loc.y,
                0  # Ignore height differences
            )
            
            # Get lateral component
            self.lateral_error = direction_to_lead.x * ego_right.x + direction_to_lead.y * ego_right.y
        
        # Store history
        self.lateral_error_history.append(self.lateral_error)

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
    
    def analyze_lane_relationship(self):
        """Analyze comprehensive lane relationship between vehicles including lane centering data"""
        # Get current locations and waypoints
        ego_loc = self.ego_vehicle.get_transform().location
        lead_loc = self.lead_vehicle.get_transform().location
        
        ego_waypoint = self.map.get_waypoint(ego_loc)
        lead_waypoint = self.map.get_waypoint(lead_loc)
        
        # Basic lane check
        same_lane = ego_waypoint.lane_id == lead_waypoint.lane_id
        
        # Get road ID to check if on same road
        same_road = ego_waypoint.road_id == lead_waypoint.road_id
        
        # Calculate lateral distance between vehicles
        ego_transform = self.ego_vehicle.get_transform()
        ego_right = ego_transform.get_right_vector()
        
        # Vector from ego to lead
        to_lead = carla.Vector3D(
            lead_loc.x - ego_loc.x,
            lead_loc.y - ego_loc.y,
            0  # Ignore height
        )
        
        # Project onto right vector to get lateral offset
        lateral_distance = to_lead.x * ego_right.x + to_lead.y * ego_right.y
        
        # Check if lead is ahead of ego (positive distance is ahead)
        ego_forward = ego_transform.get_forward_vector()
        longitudinal_distance = to_lead.x * ego_forward.x + to_lead.y * ego_forward.y
        
        # Check if lane change is possible in either direction
        left_lane_change_possible = ego_waypoint.get_left_lane() is not None
        right_lane_change_possible = ego_waypoint.get_right_lane() is not None
        
        # Calculate lane centering data - distance from center of current lane
        lane_width = ego_waypoint.lane_width
        
        # Get vector from vehicle to lane center at current location
        lane_center = ego_waypoint.transform.location
        
        # Vector from ego to lane center
        to_center = carla.Vector3D(
            lane_center.x - ego_loc.x,
            lane_center.y - ego_loc.y,
            0  # Ignore height
        )
        
        # Project onto right vector to get deviation from center
        # Positive means vehicle is to the left of center, negative means to the right
        center_deviation = to_center.x * ego_right.x + to_center.y * ego_right.y
        
        # Normalize deviation as percentage of lane width (-1.0 to 1.0)
        # where 0 is perfectly centered, -1 is right edge, 1 is left edge
        normalized_deviation = (2 * center_deviation) / lane_width
        
        return {
            "same_lane": same_lane,
            "same_road": same_road,
            "lateral_distance": lateral_distance,
            "longitudinal_distance": longitudinal_distance,
            "left_lane_available": left_lane_change_possible,
            "right_lane_available": right_lane_change_possible,
            "ego_lane_id": ego_waypoint.lane_id,
            "lead_lane_id": lead_waypoint.lane_id,
            "lane_width": lane_width,
            "center_deviation": center_deviation,
            "normalized_deviation": normalized_deviation
        }

    def step(self):
        """Perform one step of adaptive headway control with path following"""
        # Update current state
        self.update_state()
        
        # Periodically adapt parameters if using adaptive control
        if self.use_adaptive and len(self.headway_error_history) % 5 == 0:
            self.adapt_parameters()
        
        # Generate observation for active inference
        obs_idx = self._find_closest_state_idx(self.relative_dist)
        heading_obs_idx = self._find_closest_heading_idx(self.relative_heading)
        obs = [obs_idx, heading_obs_idx]

        # Update D with current state belief
        qs = self.agent.infer_states(obs)
        self.D[0] = qs[0]
        self.D[1] = qs[1]
        self.agent.D = self.D
        
        # Get policy and expected free energies
        q_pi, efe = self.agent.infer_policies()
        
        # Sample action using active inference
        action = self.agent.sample_action()
        action_idx = action[0]        # Acceleration action
        steering_action_idx = action[1]  # Steering action
        
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
        
        print(f'Ego Acceleration: {ego_acceleration:.2f}')
        print(f"Steering Action Index: {steering_action_idx}")
        
        steering_map = {}
        for i in range(self.num_steering_actions):
            # Create a range from -0.7 to 0.7 (or whatever max steering angle you want)
            # This creates a linear distribution of steering angles
            steering_map[i] = -0.7 + (1.4 * i / (self.num_steering_actions - 1))
        
        steering_angle = steering_map[steering_action_idx]
        
        # Calculate path curvature
        curvature = self.calculate_path_curvature()
        
        # Adjust steering based on path curvature
        if abs(curvature) > 0.1:  # Significant curve
            steering_smoothing_factor = 0.02  # More responsive in curves
        else:
            steering_smoothing_factor = 0.05  # Smoother on straights
        
        steering_angle = (steering_smoothing_factor * self.previous_steering +
                         (1 - steering_smoothing_factor) * steering_angle)
        
        # Add lane keeping check
        lane_status = self.analyze_lane_relationship()
        
        # Adjust control based on lane status
        if not lane_status["same_lane"] and lane_status["same_road"]:
            # Need to change lanes
            lateral_distance = lane_status["lateral_distance"]
            
            # Adjust steering to move toward lead vehicle's lane
            lane_correction = max(-0.3, min(0.3, -lateral_distance * 0.05))
            steering_angle += lane_correction
            
            print(f"Lane correction: {lane_correction:.2f}, lateral distance: {lateral_distance:.2f}")
        else:
            # We're in the same lane - apply lane centering
            if "normalized_deviation" in lane_status:
                # Apply lane centering correction
                centering_factor = 0.08  # Adjust this value as needed for responsiveness
                center_correction = -lane_status["normalized_deviation"] * centering_factor
                
                # Apply smoother correction when close to center
                if abs(lane_status["normalized_deviation"]) < 0.2:  # If within 20% of center
                    center_correction *= 0.5  # Apply gentler correction to avoid oscillation
                
                # Limit the correction magnitude
                center_correction = max(-0.2, min(0.2, center_correction))
                
                # Apply the centering correction
                steering_angle += center_correction
                
                print(f"Lane centering: {center_correction:.3f}, deviation: {lane_status['normalized_deviation']:.3f}")
        
        # Save for next iteration
        self.previous_acc = ego_acceleration
        self.previous_steering = -steering_angle
        
        control = carla.VehicleControl()
        print(f"Steering Angle: {steering_angle:.2f}")
        
        if ego_acceleration >= 0 or abs(steering_angle) > 0.1:
            # For acceleration, apply throttle (normalized to 0-1 range)
            # Assuming max acceleration of 3 m/s^2 for full throttle
            control.throttle = min(1.0, ego_acceleration / 3.0)
            control.brake = 0.0
        else:
            # For deceleration, apply brake (normalized to 0-1 range)
            # Assuming max deceleration of -5 m/s^2 for full brake
            control.throttle = 0.0
            control.brake = min(1.0, -ego_acceleration / 5.0)

        # Set steering
        control.steer = -steering_angle
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
            'catchup_mode': self.catchup_mode if self.use_adaptive else None,
            'lane_centering': lane_status.get('normalized_deviation', 0)
        }