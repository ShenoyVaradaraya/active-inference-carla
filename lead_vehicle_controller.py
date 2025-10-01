"""
Lead Vehicle Controller for CARLA
--------------------------------
Controls the lead vehicle behavior in the CARLA environment.
"""

import carla
import math
import random
import numpy as np


class LeadVehicleController:
    def __init__(self, lead_vehicle, behavior='constant', time_step=0.05):
        """
        Controller for the lead vehicle in CARLA with various behavior patterns
        
        Args:
            lead_vehicle: CARLA actor representing the lead vehicle
            behavior: Behavior pattern ('constant', 'varying', 'stopping', 'stop_and_go')
            time_step: Simulation time step in seconds
        """
        self.lead_vehicle = lead_vehicle
        self.behavior = behavior
        self.time_step = time_step
        
        # Initialize state
        self.step_counter = 0
        self.lead_velocity = 0.0
        self.lead_acceleration = 0.0
        self.previous_velocity = 0.0
        
        # Set initial values based on behavior
        if behavior == 'constant':
            self.target_velocity = 3.0  # m/s (36 km/h)
        elif behavior == 'varying':
            self.target_velocity = 5.0
        elif behavior == 'stopping':
            self.target_velocity = 5.0
        elif behavior == 'stop_and_go':
            self.target_velocity = 0.0  # Start stopped
        else:
            raise ValueError(f"Unknown behavior: {behavior}")
        
        # Apply initial velocity
        self.set_velocity(self.target_velocity)
    
    def set_velocity(self, velocity):
        """Set vehicle velocity"""
        velocity_vector = self.lead_vehicle.get_transform().get_forward_vector()
        velocity_vector = velocity_vector * velocity
        self.lead_vehicle.set_target_velocity(velocity_vector)
    
    def update_state(self):
        """Update current state"""
        velocity_vec = self.lead_vehicle.get_velocity()
        self.previous_velocity = self.lead_velocity
        self.lead_velocity = math.sqrt(
            velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2)
        self.lead_acceleration = (self.lead_velocity - self.previous_velocity) / self.time_step
    
    def constant_behavior(self):
        """Maintain constant speed with minor variations"""
        self.target_velocity = 5.0 + random.normalvariate(0, 0.1)
        self.set_velocity(self.target_velocity)
    
    def varying_behavior(self):
        """Gradually changing speed with occasional sharp changes"""
        # Calculate time-based sinusoidal variation
        t = self.step_counter * self.time_step
        variation = math.sin(t * 0.1) * 2.0 + math.sin(t * 0.05) * 1.0
        
        # Add occasional sudden changes
        if random.random() < 0.01:  # 1% chance each step
            variation += random.choice([-3.0, 3.0])
        
        self.target_velocity = 5.0 + variation
        self.target_velocity = max(0.0, min(20.0, self.target_velocity))  # Limit range
        self.set_velocity(self.target_velocity)
    
    def stopping_behavior(self):
        """Gradually slow down to a stop"""
        if self.lead_velocity > 0.1:
            self.target_velocity = max(0.0, self.lead_velocity - 0.1)
        else:
            self.target_velocity = 0.0
        self.set_velocity(self.target_velocity)
    
    def stop_and_go_behavior(self):
        """Create stop and go traffic pattern"""
        # Cycle through: accelerate -> cruise -> decelerate -> stop -> wait -> repeat
        cycle_time = 200  # steps per complete cycle (10 seconds at 0.05s per step)
        phase = self.step_counter % cycle_time
        
        if phase < 40:  # Acceleration phase (2 seconds)
            self.target_velocity = min(5.0, self.lead_velocity + 0.25)
        elif phase < 80:  # Cruising phase (2 seconds)
            self.target_velocity = 5.0 + random.normalvariate(0, 0.1)
        elif phase < 120:  # Deceleration phase (2 seconds)
            self.target_velocity = max(0.0, self.lead_velocity - 0.25)
        else:  # Stopped phase (4 seconds)
            self.target_velocity = 0.0
        
        self.set_velocity(self.target_velocity)
    
    def step(self):
        """Perform one controller step"""
        self.step_counter += 1
        self.update_state()
        
        # Apply behavior
        if self.behavior == 'constant':
            self.constant_behavior()
        elif self.behavior == 'varying':
            self.varying_behavior()
        elif self.behavior == 'stopping':
            self.stopping_behavior()
        elif self.behavior == 'stop_and_go':
            self.stop_and_go_behavior()
        
        return {
            'velocity': self.lead_velocity,
            'acceleration': self.lead_acceleration,
            'target_velocity': self.target_velocity
        }