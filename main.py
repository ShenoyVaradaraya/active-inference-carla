#!/usr/bin/env python

"""
Adaptive Headway Active Inference Controller for CARLA
-----------------------------------------------------
This script implements an adaptive headway maintenance system using active inference
in the CARLA simulation environment.
"""

import glob
import os
import sys
import time
import math
import numpy as np
import pygame
import random
import argparse
import logging
import csv
from datetime import datetime

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from adaptive_aif_with_steering import AdaptiveHeadwayController
from visualization import HeadwayVisualization, visualize_lead_vehicle_path
from lead_vehicle_controller import LeadVehicleController

# Global parameters
SECONDS_PER_STEP = 0.1  # simulation time step
TARGET_HEADWAY = 15.0  # target following distance in meters
path_update_interval = int(2.0 / SECONDS_PER_STEP)

def main():
    """Main function to run the CARLA simulation with adaptive headway controller"""
    argparser = argparse.ArgumentParser(
        description='Adaptive Headway Controller Demo')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--scenario',
        default='town03',
        help='CARLA map to use (default: Town01)')
    argparser.add_argument(
        '--adaptive',
        action='store_true',
        help='Use adaptive controller (default: False)')
    argparser.add_argument(
        '--duration',
        default=200,
        type=int,
        help='Simulation duration in seconds (default: 120)')
    
    args = argparser.parse_args()
    
    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('Listening to server %s:%s', args.host, args.port)
    
    # Setup data collection
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = f'headway_data_autopilot_{timestamp}.csv'
    
    try:
        # Connect to CARLA server
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        
        # Load desired map
        world = client.load_world(f'Town{args.scenario[-2:]}')
        world.set_weather(carla.WeatherParameters.ClearNoon)
        
        # Set synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = SECONDS_PER_STEP
        world.apply_settings(settings)
        
        # Get traffic manager
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)
        
        # Create blueprints for vehicles
        blueprint_library = world.get_blueprint_library()
        
        # Ego vehicle (controllable car)
        ego_bp = blueprint_library.filter('model3')[0]
        ego_bp.set_attribute('color', '0,0,255')  # Blue
        
        # Lead vehicle (car to follow)
        lead_bp = blueprint_library.filter('prius')[0]
        lead_bp.set_attribute('color', '255,0,0')  # Red
        
        # Get a spawn point
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[104] # 104 for town03
        
        # Spawn lead vehicle ahead of ego vehicle
        lead_spawn = carla.Transform(
            spawn_point.location + carla.Location(x=-TARGET_HEADWAY), # x for town03
            spawn_point.rotation)
        lead_vehicle = world.spawn_actor(lead_bp, lead_spawn)
        ego_vehicle = world.spawn_actor(ego_bp, spawn_point)
        # Enable autopilot for lead vehicle
        lead_vehicle.set_autopilot(True)
        # # Configure traffic manager for lead vehicle (optional)
        # traffic_manager.vehicle_percentage_speed_difference(lead_vehicle, 0.1)  # Drive at speed limit
        # traffic_manager.distance_to_leading_vehicle(lead_vehicle, 2.0)  # Set distance to vehicles ahead
        traffic_manager.ignore_lights_percentage(lead_vehicle, 100)  # Obey traffic lights
        
        # Spawn ego vehicle
        
        # lead_controller = LeadVehicleController(lead_vehicle,behavior='constant', time_step=SECONDS_PER_STEP)
        # Set up adaptive headway controller
        controller = AdaptiveHeadwayController(
            ego_vehicle, 
            lead_vehicle, 
            target_headway=TARGET_HEADWAY,
            adaptive=args.adaptive,
            time_step=SECONDS_PER_STEP)
        
        # Set up visualization
        visualizer = HeadwayVisualization(world, ego_vehicle, lead_vehicle, record=True)

        
        # Set up spectator to follow vehicles
        spectator = world.get_spectator()
        
        # Main simulation loop
        steps = 0
        max_steps = args.duration / SECONDS_PER_STEP
        
        # CSV header
        with open(data_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Step', 'Time', 'Headway', 'EgoVelocity', 'LeadVelocity',
                'EgoAcceleration', 'LeadAcceleration', 'SmoothingFactor',
                'AccScaling', 'DipFactor', 'CatchupMode'
            ])
        
        logging.info("Starting simulation loop...")
        
        while steps < max_steps:
            # Advance simulation
            world.tick()
            # lead_steer = lead_vehicle.get_control().steer
            lead_speed = lead_vehicle.get_velocity()
            lead_speed = math.sqrt(lead_speed.x**2 + lead_speed.y**2 + lead_speed.z**2)
            forward_vector = lead_vehicle.get_transform().get_forward_vector()
            velocity = min(lead_speed, 5.0)
            lead_vehicle.set_target_velocity(carla.Vector3D(
                forward_vector.x * velocity,
                forward_vector.y * velocity,
                forward_vector.z * velocity))
            # lead_controller.step()
            # Update ego vehicle
            controller_output = controller.step()
            
            # Update spectator to follow vehicles
            ego_transform = ego_vehicle.get_transform()
            spectator_transform = carla.Transform(
                ego_transform.location + carla.Location(z=30, x=-10),
                carla.Rotation(pitch=-60))
            spectator.set_transform(spectator_transform)
            
            # Update visualization
            visualizer.update(controller_output)
            
            # Record data
            with open(data_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    steps,
                    steps * SECONDS_PER_STEP,
                    controller_output['relative_dist'],
                    controller_output['ego_velocity'],
                    controller_output['lead_velocity'],
                    controller_output['ego_acceleration'],
                    controller_output['lead_acceleration'],
                    controller_output.get('smoothing_factor', 'N/A'),
                    controller_output.get('acc_scaling', 'N/A'),
                    controller_output.get('dip_factor', 'N/A'),
                    controller_output.get('catchup_mode', 'N/A')
                ])
            
            steps += 1
            if steps % path_update_interval == 0:
                visualize_lead_vehicle_path(world, lead_vehicle, duration=2.5)
            # Print status every second
            # if steps % int(1.0 / SECONDS_PER_STEP) == 0:
                # logging.info(
                #     f"Step {steps}/{max_steps}, "
                #     f"Headway: {controller_output['relative_dist']:.2f}m, "
                #     f"Ego v: {controller_output['ego_velocity']:.2f}m/s, "
                #     f"Lead v: {controller_output['lead_velocity']:.2f}m/s"
                # )
        
        logging.info("Simulation complete.")
        visualizer.toggle_recording()

    except KeyboardInterrupt:
        logging.info("User interrupted simulation.")
    
    finally:
        # Clean up
        logging.info("Cleaning up...")
        if 'world' in locals():
            if 'ego_vehicle' in locals():
                ego_vehicle.destroy()
            if 'lead_vehicle' in locals():
                lead_vehicle.destroy()
            
            # Disable synchronous mode
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
        
        logging.info(f"Data saved to {data_file}")


if __name__ == '__main__':
    main()