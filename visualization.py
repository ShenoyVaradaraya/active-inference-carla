"""
Visualization for CARLA Headway Control with Video Recording
--------------------------------------
Provides visualization of the headway maintenance system in the CARLA environment,
with added functionality to record the visualization as a video file.
"""

import carla
import pygame
import numpy as np
import math
import cv2
import os
from datetime import datetime

class HeadwayVisualization:
    def __init__(self, world, ego_vehicle, lead_vehicle, width=800, height=600, record=False):
        """
        Initialize visualization for CARLA headway control
        
        Args:
            world: CARLA world instance
            ego_vehicle: CARLA actor for ego vehicle
            lead_vehicle: CARLA actor for lead vehicle
            width: Display width for main window
            height: Display height for main window
            record: Set to True to enable video recording
        """
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.lead_vehicle = lead_vehicle
        
        # Set up combined visualization (main view + indicators)
        self.setup_combined_visualization(width, height, 400, height)
        
        # Data for plotting
        self.history_length = 100
        self.headway_history = []
        self.velocity_history = {'ego': [], 'lead': []}
        self.time_history = []
        
        # Colors
        self.colors = {
            'background': (0, 0, 0),
            'text': (255, 255, 255),
            'headway': (0, 255, 0),
            'target': (255, 255, 0),
            'ego_velocity': (0, 0, 255),
            'lead_velocity': (255, 0, 0)
        }
        
        # Video recording setup
        self.is_recording = record
        self.video_writer = None
        self.frame_count = 0
        
        if self.is_recording:
            self._setup_video_recorder()
    
    def setup_combined_visualization(self, main_width=800, main_height=600, indicator_width=400, indicator_height=600):
        """Set up a combined visualization with both camera view and indicators"""
        # Calculate combined dimensions
        self.main_width = main_width
        self.main_height = main_height
        self.indicator_width = indicator_width
        self.indicator_height = indicator_height
        
        # Combined display will have both windows side by side
        self.width = main_width + indicator_width
        self.height = max(main_height, indicator_height)
        
        # Initialize pygame for visualization
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode((self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Adaptive Headway Controller")
        self.font = pygame.font.SysFont('Arial', 20)
        self.clock = pygame.time.Clock()
        
        # Create separate surfaces for main view and indicators
        self.main_surface = pygame.Surface((main_width, main_height))
        self.indicator_surface = pygame.Surface((indicator_width, indicator_height))
        
        # Set up camera sensor for main view
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(main_width))
        camera_bp.set_attribute('image_size_y', str(main_height))
        camera_bp.set_attribute('fov', '110')
        
        # Camera transform relative to ego vehicle
        camera_transform = carla.Transform(
            carla.Location(x=-8, z=6),  # Position behind and above ego vehicle
            carla.Rotation(pitch=-15))  # Look slightly down
        
        # Attach camera to ego vehicle
        self.camera = self.world.spawn_actor(
            camera_bp, camera_transform, attach_to=self.ego_vehicle)
        
        # Set up camera callback
        self.camera.listen(lambda image: self._process_image(image))
    
    def _setup_video_recorder(self):
        """Set up video recorder"""
        # Create output directory if it doesn't exist
        output_dir = 'output_videos'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Generate a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = os.path.join(output_dir, f'headway_recording_{timestamp}.mp4')
        
        # Define the codec and create VideoWriter object - now using combined size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec for MP4 files
        fps = 30  # Frames per second
        self.video_writer = cv2.VideoWriter(
            video_filename, 
            fourcc, 
            fps, 
            (self.width, self.height)  # Combined width of both surfaces
        )
        
        print(f"Recording video to: {video_filename}")
    
    def _process_image(self, image):
        """Process camera image"""
        # Convert CARLA raw image to pygame surface
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]  # Convert from RGBA to RGB
        self.main_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    
    def update(self, controller_output):
        """Update visualization with new controller output"""
        if self.main_surface is None:
            return
        
        # Fill indicator surface with dark background
        self.indicator_surface.fill((20, 20, 20))
        
        # Calculate positions for elements to create a continuous look
        hud_height = 120
        headway_height = 50
        path_viz_height = 200
        plots_height = 140
        
        # Calculate positions with minimal gaps
        hud_y = 10
        headway_y = hud_y + hud_height + 5
        path_viz_y = headway_y + headway_height + 5
        plots_y = self.indicator_height - plots_height - 10
        
        # Draw HUD elements on indicator surface with calculated positions
        self._draw_hud(controller_output, self.indicator_surface, hud_y)
        self._draw_path_visualization(controller_output, self.indicator_surface, path_viz_y)
        self._draw_plots(controller_output, self.indicator_surface, plots_y)
        
        # Combine surfaces onto main display
        self.display.blit(self.main_surface, (0, 0))
        self.display.blit(self.indicator_surface, (self.main_width, 0))
        
        # Draw a divider between the two surfaces
        pygame.draw.line(self.display, (100, 100, 100), 
                        (self.main_width, 0), 
                        (self.main_width, self.height), 3)
        
        # Capture frame for video recording (now captures both surfaces)
        self._capture_frame()
            
        # Update the display
        pygame.display.flip()
        self.clock.tick(10)  # Limit to 60 FPS
    
    def _draw_hud(self, controller_output, surface, y_pos=10):
        """Draw HUD with current state information"""
        if surface is None:
            return
        
        # Draw HUD background - reduced height from 160 to 120
        hud_rect = pygame.Rect(10, y_pos, surface.get_width() - 20, 120)
        pygame.draw.rect(surface, (30, 30, 30), hud_rect)
        pygame.draw.rect(surface, (100, 100, 100), hud_rect, 2)
        
        # Use smaller font
        small_font = pygame.font.SysFont('Arial', 16)
        
        # Create a more compact layout with two columns
        left_col_x = 20
        right_col_x = surface.get_width() // 2
        
        # Draw headway information - left column
        headway_text = small_font.render(
            f"Headway: {controller_output['relative_dist']:.2f}m", 
            True, self.colors['text'])
        target_text = small_font.render(
            f"Target: {controller_output.get('target_headway', 8.0):.2f}m", 
            True, self.colors['text'])
        error_text = small_font.render(
            f"Error: {controller_output['relative_dist'] - controller_output.get('target_headway', 8.0):.2f}m", 
            True, self.colors['text'])
        
        # Draw velocity information - right column
        ego_vel_text = small_font.render(
            f"Ego Velocity: {controller_output['ego_velocity']:.2f}m/s", 
            True, self.colors['text'])
        lead_vel_text = small_font.render(
            f"Lead Velocity: {controller_output['lead_velocity']:.2f}m/s", 
            True, self.colors['text'])
        ego_acc_text = small_font.render(
            f"Ego Acceleration: {controller_output['ego_acceleration']:.2f}m/s²", 
            True, self.colors['text'])
        
        # Left column
        y_offset = y_pos + 15
        surface.blit(headway_text, (left_col_x, y_offset))
        y_offset += 20
        surface.blit(target_text, (left_col_x, y_offset))
        y_offset += 20
        surface.blit(error_text, (left_col_x, y_offset))
        
        # Right column
        y_offset = y_pos + 15
        surface.blit(ego_vel_text, (right_col_x, y_offset))
        y_offset += 20
        surface.blit(lead_vel_text, (right_col_x, y_offset))
        y_offset += 20
        surface.blit(ego_acc_text, (right_col_x, y_offset))
        
        # Add adaptive parameters if available - place in bottom rows
        if controller_output.get('smoothing_factor') is not None:
            y_offset = y_pos + 75
            
            mode_text = small_font.render(
                f"Mode: {'Catchup' if controller_output.get('catchup_mode') else 'Normal'}", 
                True, self.colors['text'])
            surface.blit(mode_text, (left_col_x, y_offset))
            
            smoothing_text = small_font.render(
                f"Smoothing: {controller_output.get('smoothing_factor', 0.0):.2f}", 
                True, self.colors['text'])
            surface.blit(smoothing_text, (right_col_x, y_offset))
            
            y_offset += 20
            
            dip_text = small_font.render(
                f"Dip: {controller_output.get('dip_factor', 0.0):.2f}", 
                True, self.colors['text'])
            surface.blit(dip_text, (left_col_x, y_offset))
            
            acc_text = small_font.render(
                f"Acc Scale: {controller_output.get('acc_scaling', 0.0):.2f}", 
                True, self.colors['text'])
            surface.blit(acc_text, (right_col_x, y_offset))
        
        # Add recording indicator
        if self.is_recording:
            recording_text = small_font.render(
                f"Recording... Frame: {self.frame_count}", 
                True, (255, 0, 0))
            surface.blit(recording_text, (surface.get_width() - 200, y_pos + 95))
    
    def _draw_headway_indicator(self, controller_output, surface, y_pos=170):
        """Draw visual indicator of headway distance"""
        if surface is None:
            return
        
        # Draw in the middle of the indicator surface - reduced height
        width, height = surface.get_size()
        center_x = width // 2
        indicator_y = y_pos + 25  # Center within the allocated space
        
        # Create a box for the headway indicator - reduced height from 80 to 50
        indicator_height = 50
        indicator_rect = pygame.Rect(10, indicator_y - indicator_height // 2, 
                               width - 20, indicator_height)
        pygame.draw.rect(surface, (40, 40, 40), indicator_rect)
        pygame.draw.rect(surface, (100, 100, 100), indicator_rect, 2)
        
        # Smaller font
        small_font = pygame.font.SysFont('Arial', 14)
        
        # Title for the indicator
        title = small_font.render("Headway Visualization", True, (255, 255, 255))
        surface.blit(title, (center_x - title.get_width() // 2, y_pos))
        
        # Draw ego vehicle on left - smaller
        ego_rect = pygame.Rect(20, indicator_y - 10, 30, 20)
        pygame.draw.rect(surface, self.colors['ego_velocity'], ego_rect)
        
        # Calculate scale factor - how many pixels per meter
        # We want to fit the target headway within about 60% of the indicator width
        scale_factor = (indicator_rect.width * 0.6) / controller_output.get('target_headway', 5.0)
        
        # Draw target headway marker
        target_x = 20 + 30 + controller_output.get('target_headway', 5.0) * scale_factor
        pygame.draw.line(surface, self.colors['target'], 
                   (target_x, indicator_y - 15), 
                   (target_x, indicator_y + 15), 
                   2)
        
        # Draw current headway line
        pygame.draw.line(surface, self.colors['headway'], 
                   (50, indicator_y), 
                   (50 + controller_output['relative_dist'] * scale_factor, indicator_y), 
                   2)
        
        # Draw lead vehicle at current headway distance - smaller
        lead_rect = pygame.Rect(
            20 + 30 + controller_output['relative_dist'] * scale_factor, 
            indicator_y - 10, 
            30, 
            20
        )
        pygame.draw.rect(surface, self.colors['lead_velocity'], lead_rect)
        
        # Add compact labels
        current_label = small_font.render(f"{controller_output['relative_dist']:.1f}m", 
                                  True, self.colors['headway'])
        target_label = small_font.render(f"Target: {controller_output.get('target_headway', 12.0)}m", 
                                  True, self.colors['target'])
        
        # Position labels more compactly
        surface.blit(target_label, (target_x - 40, indicator_y - 30))
        surface.blit(current_label, (indicator_rect.x + indicator_rect.width - 60, indicator_y - 20))
    
    def _draw_path_visualization(self, controller_output, surface, y_pos=240):
        """Draw path visualization"""
        if surface is None:
            return
        
        # Create a larger, wider top-down view area
        view_width = min(surface.get_width() - 20, 380)  # Increased width
        view_height = 200  # Increased height
        margin = 10
        
        # Adjusted position to account for smaller HUD and headway indicator
        view_rect = pygame.Rect(
            surface.get_width() // 2 - view_width // 2, 
            y_pos,  # Use the provided y position
            view_width, 
            view_height
        )
        
        # Title for the path visualization
        small_font = pygame.font.SysFont('Arial', 14)
        title = small_font.render("Path Visualization", True, (255, 255, 255))
        surface.blit(title, (surface.get_width() // 2 - title.get_width() // 2, y_pos - 18))
        
        pygame.draw.rect(surface, (30, 30, 30), view_rect)
        pygame.draw.rect(surface, (100, 100, 100), view_rect, 2)
        
        # Center of view will be ego vehicle
        center_x = view_rect.x + view_rect.width // 2
        center_y = view_rect.y + view_rect.height // 2
        
        # Draw ego vehicle - made larger for better visibility
        pygame.draw.circle(surface, (0, 0, 255), (center_x, center_y), 9)
        
        # Calculate scale factor for visualization (pixels per meter)
        # Scale adjusted for larger view
        scale = 3.0  # Adjusted scale for larger view
        
        # Get ego and lead positions
        ego_pos = controller_output['ego_position']
        lead_pos = controller_output['lead_position']

        # Calculate relative position (lead - ego)
        rel_x = lead_pos.x - ego_pos.x
        rel_y = lead_pos.y - ego_pos.y
        
        # Draw lead vehicle position
        lead_x = center_x + int(rel_x * scale)
        lead_y = center_y + int(rel_y * scale)
        pygame.draw.circle(surface, (255, 0, 0), (lead_x, lead_y), 9)
        
        # Draw line connecting ego to lead
        pygame.draw.line(surface, (255, 255, 0), 
                (center_x, center_y), (lead_x, lead_y), 2)
        
        # Add labels with slightly larger font for better readability
        label_font = pygame.font.SysFont('Arial', 14)
        ego_label = label_font.render("Ego", True, (0, 0, 255))
        lead_label = label_font.render("Lead", True, (255, 0, 0))
        surface.blit(ego_label, (center_x + 12, center_y - 12))
        surface.blit(lead_label, (lead_x + 12, lead_y - 12))
        
        # Add distance label
        distance = ego_pos.distance(lead_pos)
        dist_label = label_font.render(f"3D Distance: {distance:.2f}m", True, (255, 255, 255))
        surface.blit(dist_label, (view_rect.x + 10, view_rect.y + view_rect.height - 25))
        
        # Add grid lines for better spatial reference
        grid_color = (60, 60, 60)
        grid_spacing = 20  # pixels
        
        
        # Add compass directions
        compass_font = pygame.font.SysFont('Arial', 16)
        margin_from_edge = 15
        
        # North - top center
        north_label = compass_font.render("N", True, (180, 180, 180))
        north_x = center_x - north_label.get_width() // 2
        north_y = view_rect.y + margin_from_edge
        surface.blit(north_label, (north_x, north_y))
        
        # East - right center
        east_label = compass_font.render("E", True, (180, 180, 180))
        east_x = view_rect.x + view_width - margin_from_edge - east_label.get_width()
        east_y = center_y - east_label.get_height() // 2
        surface.blit(east_label, (east_x, east_y))
        
        # South - bottom center
        south_label = compass_font.render("S", True, (180, 180, 180))
        south_x = center_x - south_label.get_width() // 2
        south_y = view_rect.y + view_height - margin_from_edge - south_label.get_height()
        surface.blit(south_label, (south_x, south_y))
        
        # West - left center
        west_label = compass_font.render("W", True, (180, 180, 180))
        west_x = view_rect.x + margin_from_edge
        west_y = center_y - west_label.get_height() // 2
        surface.blit(west_label, (west_x, west_y))
        
        
        
    
    def _draw_plots(self, controller_output, surface, y_pos=450):
        """Draw performance plots"""
        if surface is None:
            return
        
        # Update histories
        self.headway_history.append(controller_output['relative_dist'])
        self.velocity_history['ego'].append(controller_output['ego_velocity'])
        self.velocity_history['lead'].append(controller_output['lead_velocity'])
        self.time_history.append(len(self.time_history))
        
        # Limit history length
        if len(self.headway_history) > self.history_length:
            self.headway_history = self.headway_history[-self.history_length:]
            self.velocity_history['ego'] = self.velocity_history['ego'][-self.history_length:]
            self.velocity_history['lead'] = self.velocity_history['lead'][-self.history_length:]
            self.time_history = self.time_history[-self.history_length:]
        
        # Calculate plot area dimensions - reduced height
        plot_width = surface.get_width() - 20
        plot_height = 140  # Reduced from 180
        
        # Draw background for plots
        plot_rect = pygame.Rect(10, y_pos, plot_width, plot_height)
        pygame.draw.rect(surface, (50, 50, 50), plot_rect)
        pygame.draw.rect(surface, (100, 100, 100), plot_rect, 2)
        
        # Draw divider between plots
        mid_y = plot_rect.y + plot_rect.height // 2
        pygame.draw.line(surface, (100, 100, 100), 
                        (plot_rect.x, mid_y), 
                        (plot_rect.x + plot_rect.width, mid_y), 2)
        
        # Smaller font
        small_font = pygame.font.SysFont('Arial', 14)
        
        # Draw headway plot in top half
        if len(self.headway_history) > 1:
            # Calculate scaling factors
            x_scale = plot_width / self.history_length
            y_scale = (plot_height / 2 - 10) / max(15, max(self.headway_history))
            
            # Draw target headway
            target_y = mid_y - controller_output.get('target_headway', 5.0) * y_scale
            pygame.draw.line(surface, self.colors['target'], 
                           (plot_rect.x + 5, target_y), 
                           (plot_rect.x + plot_width - 5, target_y), 2)
            
            # Draw headway history
            points = [(plot_rect.x + 5 + i * x_scale, mid_y - h * y_scale) 
                     for i, h in enumerate(self.headway_history)]
            if len(points) > 1:
                pygame.draw.lines(surface, self.colors['headway'], False, points, 2)
            
            # Label
            headway_text = small_font.render(
                f"Headway: {controller_output['relative_dist']:.2f}m", 
                True, self.colors['headway'])
            surface.blit(headway_text, (plot_rect.x + 10, plot_rect.y + 5))
        
        # Draw velocity plot in bottom half
        if len(self.velocity_history['ego']) > 1:
            # Calculate scaling factors
            x_scale = plot_width / self.history_length
            max_velocity = max(
                max(self.velocity_history['ego'] + [0.1]), 
                max(self.velocity_history['lead'] + [0.1]))
            y_scale = (plot_height / 2 - 10) / max(10, max_velocity)
            
            # Draw ego velocity history
            ego_points = [(plot_rect.x + 5 + i * x_scale, 
                          plot_rect.y + plot_rect.height - 5 - v * y_scale) 
                        for i, v in enumerate(self.velocity_history['ego'])]
            if len(ego_points) > 1:
                pygame.draw.lines(surface, self.colors['ego_velocity'], False, ego_points, 2)
            
            # Draw lead velocity history
            lead_points = [(plot_rect.x + 5 + i * x_scale, 
                          plot_rect.y + plot_rect.height - 5 - v * y_scale) 
                         for i, v in enumerate(self.velocity_history['lead'])]
            if len(lead_points) > 1:
                pygame.draw.lines(surface, self.colors['lead_velocity'], False, lead_points, 2)
            
            # Labels - more compact
            ego_text = small_font.render(
                f"Ego: {controller_output['ego_velocity']:.2f}m/s", 
                True, self.colors['ego_velocity'])
            lead_text = small_font.render(
                f"Lead: {controller_output['lead_velocity']:.2f}m/s", 
                True, self.colors['lead_velocity'])
            surface.blit(ego_text, (plot_rect.x + 10, mid_y + 5))
            surface.blit(lead_text, (plot_rect.x + 120, mid_y + 5))
    
    def _capture_frame(self):
        """Capture current frame for video recording"""
        if not self.is_recording:
            return
            
        # Convert pygame display to numpy array (now captures both surfaces)
        pygame_array = pygame.surfarray.array3d(self.display)
        # Reshape array to expected dimensions (height, width, channels)
        frame = pygame_array.transpose([1, 0, 2])
        # Convert from RGB to BGR (OpenCV format)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Write frame to video
        self.video_writer.write(frame)
        self.frame_count += 1
    
    def toggle_recording(self):
        """Toggle video recording on/off"""
        if self.is_recording:
            # Stop recording
            if self.video_writer:
                self.video_writer.release()
                print(f"Recording stopped. Saved {self.frame_count} frames.")
            self.is_recording = False
            self.video_writer = None
        else:
            # Start recording
            self.is_recording = True
            self._setup_video_recorder()
            self.frame_count = 0
    
    def destroy(self):
        """Clean up resources"""
        # Finish video recording if active
        if self.is_recording and self.video_writer:
            self.video_writer.release()
            print(f"Recording finished. Saved {self.frame_count} frames.")
        
        if self.camera:
            self.camera.destroy()
        pygame.quit()
    
# Add this function to your script
def visualize_lead_vehicle_path(world, lead_vehicle, duration=10.0, path_length=100):
    """
    Visualizes the future path that the autopilot lead vehicle will take.
    
    Args:
        world: CARLA world object
        lead_vehicle: The vehicle on autopilot
        duration: How long the visualization should stay visible (seconds)
        path_length: How many waypoints to visualize ahead
    """
    # Get the map
    carla_map = world.get_map()
    
    # Get current vehicle waypoint
    current_waypoint = carla_map.get_waypoint(lead_vehicle.get_location())
    
    # Get future waypoints
    waypoints = []
    next_wp = current_waypoint
    
    for i in range(path_length):
        # Get next waypoints (we take the first option at each junction)
        next_waypoints = next_wp.next(2.0)  # 2.0 meter interval
        if not next_waypoints:
            break
        next_wp = next_waypoints[0]  # Take first path at junctions
        waypoints.append(next_wp)
    
    # Visualize waypoints using debug helper
    debug = world.debug
    for i, wp in enumerate(waypoints):
        # Color gradient from red to yellow (for distance visualization)
        r = 255
        g = min(255, int((i / len(waypoints)) * 255))
        b = 0
        
        # Draw point at waypoint location
        debug.draw_point(
            wp.transform.location + carla.Location(z=0.5),  # Slightly above ground
            size=0.1,
            color=carla.Color(r, g, b),
            life_time=duration
        )
        
        # Draw line to next waypoint if not the last one
        if i < len(waypoints) - 1:
            debug.draw_line(
                wp.transform.location + carla.Location(z=0.5),
                waypoints[i+1].transform.location + carla.Location(z=0.5),
                thickness=0.1,
                color=carla.Color(r, g, b),
                life_time=duration
            )
    
    return waypoints