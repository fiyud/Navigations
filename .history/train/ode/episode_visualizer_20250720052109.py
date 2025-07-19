import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import List, Dict, Tuple
import os
from datetime import datetime

class EpisodeVisualizer:
    def __init__(self, save_dir: str = "./visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.reset()
    
    def reset(self):
        """Reset for new episode"""
        self.agent_positions = []
        self.goal_positions = []
        self.goal_images = []
        self.current_images = []
        self.distances = []
        self.rewards = []
        self.timestamps = []
        self.episode_info = {}
    
    def log_step(self, obs: Dict, reward: float, info: Dict, goal_pos: np.ndarray = None):
        """Log a single step"""
        # Extract agent position with better debugging
        agent_pos = None
        
        # Check obs first
        if 'position' in obs:
            agent_pos = obs['position']
            print(f"DEBUG: Found position in obs: {agent_pos}")
        elif 'agent_position' in obs:
            agent_pos = obs['agent_position']  
            print(f"DEBUG: Found agent_position in obs: {agent_pos}")
        elif 'pos' in obs:
            agent_pos = obs['pos']
            print(f"DEBUG: Found pos in obs: {agent_pos}")
        
        # Check info if not found in obs
        if agent_pos is None:
            if 'agent_position' in info:
                agent_pos = info['agent_position']
                print(f"DEBUG: Found agent_position in info: {agent_pos}")
            elif 'position' in info:
                agent_pos = info['position']
                print(f"DEBUG: Found position in info: {agent_pos}")
            elif 'agent_x' in info and 'agent_z' in info:
                agent_pos = np.array([info['agent_x'], info.get('agent_y', 0), info['agent_z']])
                print(f"DEBUG: Constructed position from agent_x/z: {agent_pos}")
        
        # Final fallback
        if agent_pos is None:
            print(f"DEBUG: No position found in obs keys: {list(obs.keys())}")
            print(f"DEBUG: No position found in info keys: {list(info.keys())}")
            # Check if agent_position exists but is None or empty
            if 'agent_position' in info:
                print(f"DEBUG: agent_position exists but value is: {info['agent_position']} (type: {type(info['agent_position'])})")
            agent_pos = np.array([0.0, 0.0, 0.0])
        
        # Ensure agent_pos is numpy array and has 3 dimensions
        if agent_pos is not None:
            if not isinstance(agent_pos, np.ndarray):
                agent_pos = np.array(agent_pos)
            if len(agent_pos) < 3:
                # Pad with zeros if missing dimensions
                agent_pos = np.pad(agent_pos, (0, 3 - len(agent_pos)), 'constant')
            
            self.agent_positions.append(agent_pos.copy())
            print(f"DEBUG: Logged position {len(self.agent_positions)}: {agent_pos}")
        
        # Store goal position if provided
        if goal_pos is not None:
            if not isinstance(goal_pos, np.ndarray):
                goal_pos = np.array(goal_pos)
            self.goal_positions.append(goal_pos.copy())
            print(f"DEBUG: Logged goal position: {goal_pos}")
        
        # Store images (handle different formats)
        if 'rgb' in obs:
            rgb_img = obs['rgb']
            # Handle different image formats
            if len(rgb_img.shape) == 3:
                if rgb_img.shape[0] == 3:  # CHW format
                    rgb_img = np.transpose(rgb_img, (1, 2, 0))  # Convert to HWC
                elif rgb_img.shape[2] == 3:  # Already HWC
                    pass
                else:
                    print(f"DEBUG: Unexpected RGB shape: {rgb_img.shape}")
            
            # Ensure values are in [0, 255] range
            if rgb_img.max() <= 1.0:
                rgb_img = (rgb_img * 255).astype(np.uint8)
            
            self.current_images.append(rgb_img.copy())
        
        if 'goal_rgb' in obs:
            goal_img = obs['goal_rgb']
            # Handle different image formats
            if len(goal_img.shape) == 3:
                if goal_img.shape[0] == 3:  # CHW format
                    goal_img = np.transpose(goal_img, (1, 2, 0))  # Convert to HWC
            
            # Ensure values are in [0, 255] range  
            if goal_img.max() <= 1.0:
                goal_img = (goal_img * 255).astype(np.uint8)
            
            # Only add if it's different from the last one or if it's the first
            if len(self.goal_images) == 0 or not np.array_equal(goal_img, self.goal_images[-1]):
                self.goal_images.append(goal_img.copy())
        
        # Store metrics
        self.distances.append(info.get('distance_to_goal', 0.0))
        self.rewards.append(reward)
        self.timestamps.append(len(self.agent_positions))
        
        # Store episode info
        self.episode_info = {
            'goal_conditioned': info.get('goal_conditioned', False),
            'success': info.get('success', False),
            'scene_type': info.get('scene_type', 'unknown')
        }
    
    def plot_episode(self, episode_num: int = None, show: bool = False):
        """Create comprehensive episode visualization"""
        if not self.agent_positions:
            print("No data to plot")
            return
        
        # Convert positions to numpy array
        positions = np.array(self.agent_positions)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Top view trajectory plot
        ax1 = plt.subplot(2, 4, 1)
        self._plot_trajectory_top_view(ax1, positions)
        
        # 2. 3D trajectory plot
        ax2 = plt.subplot(2, 4, 2, projection='3d')
        self._plot_trajectory_3d(ax2, positions)
        
        # 3. Distance over time
        ax3 = plt.subplot(2, 4, 3)
        self._plot_distance_over_time(ax3)
        
        # 4. Reward over time
        ax4 = plt.subplot(2, 4, 4)
        self._plot_reward_over_time(ax4)
        
        # 5. Goal image (if available)
        ax5 = plt.subplot(2, 4, 5)
        self._plot_goal_image(ax5)
        
        # 6. First agent view
        ax6 = plt.subplot(2, 4, 6)
        self._plot_agent_view(ax6, 0, "Start View")
        
        # 7. Middle agent view
        ax7 = plt.subplot(2, 4, 7)
        mid_idx = len(self.current_images) // 2 if self.current_images else 0
        self._plot_agent_view(ax7, mid_idx, "Mid View")
        
        # 8. Final agent view
        ax8 = plt.subplot(2, 4, 8)
        final_idx = len(self.current_images) - 1 if self.current_images else 0
        self._plot_agent_view(ax8, final_idx, "End View")
        
        # Add overall title
        success_str = "SUCCESS" if self.episode_info.get('success', False) else "FAIL"
        goal_str = "Goal-Conditioned" if self.episode_info.get('goal_conditioned', False) else "Exploration"
        scene_str = self.episode_info.get('scene_type', 'unknown')
        
        fig.suptitle(f'Episode {episode_num} - {success_str} - {goal_str} - {scene_str}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot (always save, never show)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"episode_{episode_num}_{timestamp}.png" if episode_num else f"episode_{timestamp}.png"
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Episode visualization saved to: {filepath}")
        
        # Always close the figure to free memory
        plt.close()
    
    def _plot_trajectory_top_view(self, ax, positions):
        """Plot top-down view of agent trajectory"""
        if len(positions) == 0:
            ax.text(0.5, 0.5, 'No position data', transform=ax.transAxes, ha='center')
            ax.set_title('Top View Trajectory - No Data')
            return
        
        print(f"DEBUG: Plotting {len(positions)} positions")
        print(f"DEBUG: Position range X: {positions[:, 0].min():.2f} to {positions[:, 0].max():.2f}")
        print(f"DEBUG: Position range Z: {positions[:, 2].min():.2f} to {positions[:, 2].max():.2f}")
        
        # Plot trajectory line
        if len(positions) > 1:
            ax.plot(positions[:, 0], positions[:, 2], 'b-', alpha=0.7, linewidth=2, label='Agent Path')
        
        # Plot start position
        ax.scatter(positions[0, 0], positions[0, 2], color='green', s=150, marker='o', 
                  label='Start', zorder=5, edgecolor='darkgreen', linewidth=2)
        
        # Plot end position (if different from start)
        if len(positions) > 1:
            ax.scatter(positions[-1, 0], positions[-1, 2], color='red', s=150, marker='s', 
                      label='End', zorder=5, edgecolor='darkred', linewidth=2)
        
        # Plot intermediate positions as dots
        if len(positions) > 2:
            ax.scatter(positions[1:-1, 0], positions[1:-1, 2], color='lightblue', s=20, 
                      alpha=0.6, zorder=3)
        
        # Plot goal positions if available
        if self.goal_positions:
            goal_pos = np.array(self.goal_positions)
            if len(goal_pos) > 0:
                # Use the last goal position
                goal_x, goal_z = goal_pos[-1, 0], goal_pos[-1, 2]
                ax.scatter(goal_x, goal_z, color='gold', s=200, marker='*', 
                          label='Goal', zorder=6, edgecolor='black', linewidth=2)
                print(f"DEBUG: Goal position: ({goal_x:.2f}, {goal_z:.2f})")
        
        # Add step numbers for debugging
        if len(positions) <= 20:  # Only for shorter episodes
            for i, pos in enumerate(positions[::max(1, len(positions)//10)]):
                ax.annotate(f'{i}', (pos[0], pos[2]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8, alpha=0.7)
        
        ax.set_xlabel('X Position (meters)')
        ax.set_ylabel('Z Position (meters)')
        ax.set_title(f'Top View Trajectory ({len(positions)} steps)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio but allow some padding
        try:
            ax.set_aspect('equal', adjustable='box')
        except:
            pass  # Fallback if aspect ratio fails
    
    def _plot_trajectory_3d(self, ax, positions):
        """Plot 3D trajectory"""
        if len(positions) == 0:
            return
        
        ax.plot(positions[:, 0], positions[:, 2], positions[:, 1], 'b-', alpha=0.7, linewidth=2)
        ax.scatter(positions[0, 0], positions[0, 2], positions[0, 1], color='green', s=100, label='Start')
        ax.scatter(positions[-1, 0], positions[-1, 2], positions[-1, 1], color='red', s=100, label='End')
        
        if self.goal_positions:
            goal_pos = np.array(self.goal_positions)
            if len(goal_pos) > 0:
                ax.scatter(goal_pos[-1, 0], goal_pos[-1, 2], goal_pos[-1, 1], 
                          color='gold', s=200, marker='*', label='Goal')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_title('3D Trajectory')
        ax.legend()
    
    def _plot_distance_over_time(self, ax):
        """Plot distance to goal over time"""
        if not self.distances:
            ax.text(0.5, 0.5, 'No distance data', transform=ax.transAxes, ha='center')
            return
        
        ax.plot(self.timestamps, self.distances, 'r-', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Distance to Goal')
        ax.set_title('Distance Over Time')
        ax.grid(True, alpha=0.3)
        
        # Add success threshold line if applicable
        if hasattr(self, 'success_threshold'):
            ax.axhline(y=self.success_threshold, color='green', linestyle='--', 
                      label=f'Success Threshold ({self.success_threshold})')
            ax.legend()
    
    def _plot_reward_over_time(self, ax):
        """Plot reward over time"""
        if not self.rewards:
            ax.text(0.5, 0.5, 'No reward data', transform=ax.transAxes, ha='center')
            return
        
        ax.plot(self.timestamps, self.rewards, 'g-', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Over Time')
        ax.grid(True, alpha=0.3)
        
        # Add cumulative reward
        cumulative_rewards = np.cumsum(self.rewards)
        ax2 = ax.twinx()
        ax2.plot(self.timestamps, cumulative_rewards, 'orange', alpha=0.7, linestyle='--')
        ax2.set_ylabel('Cumulative Reward', color='orange')
    
    def _plot_goal_image(self, ax):
        """Plot goal image"""
        if self.goal_images:
            goal_img = self.goal_images[-1]
            if goal_img.max() > 0:  # Check if image has content
                ax.imshow(goal_img)
                ax.set_title('Goal Image')
            else:
                ax.text(0.5, 0.5, 'No Goal Image', transform=ax.transAxes, ha='center')
        else:
            ax.text(0.5, 0.5, 'No Goal Image', transform=ax.transAxes, ha='center')
        ax.axis('off')
    
    def _plot_agent_view(self, ax, idx: int, title: str):
        """Plot agent's view at specific timestep"""
        if idx < len(self.current_images):
            img = self.current_images[idx]
            ax.imshow(img)
            ax.set_title(f'{title} (Step {idx})')
        else:
            ax.text(0.5, 0.5, f'No image at step {idx}', transform=ax.transAxes, ha='center')
        ax.axis('off')
    
    def debug_data(self):
        """Debug method to check what data has been collected"""
        print("\n" + "="*50)
        print("VISUALIZATION DEBUG INFO")
        print("="*50)
        print(f"Agent positions collected: {len(self.agent_positions)}")
        if self.agent_positions:
            print(f"First position: {self.agent_positions[0]}")
            print(f"Last position: {self.agent_positions[-1]}")
            positions = np.array(self.agent_positions)
            print(f"X range: {positions[:, 0].min():.2f} to {positions[:, 0].max():.2f}")
            print(f"Y range: {positions[:, 1].min():.2f} to {positions[:, 1].max():.2f}")
            print(f"Z range: {positions[:, 2].min():.2f} to {positions[:, 2].max():.2f}")
        
        print(f"Goal positions collected: {len(self.goal_positions)}")
        if self.goal_positions:
            print(f"Goal position: {self.goal_positions[-1]}")
        
        print(f"Images collected: {len(self.current_images)}")
        print(f"Goal images collected: {len(self.goal_images)}")
        print(f"Distance measurements: {len(self.distances)}")
        print(f"Reward measurements: {len(self.rewards)}")
        print(f"Episode info: {self.episode_info}")
        print("="*50)
    
    def print_episode_summary(self):
        """Print summary of episode"""
        if not self.agent_positions:
            print("No episode data available")
            return
        
        self.debug_data()  # Call debug first
        
        positions = np.array(self.agent_positions)
        
        print("\n" + "="*50)
        print("EPISODE SUMMARY")
        print("="*50)
        print(f"Episode Type: {'Goal-Conditioned' if self.episode_info.get('goal_conditioned') else 'Exploration'}")
        print(f"Success: {self.episode_info.get('success', False)}")
        print(f"Scene Type: {self.episode_info.get('scene_type', 'unknown')}")
        print(f"Total Steps: {len(positions)}")
        
        print(f"\nAgent Trajectory:")
        print(f"  Start Position: ({positions[0, 0]:.2f}, {positions[0, 1]:.2f}, {positions[0, 2]:.2f})")
        print(f"  End Position: ({positions[-1, 0]:.2f}, {positions[-1, 1]:.2f}, {positions[-1, 2]:.2f})")
        
        # Calculate movement distance
        if len(positions) > 1:
            total_distance = 0
            for i in range(1, len(positions)):
                step_dist = np.linalg.norm(positions[i] - positions[i-1])
                total_distance += step_dist
            print(f"  Total Distance Traveled: {total_distance:.2f}")
            
            # Displacement (straight line distance)
            displacement = np.linalg.norm(positions[-1] - positions[0])
            print(f"  Net Displacement: {displacement:.2f}")
        
        if self.goal_positions:
            goal_pos = self.goal_positions[-1]
            print(f"  Goal Position: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f}, {goal_pos[2]:.2f})")
            
            # Distance to goal at start and end
            start_to_goal = np.linalg.norm(positions[0] - goal_pos)
            end_to_goal = np.linalg.norm(positions[-1] - goal_pos)
            print(f"  Start Distance to Goal: {start_to_goal:.2f}")
            print(f"  End Distance to Goal: {end_to_goal:.2f}")
            print(f"  Goal Distance Improvement: {start_to_goal - end_to_goal:.2f}")
        
        if self.distances:
            print(f"\nDistance Metrics:")
            print(f"  Initial Distance: {self.distances[0]:.2f}")
            print(f"  Final Distance: {self.distances[-1]:.2f}")
            print(f"  Min Distance: {min(self.distances):.2f}")
            print(f"  Distance Improvement: {self.distances[0] - self.distances[-1]:.2f}")
        
        if self.rewards:
            print(f"\nReward Metrics:")
            print(f"  Total Reward: {sum(self.rewards):.2f}")
            print(f"  Average Reward: {np.mean(self.rewards):.3f}")
            print(f"  Max Single Reward: {max(self.rewards):.2f}")
            print(f"  Min Single Reward: {min(self.rewards):.2f}")


def add_visualization_to_environment(env_class):
    """Decorator to add visualization capabilities to environment"""
    
    class VisualizedEnvironment(env_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.visualizer = EpisodeVisualizer()
            self.episode_count = 0
        
        def reset(self):
            # Plot previous episode if exists
            if hasattr(self, 'visualizer') and self.visualizer.agent_positions:
                self.visualizer.print_episode_summary()
                self.visualizer.plot_episode(self.episode_count, show=False)
                self.episode_count += 1
            
            # Reset for new episode
            obs = super().reset()
            self.visualizer.reset()
            
            # Log initial step
            self.visualizer.log_step(
                obs=obs,
                reward=0.0,
                info={'goal_conditioned': self.is_goal_conditioned, 'distance_to_goal': self._distance_to_goal() if hasattr(self, '_distance_to_goal') else 0.0},
                goal_pos=self.goal_position if hasattr(self, 'goal_position') else None
            )
            
            return obs
        
        def step(self, action):
            obs, reward, done, info = super().step(action)
            
            # Log step
            self.visualizer.log_step(
                obs=obs,
                reward=reward,
                info=info,
                goal_pos=self.goal_position if hasattr(self, 'goal_position') else None
            )
            
            return obs, reward, done, info
    
    return VisualizedEnvironment


# Usage example:
# VisualizedEnv = add_visualization_to_environment(YourEnvironmentClass)
# env = VisualizedEnv(...)