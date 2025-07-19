import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import yaml
import os

class GridBasedCurriculumManager:
    def __init__(self, config: Dict, dataset_type: str = 'combined'):
        self.config = config
        self.dataset_type = dataset_type
        
        # Performance tracking windows
        self.success_window = deque(maxlen=config.get('curriculum_window_size', 100))
        self.collision_window = deque(maxlen=config.get('curriculum_window_size', 100))
        self.episode_length_window = deque(maxlen=config.get('curriculum_window_size', 100))
        
        self.current_level = 0
        self.success_threshold = config.get('curriculum_success_threshold', 0.7)
        self.min_episodes = config.get('curriculum_min_episodes', 100)
        self.episodes_at_level = 0
        
        self.scene_grid_sizes = self._load_or_compute_scene_grid_sizes()
        # Create curriculum levels based on grid sizes
        self.levels = self._create_grid_based_levels()
        
        self.level_history = []
        self.performance_history = []
        
    def _load_or_compute_scene_grid_sizes(self) -> Dict[str, int]:
        """Load precomputed grid sizes or compute them"""
        grid_file = os.path.join(
            os.path.dirname(self.config.get('splits_file', '')), 
            'scene_grid_sizes.yaml'
        )
        
        if os.path.exists(grid_file):
            with open(grid_file, 'r') as f:
                return yaml.safe_load(f)
        
        # If not precomputed, return estimated sizes based on scene type
        # You'll need to run a separate script to actually compute these
        print("Warning: Scene grid sizes not found. Using estimates.")
        return self._estimate_scene_grid_sizes()
    
    def _estimate_scene_grid_sizes(self) -> Dict[str, int]:
        """Estimate grid sizes based on scene naming patterns"""
        grid_sizes = {}
        
        # iTHOR estimates (based on room type and number)
        # Generally: kitchens < bathrooms < bedrooms < living rooms
        for i in range(1, 31):
            grid_sizes[f'FloorPlan{i}'] = 50 + (i % 10) * 5  # Kitchen: 50-100 grids
        
        for i in range(201, 231):
            grid_sizes[f'FloorPlan{i}'] = 80 + (i % 10) * 8  # Living room: 80-160 grids
            
        for i in range(301, 331):
            grid_sizes[f'FloorPlan{i}'] = 60 + (i % 10) * 6  # Bedroom: 60-120 grids
            
        for i in range(401, 431):
            grid_sizes[f'FloorPlan{i}'] = 40 + (i % 10) * 4  # Bathroom: 40-80 grids
        
        # RoboTHOR estimates (generally larger apartments)
        for i in range(1, 13):
            for j in range(1, 6):
                grid_sizes[f'FloorPlan_Train{i}_{j}'] = 100 + i * 10 + j * 5  # 105-175 grids
                
        for i in range(1, 6):
            for j in range(1, 4):
                grid_sizes[f'FloorPlan_Val{i}_{j}'] = 110 + i * 10 + j * 5  # 115-165 grids
                
        for i in range(1, 8):
            for j in range(1, 3):
                grid_sizes[f'FloorPlan_Test{i}_{j}'] = 120 + i * 10 + j * 5  # 125-185 grids
        
        return grid_sizes
    
    def _create_grid_based_levels(self) -> List[Dict]:
        """Create curriculum levels based on grid sizes"""
        # Sort all scenes by grid size
        sorted_scenes = sorted(
            self.scene_grid_sizes.items(), 
            key=lambda x: x[1]
        )
        
        # Determine grid size ranges for each level
        min_grid = min(self.scene_grid_sizes.values())
        max_grid = max(self.scene_grid_sizes.values())
        grid_range = max_grid - min_grid
        
        if self.dataset_type == 'combined':
            levels = [
                {
                    'name': 'Tiny Spaces',
                    'min_grid_size': min_grid,
                    'max_grid_size': min_grid + grid_range * 0.15,
                    'max_distance': 2.0,
                    'max_episode_steps': 150,
                    'goal_prob': 0.9,
                    'collision_penalty_multiplier': 0.2,
                    'num_scenes': 10
                },
                {
                    'name': 'Small Spaces',
                    'min_grid_size': min_grid + grid_range * 0.15,
                    'max_grid_size': min_grid + grid_range * 0.3,
                    'max_distance': 4.0,
                    'max_episode_steps': 200,
                    'goal_prob': 0.8,
                    'collision_penalty_multiplier': 0.5,
                    'num_scenes': 20
                },
                {
                    'name': 'Medium Spaces',
                    'min_grid_size': min_grid + grid_range * 0.3,
                    'max_grid_size': min_grid + grid_range * 0.5,
                    'max_distance': 6.0,
                    'max_episode_steps': 300,
                    'goal_prob': 0.7,
                    'collision_penalty_multiplier': 0.8,
                    'num_scenes': 30
                },
                {
                    'name': 'Large Spaces',
                    'min_grid_size': min_grid + grid_range * 0.5,
                    'max_grid_size': min_grid + grid_range * 0.75,
                    'max_distance': 8.0,
                    'max_episode_steps': 400,
                    'goal_prob': 0.6,
                    'collision_penalty_multiplier': 1.0,
                    'num_scenes': 40
                },
                {
                    'name': 'Complex Spaces',
                    'min_grid_size': min_grid + grid_range * 0.75,
                    'max_grid_size': max_grid,
                    'max_distance': 10.0,
                    'max_episode_steps': 450,
                    'goal_prob': 0.5,
                    'collision_penalty_multiplier': 1.2,
                    'num_scenes': 50
                },
                {
                    'name': 'All Spaces',
                    'min_grid_size': min_grid,
                    'max_grid_size': max_grid,
                    'max_distance': float('inf'),
                    'max_episode_steps': 500,
                    'goal_prob': 0.5,
                    'collision_penalty_multiplier': 1.5,
                    'num_scenes': -1  # All scenes
                }
            ]
        else:
            num_levels = 4
            levels = []
            for i in range(num_levels):
                level_min = min_grid + (grid_range * i / num_levels)
                level_max = min_grid + (grid_range * (i + 1) / num_levels)
                
                levels.append({
                    'name': f'Level {i+1}',
                    'min_grid_size': level_min,
                    'max_grid_size': level_max,
                    'max_distance': 3.0 + i * 2.0,
                    'max_episode_steps': 200 + i * 75,
                    'goal_prob': 0.8 - i * 0.1,
                    'collision_penalty_multiplier': 0.5 + i * 0.3,
                    'num_scenes': 10 + i * 10
                })
            
            # Add final level with all scenes
            levels.append({
                'name': 'Expert',
                'min_grid_size': min_grid,
                'max_grid_size': max_grid,
                'max_distance': float('inf'),
                'max_episode_steps': 500,
                'goal_prob': 0.5,
                'collision_penalty_multiplier': 1.5,
                'num_scenes': -1
            })
        
        return levels
    
    def get_current_settings(self) -> Dict:
        """Get current curriculum settings"""
        level = self.levels[min(self.current_level, len(self.levels) - 1)]
        return level.copy()
    
    def get_current_scenes(self, all_scenes: List[str]) -> List[str]:
        """Get scenes for current curriculum level based on grid size"""
        level = self.get_current_settings()
        
        # Filter scenes by grid size
        valid_scenes = []
        for scene in all_scenes:
            if scene in self.scene_grid_sizes:
                grid_size = self.scene_grid_sizes[scene]
                if level['min_grid_size'] <= grid_size <= level['max_grid_size']:
                    valid_scenes.append(scene)
        
        # If no scenes found in range, expand the range slightly
        if not valid_scenes:
            print(f"Warning: No scenes found for grid range {level['min_grid_size']}-{level['max_grid_size']}")
            valid_scenes = all_scenes[:10]  # Fallback
        
        # Limit number of scenes if specified
        num_scenes = level['num_scenes']
        if num_scenes > 0 and len(valid_scenes) > num_scenes:
            # Sort by grid size and take a distributed sample
            sorted_valid = sorted(valid_scenes, key=lambda s: self.scene_grid_sizes.get(s, 0))
            
            # Take evenly distributed scenes
            indices = np.linspace(0, len(sorted_valid) - 1, num_scenes, dtype=int)
            valid_scenes = [sorted_valid[i] for i in indices]
        
        print(f"Level {self.current_level} ({level['name']}): "
              f"{len(valid_scenes)} scenes, "
              f"grid range {level['min_grid_size']:.0f}-{level['max_grid_size']:.0f}")
        
        return valid_scenes
    
    def update(self, episode_success: bool, episode_length: int, collision_count: int):
        """Update curriculum based on performance"""
        self.success_window.append(float(episode_success))
        self.episode_length_window.append(episode_length)
        self.collision_window.append(collision_count)
        self.episodes_at_level += 1
        
        # Check if we should evaluate for advancement
        if self.episodes_at_level >= self.min_episodes and len(self.success_window) >= self.min_episodes:
            current_success_rate = np.mean(self.success_window)
            avg_episode_length = np.mean(self.episode_length_window)
            avg_collisions = np.mean(self.collision_window)
            
            self.performance_history.append({
                'level': self.current_level,
                'success_rate': current_success_rate,
                'avg_length': avg_episode_length,
                'avg_collisions': avg_collisions,
                'episodes': self.episodes_at_level,
                'grid_range': (
                    self.levels[self.current_level]['min_grid_size'],
                    self.levels[self.current_level]['max_grid_size']
                )
            })
            
            if self._should_advance(current_success_rate, avg_collisions):
                self.advance_level()
            elif self._should_decrease(current_success_rate):
                self.decrease_level()
    
    def _should_advance(self, success_rate: float, avg_collisions: float) -> bool:
        if self.current_level >= len(self.levels) - 1:
            return False
        
        if success_rate < self.success_threshold:
            return False
        
        if self.current_level >= 2:
            max_collisions = 5.0 - self.current_level * 0.5
            if avg_collisions > max_collisions:
                return False
        
        return True
    
    def _should_decrease(self, success_rate: float) -> bool:
        if self.current_level == 0:
            return False
        
        return success_rate < 0.3 and self.episodes_at_level > self.min_episodes * 2
    
    def advance_level(self):
        if self.current_level < len(self.levels) - 1:
            self.current_level += 1
            self.episodes_at_level = 0
            self.success_window.clear()
            self.collision_window.clear()
            self.episode_length_window.clear()
            
            level = self.get_current_settings()
            print(f"\n{'='*60}")
            print(f"CURRICULUM: Advancing to Level {self.current_level}")
            print(f"Name: {level['name']}")
            print(f"Grid Size Range: {level['min_grid_size']:.0f} - {level['max_grid_size']:.0f}")
            print(f"Max Distance: {level['max_distance']}")
            print(f"Max Steps: {level['max_episode_steps']}")
            print(f"Goal Probability: {level['goal_prob']}")
            print(f"{'='*60}\n")
    
    def decrease_level(self):
        """Decrease curriculum level"""
        if self.current_level > 0:
            self.current_level -= 1
            self.episodes_at_level = 0
            self.success_window.clear()
            self.collision_window.clear()
            self.episode_length_window.clear()
            
            print(f"\nCURRICULUM: Decreasing to Level {self.current_level} due to low performance")
    
    def get_progress_stats(self) -> Dict:
        """Get current progress statistics"""
        level = self.levels[self.current_level]
        return {
            'current_level': self.current_level,
            'level_name': level['name'],
            'grid_range': (level['min_grid_size'], level['max_grid_size']),
            'episodes_at_level': self.episodes_at_level,
            'current_success_rate': np.mean(self.success_window) if self.success_window else 0.0,
            'levels_completed': self.current_level,
            'total_levels': len(self.levels)
        }
    
    def save_grid_sizes(self, scene_grid_sizes: Dict[str, int], output_dir: str):
        """Save computed grid sizes for future use"""
        os.makedirs(output_dir, exist_ok=True)
        grid_file = os.path.join(output_dir, 'scene_grid_sizes.yaml')
        
        with open(grid_file, 'w') as f:
            yaml.dump(scene_grid_sizes, f, default_flow_style=False)
        
        print(f"Saved grid sizes to {grid_file}")