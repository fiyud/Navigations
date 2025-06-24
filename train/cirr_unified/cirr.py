import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque

class EnhancedCurriculumManager:
    def __init__(self, config: Dict, dataset_type: str = 'combined'):
        self.config = config
        self.dataset_type = dataset_type
        
        self.success_window = deque(maxlen=config.get('curriculum_window_size', 100))
        self.collision_window = deque(maxlen=config.get('curriculum_window_size', 100))
        self.episode_length_window = deque(maxlen=config.get('curriculum_window_size', 100))

        # Curriculum params
        self.current_level = 0
        self.success_threshold = config.get('curriculum_success_threshold', 0.7)
        self.min_episodes = config.get('curriculum_min_episodes', 100)
        self.episodes_at_level = 0
        
        # curriculum levels
        self.levels = self._define_curriculum_levels()
        
        # Scene progression
        self.available_scenes = {
            'ithor': {
                'kitchen': [f'FloorPlan{i}' for i in range(1, 31)],
                'living_room': [f'FloorPlan{i}' for i in range(201, 231)],
                'bedroom': [f'FloorPlan{i}' for i in range(301, 331)],
                'bathroom': [f'FloorPlan{i}' for i in range(401, 431)]
            },
            'robothor': {
                'train': [f'FloorPlan_Train{i}_{j}' for i in range(1, 13) for j in range(1, 6)],
                'val': [f'FloorPlan_Val{i}_{j}' for i in range(1, 6) for j in range(1, 4)]
            }
        }
        
        self.level_history = []
        self.performance_history = []
        
    def _define_curriculum_levels(self) -> List[Dict]:
        """Define progression of difficulty levels"""
        if self.dataset_type == 'combined':
            levels = [
                # Level 0: Single room type, short distances
                {
                    'name': 'Basic Navigation',
                    'room_types': ['kitchen'],
                    'num_scenes': 3,
                    'max_distance': 2.0,
                    'max_episode_steps': 150,
                    'goal_prob': 0.9,
                    'collision_penalty_multiplier': 0.2,
                    'use_robothor': False
                },
                
                # Level 1: Two room types, medium distances
                {
                    'name': 'Multi-Room Easy',
                    'room_types': ['kitchen', 'living_room'],
                    'num_scenes': 10,
                    'max_distance': 5.0,
                    'max_episode_steps': 200,
                    'goal_prob': 0.7,
                    'collision_penalty_multiplier': 0.7,
                    'use_robothor': False
                },
                
                # Level 2: All iTHOR rooms, longer distances
                {
                    'name': 'Full iTHOR',
                    'room_types': ['kitchen', 'living_room', 'bedroom', 'bathroom'],
                    'num_scenes': 20,
                    'max_distance': 8.0,
                    'max_episode_steps': 300,
                    'goal_prob': 0.6,
                    'collision_penalty_multiplier': 1.0,
                    'use_robothor': False
                },
                
                # Level 3: Introduction to RoboTHOR
                {
                    'name': 'RoboTHOR Introduction',
                    'room_types': ['kitchen', 'living_room'],
                    'num_scenes': 15,
                    'max_distance': 6.0,
                    'max_episode_steps': 300,
                    'goal_prob': 0.6,
                    'collision_penalty_multiplier': 1.0,
                    'use_robothor': True,
                    'robothor_ratio': 0.3
                },
                
                # Level 4: Mixed environments
                {
                    'name': 'Mixed Environments',
                    'room_types': ['all'],
                    'num_scenes': 30,
                    'max_distance': 10.0,
                    'max_episode_steps': 400,
                    'goal_prob': 0.5,
                    'collision_penalty_multiplier': 1.2,
                    'use_robothor': True,
                    'robothor_ratio': 0.5
                },
                
                # Level 5: Full difficulty
                {
                    'name': 'Expert Navigation',
                    'room_types': ['all'],
                    'num_scenes': -1,  # All available
                    'max_distance': float('inf'),
                    'max_episode_steps': 500,
                    'goal_prob': 0.5,
                    'collision_penalty_multiplier': 1.5,
                    'use_robothor': True,
                    'robothor_ratio': 0.5
                }
            ]
            
        elif self.dataset_type == 'ithor':
            levels = [
                {
                    'name': 'Single Room',
                    'room_types': ['kitchen'],
                    'num_scenes': 5,
                    'max_distance': 3.0,
                    'max_episode_steps': 100,
                    'goal_prob': 0.8
                },
                {
                    'name': 'Two Rooms',
                    'room_types': ['kitchen', 'living_room'],
                    'num_scenes': 10,
                    'max_distance': 5.0,
                    'max_episode_steps': 200,
                    'goal_prob': 0.7
                },
                {
                    'name': 'All Rooms',
                    'room_types': ['all'],
                    'num_scenes': 30,
                    'max_distance': 8.0,
                    'max_episode_steps': 300,
                    'goal_prob': 0.6
                },
                {
                    'name': 'Expert',
                    'room_types': ['all'],
                    'num_scenes': -1,
                    'max_distance': float('inf'),
                    'max_episode_steps': 500,
                    'goal_prob': 0.5
                }
            ]
            
        else:  # robothor
            levels = [
                {
                    'name': 'Few Scenes',
                    'num_scenes': 10,
                    'max_distance': 4.0,
                    'max_episode_steps': 200,
                    'goal_prob': 0.8
                },
                {
                    'name': 'More Scenes',
                    'num_scenes': 30,
                    'max_distance': 6.0,
                    'max_episode_steps': 300,
                    'goal_prob': 0.6
                },
                {
                    'name': 'Full Dataset',
                    'num_scenes': -1,
                    'max_distance': float('inf'),
                    'max_episode_steps': 500,
                    'goal_prob': 0.5
                }
            ]
        
        return levels
    
    def get_current_settings(self) -> Dict:
        """Get current curriculum settings"""
        level = self.levels[min(self.current_level, len(self.levels) - 1)]
        return level.copy()
    
    def get_current_scenes(self, all_scenes: List[str]) -> List[str]:
        level = self.get_current_settings()
        
        if self.dataset_type == 'combined':
            selected_scenes = []
            
            # Get iTHOR scenes
            if level.get('room_types', ['all'])[0] == 'all':
                ithor_scenes = []
                for room_scenes in self.available_scenes['ithor'].values():
                    ithor_scenes.extend(room_scenes)
            else:
                ithor_scenes = []
                for room_type in level['room_types']:
                    if room_type in self.available_scenes['ithor']:
                        ithor_scenes.extend(self.available_scenes['ithor'][room_type])
            
            # Add RoboTHOR scenes if enabled
            if level.get('use_robothor', False):
                robothor_ratio = level.get('robothor_ratio', 0.3)
                robothor_scenes = self.available_scenes['robothor']['train']
                
                # Calculate split
                num_scenes = level['num_scenes'] if level['num_scenes'] > 0 else len(all_scenes)
                num_robothor = int(num_scenes * robothor_ratio)
                num_ithor = num_scenes - num_robothor
                
                # Sample scenes
                if num_ithor > 0 and ithor_scenes:
                    selected_scenes.extend(
                        np.random.choice(ithor_scenes, 
                                       min(num_ithor, len(ithor_scenes)), 
                                       replace=False)
                    )
                
                if num_robothor > 0 and robothor_scenes:
                    selected_scenes.extend(
                        np.random.choice(robothor_scenes, 
                                       min(num_robothor, len(robothor_scenes)), 
                                       replace=False)
                    )
            else:
                num_scenes = level['num_scenes'] if level['num_scenes'] > 0 else len(ithor_scenes)
                selected_scenes = list(np.random.choice(
                    ithor_scenes, 
                    min(num_scenes, len(ithor_scenes)), 
                    replace=False
                ))
            
            return selected_scenes
            
        else:
            # Original logic for single dataset
            num_scenes = level['num_scenes']
            if num_scenes <= 0 or num_scenes >= len(all_scenes):
                return all_scenes
            
            # Ensure we have scenes from each room type if specified
            if 'room_types' in level and level['room_types'][0] != 'all':
                selected_scenes = []
                for room_type in level['room_types']:
                    room_scenes = [s for s in all_scenes if self._get_room_type(s) == room_type]
                    if room_scenes:
                        num_per_room = max(1, num_scenes // len(level['room_types']))
                        selected_scenes.extend(
                            np.random.choice(room_scenes, 
                                           min(num_per_room, len(room_scenes)), 
                                           replace=False)
                        )
                return selected_scenes[:num_scenes]
            else:
                return list(np.random.choice(all_scenes, num_scenes, replace=False))
    
    def _get_room_type(self, scene_name: str) -> str:
        """Determine room type from scene name"""
        if 'Train' in scene_name or 'Val' in scene_name or 'Test' in scene_name:
            return 'robothor'
        
        try:
            num = int(''.join(filter(str.isdigit, scene_name)))
            if 1 <= num <= 30:
                return 'kitchen'
            elif 201 <= num <= 230:
                return 'living_room'
            elif 301 <= num <= 330:
                return 'bedroom'
            elif 401 <= num <= 430:
                return 'bathroom'
        except:
            pass
        
        return 'unknown'
    
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
            
            # Record performance
            self.performance_history.append({
                'level': self.current_level,
                'success_rate': current_success_rate,
                'avg_length': avg_episode_length,
                'avg_collisions': avg_collisions,
                'episodes': self.episodes_at_level
            })
            
            # Check advancement criteria
            if self._should_advance(current_success_rate, avg_collisions):
                self.advance_level()
            elif self._should_decrease(current_success_rate):
                self.decrease_level()
    
    def _should_advance(self, success_rate: float, avg_collisions: float) -> bool:
        """Check if ready to advance to next level"""
        if self.current_level >= len(self.levels) - 1:
            return False
        
        if success_rate < self.success_threshold:
            return False
        
        # Additional criteria for higher levels
        if self.current_level >= 2:
            # Require low collision rate
            max_collisions = 5.0 - self.current_level * 0.5
            if avg_collisions > max_collisions:
                return False
        
        return True
    
    def _should_decrease(self, success_rate: float) -> bool:
        """Check if difficulty should be reduced"""
        # Don't decrease from first level
        if self.current_level == 0:
            return False
        
        # If success rate is very low, consider decreasing
        return success_rate < 0.3 and self.episodes_at_level > self.min_episodes * 2
    
    def advance_level(self):
        if self.current_level < len(self.levels) - 1:
            self.current_level += 1
            self.episodes_at_level = 0
            self.success_window.clear()
            self.collision_window.clear()
            self.episode_length_window.clear()
            
            level = self.get_current_settings()
            print(f"\n{'='*50}")
            print(f"CURRICULUM: Advancing to Level {self.current_level}")
            print(f"Name: {level['name']}")
            print(f"Max Distance: {level['max_distance']}")
            print(f"Max Steps: {level['max_episode_steps']}")
            print(f"Goal Probability: {level['goal_prob']}")
            print(f"{'='*50}\n")
    
    def decrease_level(self):
        if self.current_level > 0:
            self.current_level -= 1
            self.episodes_at_level = 0
            self.success_window.clear()
            self.collision_window.clear()
            self.episode_length_window.clear()
            
            print(f"\nCURRICULUM: Decreasing to Level {self.current_level} due to low performance")
    
    def get_progress_stats(self) -> Dict:
        return {
            'current_level': self.current_level,
            'level_name': self.levels[self.current_level]['name'],
            'episodes_at_level': self.episodes_at_level,
            'current_success_rate': np.mean(self.success_window) if self.success_window else 0.0,
            'levels_completed': self.current_level,
            'total_levels': len(self.levels)
        }