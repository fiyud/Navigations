import ai2thor
from ai2thor.controller import Controller
import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Optional, Any
import random
import gym
from gym import spaces
from episode_visualizer import EpisodeVisualizer
from collections import deque

class EnhancedAI2ThorEnv(gym.Env):
    def __init__(
        self,
        scene_names: List[str] = None,
        image_size: Tuple[int, int] = (224, 224),
        max_episode_steps: int = 500,
        success_distance: float = 1.0,
        rotation_step: int = 30,
        movement_step: float = 0.4,
        context_size: int = 5,
        goal_prob: float = 0.5,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.max_episode_steps = max_episode_steps
        self.success_distance = success_distance
        self.rotation_step = rotation_step
        self.movement_step = movement_step
        self.context_size = context_size
        self.goal_prob = goal_prob
        
        self.visualizer = EpisodeVisualizer()
        self.episode_count = 0
        self.log_visualization = True

        # Separate iTHOR and RoboTHOR scenes
        self.scene_names = scene_names if scene_names else []
        self.ithor_scenes = []
        self.robothor_scenes = []

        self.collision_history = deque(maxlen=10)  # Track recent collisions
        self.stuck_counter = 0
        self.last_positions = deque(maxlen=5)  # Track recent positions
        self.last_action = None
        self.consecutive_collisions = 0

        def _is_robothor_scene(self, scene_name: str) -> bool:
            if scene_name is None:
                return False
            return any(keyword in scene_name for keyword in ['Train', 'Val', 'Test'])
        
        print(f"Environment initialized with {len(self.ithor_scenes)} iTHOR and {len(self.robothor_scenes)} RoboTHOR scenes")
        
        # Initialize controller
        self.controller = None
        self.current_dataset_type = None
        self._initialize_controller()
        
        # Gym spaces
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Dict({
            'rgb': spaces.Box(low=0, high=255, shape=(3, *image_size), dtype=np.uint8),
            'goal_rgb': spaces.Box(low=0, high=255, shape=(3, *image_size), dtype=np.uint8),
            'context': spaces.Box(low=0, high=255, shape=(3 * context_size, *image_size), dtype=np.uint8),
            'goal_mask': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'goal_position': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })
        
        # Episode variables
        self.current_scene = None
        self.current_step = 0
        self.context_buffer = []
        self.goal_position = None
        self.goal_image = None
        self.is_goal_conditioned = False
        self.initial_position = None
        self.visited_positions = set()
        self.position_visit_counts = {}
        
    def _is_robothor_scene(self, scene_name):
        if scene_name is None:
            return False
        return any(keyword in scene_name for keyword in ['Train', 'Val', 'Test'])
    
    def _initialize_controller(self, scene_name: Optional[str] = None):
        """Initialize or reinitialize controller based on scene type"""
        if scene_name is None:
            scene_name = random.choice(self.scene_names) if self.scene_names else 'FloorPlan1'
        
        is_robothor = self._is_robothor_scene(scene_name)
        
        if self.controller is not None and self.current_dataset_type == is_robothor:
            return
        
        if self.controller is not None:
            self.controller.stop()
        
        if is_robothor:
            print(f"Initializing RoboTHOR controller for scene: {scene_name}")
            self.controller = Controller(
                agentMode="locobot",  # RoboTHOR uses locobot
                scene=scene_name,
                gridSize=0.4,
                snapToGrid=False,
                rotateStepDegrees=self.rotation_step,
                renderDepthImage=False,
                renderInstanceSegmentation=False,
                width=self.image_size[0],
                height=self.image_size[1],
                fieldOfView=60,
               # commit_id="5e1af1e57b07a9b5e9fbb81a7e68e6375e3c3608"  # RoboTHOR compatible version
            )
        else:
            print(f"Initializing iTHOR controller for scene: {scene_name}")
            self.controller = Controller(
                agentMode="locobot",
                scene=scene_name,
                gridSize=0.25,
                snapToGrid=False,
                rotateStepDegrees=self.rotation_step,
                renderDepthImage=False,
                renderInstanceSegmentation=False,
                width=self.image_size[0],
                height=self.image_size[1],
                fieldOfView=60,
                visibilityDistance=1.5
            )
        
        self.current_dataset_type = is_robothor
        self.current_scene = scene_name
    
    def reset(self) -> Dict[str, np.ndarray]:
        scene_name = random.choice(self.scene_names)
        
        if self.current_scene != scene_name or self.controller is None:
            self._initialize_controller(scene_name)
            self.controller.reset(scene=scene_name)
        
        # Get reachable positions
        try:
            reachable_positions = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        except:
            # Fallback for RoboTHOR if GetReachablePositions doesn't work
            reachable_positions = self._get_robothor_reachable_positions()
        
        if reachable_positions:
            start_pos = random.choice(reachable_positions)
            self.controller.step(
                action="Teleport",
                position=start_pos,
                rotation=dict(x=0, y=random.choice([0, 90, 180, 270]), z=0)
            )
        
        self.current_step = 0
        self.context_buffer = []
        self.visited_positions = set()
        self.position_visit_counts = {}
        self.initial_position = self._get_agent_position()

        # Determine if goal-conditioned
        self.is_goal_conditioned = random.random() < self.goal_prob
        
        if self.is_goal_conditioned:
            self._set_random_goal()
            self.initial_goal_distance = self._distance_to_goal()
            self._prev_distance_to_goal = self.initial_goal_distance
        else:
            self.goal_position = None
            self.goal_image = None

        print(f"Episode type: {'GOAL' if self.is_goal_conditioned else 'EXPLORATION'}, "
            f"goal_prob: {self.goal_prob}, random: {random.random()}")

        # Get init observation
        obs = self._get_observation()
        
        for _ in range(self.context_size):
            self.context_buffer.append(obs['rgb'])
        
        # Format observation before logging
        formatted_obs = self._format_observation(obs)

        # Log after formatting observation
        initial_info = {
            'goal_conditioned': self.is_goal_conditioned,
            'distance_to_goal': self._distance_to_goal() if self.is_goal_conditioned else 0.0,
            'success': False,
            'scene_type': 'robothor' if (self.current_scene and self._is_robothor_scene(self.current_scene)) else 'ithor',
            'agent_position': np.array([self.initial_position['x'], self.initial_position['y'], self.initial_position['z']])
        }
        self.log_with_position(formatted_obs, 0.0, initial_info, self.goal_position)

        if hasattr(self, 'visualizer') and self.visualizer.agent_positions and self.log_visualization:
            self.visualizer.print_episode_summary()
            self.visualizer.plot_episode(self.episode_count, show=False)
            self.episode_count += 1

        if hasattr(self, 'visualizer'):
            self.visualizer.reset()

        return formatted_obs
    
    def log_with_position(self, obs, reward, info, goal_pos):
        """Log step with guaranteed position data"""
        if hasattr(self, 'visualizer') and self.log_visualization:
            # Always get fresh position
            agent_pos = self._get_agent_position()
            
            # Add position to info if not present
            if 'agent_position' not in info:
                info['agent_position'] = np.array([agent_pos['x'], agent_pos['y'], agent_pos['z']])
            
            self.visualizer.log_step(
                obs=obs,
                reward=reward,
                info=info,
                goal_pos=goal_pos
            )

    def _get_robothor_reachable_positions(self):
        positions = []
        bounds = self.controller.last_event.metadata.get('sceneBounds', {})
        
        if bounds:
            x_min, x_max = bounds.get('center', {}).get('x', 0) - 5, bounds.get('center', {}).get('x', 0) + 5
            z_min, z_max = bounds.get('center', {}).get('z', 0) - 5, bounds.get('center', {}).get('z', 0) + 5
        else:
            x_min, x_max = -5, 5
            z_min, z_max = -5, 5
        
        for x in np.arange(x_min, x_max, 0.5):
            for z in np.arange(z_min, z_max, 0.5):
                positions.append({'x': float(x), 'y': 0.0, 'z': float(z)})
        
        reachable = []
        current_pos = self._get_agent_position()
        
        for pos in positions[:50]:  # Test subset to avoid taking too long
            event = self.controller.step(action="Teleport", position=pos)
            if event.metadata['lastActionSuccess']:
                reachable.append(pos)
        
        self.controller.step(action="Teleport", position=current_pos)
        
        return reachable if reachable else [current_pos]
    
    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        if self.is_goal_conditioned and hasattr(self, 'goal_position'):
            self._prev_distance_to_goal = self._distance_to_goal()

        if self.consecutive_collisions > 2:
            if self.last_action in [0, 1]:  # If was trying to move forward/back
                action = random.choice([2, 3])  # Force rotation
            elif self.last_action in [2, 3]:  # If was rotating
                action = 1  # Try backing up

        self.last_action = action

        action_map = {
            0: {"action": "MoveAhead", "moveMagnitude": 0.4},
            1: {"action": "MoveBack", "moveMagnitude": 0.4},
            2: {"action": "RotateLeft", "degrees": 30},
            3: {"action": "RotateRight", "degrees": 30},
            4: {"action": "MoveAhead", "moveMagnitude": 1.0},
            5: {"action": "RotateLeft", "degrees": 60},
            6: {"action": "RotateRight", "degrees": 60},
        }
        
        try:
            event = self.controller.step(**action_map[action])
        except Exception as e:
            print(f"AI2THOR step failed: {e}")
            # Return a failed step
            obs = self._get_observation()
            return self._format_observation(obs), -1.0, True, {'error': str(e)}
        
        self.current_step += 1
        
        current_pos = self._get_agent_position()
        self.last_positions.append((current_pos['x'], current_pos['z']))


        if len(self.last_positions) == 5:
            positions = list(self.last_positions)
            max_dist = max(
                np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
                for p1, p2 in zip(positions[:-1], positions[1:])
            )
            if max_dist < 0.1:  # Barely moved in last 5 steps
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
        
        if not event.metadata['lastActionSuccess']:
            self.consecutive_collisions += 1
            self.collision_history.append(action)
        else:
            self.consecutive_collisions = 0

        obs = self._get_observation()
        
        # Update context buffer
        self.context_buffer.append(obs['rgb'])
        if len(self.context_buffer) > self.context_size:
            self.context_buffer.pop(0)
        
        reward = self._calculate_reward(event, obs)
        done = self._is_done(obs)

        agent_pos = self._get_agent_position()

        info = {
            'success': self._is_success(obs),
            'collision': not event.metadata['lastActionSuccess'],
            'step': self.current_step,
            'goal_conditioned': self.is_goal_conditioned,
            'distance_to_goal': self._distance_to_goal() if self.is_goal_conditioned else 0.0,
            'scene_type': 'robothor' if (self.current_scene and self._is_robothor_scene(self.current_scene)) else 'ithor',
            'agent_position': np.array([agent_pos['x'], agent_pos['y'], agent_pos['z']]),
        }
        
        # Format observation before logging
        formatted_obs = self._format_observation(obs)
        
        # Log after formatting observation
        if hasattr(self, 'visualizer') and self.log_visualization:
            self.log_with_position(formatted_obs, reward, info, self.goal_position)

        return formatted_obs, reward, done, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        event = self.controller.last_event
        
        # Get RGB image
        rgb_image = np.array(event.frame)
        rgb_image = np.transpose(rgb_image, (2, 0, 1))
        
        if self.goal_image is not None:
            goal_rgb = np.array(self.goal_image)
            goal_rgb = np.transpose(goal_rgb, (2, 0, 1))
        else:
            goal_rgb = np.zeros((3, *self.image_size), dtype=np.uint8)
        
        # ADD THIS: Get agent position
        agent_position = self._get_agent_position()
        
        return {
            'rgb': rgb_image,
            'goal_rgb': goal_rgb,
            'position': np.array([
                agent_position['x'],
                agent_position['y'], 
                agent_position['z']
            ]),
            'rotation': event.metadata['agent']['rotation']['y']
        }
    
    def _format_observation(self, obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Format observation for output"""
        if len(self.context_buffer) == self.context_size:
            context = np.concatenate(self.context_buffer, axis=0)
        else:
            # Pad with zeros if not enough context
            context = np.concatenate(
                self.context_buffer + [np.zeros_like(obs['rgb'])] * (self.context_size - len(self.context_buffer)),
                axis=0
            )
        
        formatted_obs = {
                'rgb': obs['rgb'].astype(np.uint8),
                'goal_rgb': obs['goal_rgb'].astype(np.uint8),
                'context': context.astype(np.uint8),
                'goal_mask': np.array([1.0 if self.is_goal_conditioned else 0.0], dtype=np.float32),
                # 'goal_mask': np.array([0.0 if self.is_goal_conditioned else 1.0], dtype=np.float32),
                # 'goal_mask': np.array([1.0 if self.is_goal_conditioned else 0.0], dtype=np.float32),
                'goal_position': self.goal_position if self.goal_position is not None else np.zeros(3, dtype=np.float32),
                # 'position': obs['position'].astype(np.float32),
            }
        
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
        
        if self._debug_counter % 100 == 0:
            print(f"[ENV] Sending observation - is_goal_conditioned: {self.is_goal_conditioned}, "
                f"goal_mask: {formatted_obs['goal_mask'][0]}")
        
        return formatted_obs
    
    def _set_random_goal(self):
        """Set goal using strict object-based selection to avoid walls"""
        try:
            reachable_positions = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        except:
            reachable_positions = self._get_robothor_reachable_positions()
        
        if not reachable_positions:
            print("No reachable positions found!")
            return
        
        current_pos = self._get_agent_position()
        
        # Try up to 10 times to find a good goal
        max_attempts = 10
        for attempt in range(max_attempts):
            # Get object-based goals with strict filtering
            object_goals = self._get_strict_object_goals(reachable_positions, current_pos)
            
            if object_goals:
                # Select the best goal
                best_goal = max(object_goals, key=lambda g: g['quality_score'])
                
                print(f"Selected goal with score {best_goal['quality_score']:.2f}, "
                    f"{best_goal['object_count']} objects visible")
                
                self._set_goal_from_position(best_goal['position'], best_goal['rotation'])
                return
        
        # If we still haven't found a good goal, try with relaxed constraints
        print("No high-quality goals found, trying with relaxed constraints...")
        
        # Last resort: find any position with at least one object
        emergency_goals = self._get_emergency_object_goals(reachable_positions, current_pos)
        if emergency_goals:
            goal = random.choice(emergency_goals)
            self._set_goal_from_position(goal['position'], goal['rotation'])
        else:
            print("WARNING: Could not find any object-based goals!")
            # Don't set a goal at all - switch to exploration mode
            self.is_goal_conditioned = False
            self.goal_position = None
            self.goal_image = None
    
    def _get_strict_object_goals(self, reachable_positions, current_pos):
        """Find positions with high-quality object views, strictly avoiding walls"""
        object_goals = []
        
        # Store current state
        current_state = {
            'position': current_pos,
            'rotation': self.controller.last_event.metadata['agent']['rotation']
        }
        
        # High-value object types that make good navigation targets
        high_value_objects = {
            # Electronics
            'Television': 3, 'Laptop': 3, 'Computer': 3, 'Monitor': 3,
            # Furniture 
            'Sofa': 2, 'ArmChair': 2, 'Bed': 2, 'DiningTable': 2,
            # Appliances
            'Microwave': 2, 'Fridge': 2, 'StoveBurner': 2, 'CoffeeMachine': 2,
            # Decorative
            'Painting': 3, 'HousePlant': 2, 'FloorLamp': 2, 'DeskLamp': 2,
            # Interactive items
            'Book': 1, 'RemoteControl': 1, 'Newspaper': 1, 'CellPhone': 1
        }
        
        # Sample positions to test (don't test all for efficiency)
        test_positions = random.sample(reachable_positions, 
                                    min(len(reachable_positions), 30))
        
        for pos in test_positions:
            distance = self._calculate_distance(current_pos, pos)
            
            # Good navigation distance
            if not (2.5 < distance < 8.0):
                continue
            
            # Test 4 cardinal directions
            for rotation in [0, 90, 180, 270]:
                self.controller.step(
                    action="Teleport",
                    position=pos,
                    rotation={'x': 0, 'y': rotation, 'z': 0}
                )
                
                # Get frame and check immediately for wall
                frame = self.controller.last_event.frame
                if self._is_looking_at_wall(frame):
                    continue
                
                # Get visible objects
                event = self.controller.last_event
                visible_objects = []
                total_value = 0
                
                for obj in event.metadata['objects']:
                    if (obj.get('visible', False) and 
                        obj['objectType'] in high_value_objects and
                        self._is_object_in_good_position(obj)):
                        
                        visible_objects.append(obj)
                        total_value += high_value_objects[obj['objectType']]
                
                # Require at least 2 valuable objects or 1 high-value object
                if (len(visible_objects) >= 2 or 
                    (len(visible_objects) >= 1 and total_value >= 3)):
                    
                    # Final quality check
                    quality_score = self._compute_view_quality_score(
                        frame, visible_objects, total_value
                    )
                    
                    if quality_score > 30:  # Strict threshold
                        object_goals.append({
                            'position': pos,
                            'rotation': {'x': 0, 'y': rotation, 'z': 0},
                            'quality_score': quality_score,
                            'object_count': len(visible_objects),
                            'object_value': total_value,
                            'distance': distance
                        })
        
        # Restore position
        self.controller.step(
            action="Teleport",
            position=current_state['position'],
            rotation=current_state['rotation']
        )
        
        return object_goals

    def _is_looking_at_wall(self, frame):
        """Check if the view is dominated by a wall"""
        import cv2
        
        img = np.array(frame)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Check for uniform color (walls tend to be uniform)
        color_std = np.std(img.reshape(-1, 3), axis=0).mean()
        if color_std < 15:  # Very uniform color
            return True
        
        # Check for lack of edges (walls have few edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        if edge_ratio < 0.02:  # Very few edges
            return True
        
        # Check for large uniform regions
        # Divide image into 9 regions (3x3 grid)
        h, w = gray.shape
        region_size = h // 3, w // 3
        
        uniform_regions = 0
        for i in range(3):
            for j in range(3):
                region = gray[i*region_size[0]:(i+1)*region_size[0],
                            j*region_size[1]:(j+1)*region_size[1]]
                if np.std(region) < 10:
                    uniform_regions += 1
        
        if uniform_regions >= 6:  # Most regions are uniform
            return True
        
        return False

    def _is_object_in_good_position(self, obj):
        """Check if object is well-positioned in the frame"""
        if 'screenPosition' not in obj:
            return True  # If no screen position, assume it's okay
        
        x = obj['screenPosition']['x']
        y = obj['screenPosition']['y']
        
        # Object should not be at the very edges
        return 0.1 < x < 0.9 and 0.1 < y < 0.9

    def _compute_view_quality_score(self, frame, visible_objects, object_value):
        """Compute a comprehensive quality score for the view"""
        import cv2
        
        score = 0
        
        # Base score from objects
        score += len(visible_objects) * 10
        score += object_value * 5
        
        # Visual quality analysis
        img = np.array(frame)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Texture complexity
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var > 500:
            score += 20
        elif laplacian_var > 200:
            score += 10
        
        # Color diversity
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        unique_hues = len(np.unique(hsv[:, :, 0] // 10))  # Quantize hue
        if unique_hues > 10:
            score += 15
        
        # Penalty for too much uniform area (likely walls)
        uniform_pixels = np.sum(cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, 
                                                np.ones((3, 3))) < 5)
        uniform_ratio = uniform_pixels / gray.size
        if uniform_ratio > 0.5:
            score -= 30
        
        # Bonus for central objects
        central_objects = sum(1 for obj in visible_objects 
                            if 'screenPosition' in obj and
                            0.3 < obj['screenPosition']['x'] < 0.7 and
                            0.3 < obj['screenPosition']['y'] < 0.7)
        score += central_objects * 10
        
        return score

    def _get_emergency_object_goals(self, reachable_positions, current_pos):
        """Fallback: find any position with at least one meaningful object"""
        emergency_goals = []
        
        current_state = {
            'position': current_pos,
            'rotation': self.controller.last_event.metadata['agent']['rotation']
        }
        
        # Any object except walls/floor/ceiling
        any_objects = set(self.controller.last_event.metadata['objects'][0].keys()) - {
            'Wall', 'Floor', 'Ceiling', 'window', 'doorway', 'doorframe'
        }
        
        sample_positions = random.sample(reachable_positions, 
                                    min(len(reachable_positions), 20))
        
        for pos in sample_positions:
            distance = self._calculate_distance(current_pos, pos)
            if distance < 2.0:
                continue
                
            for rotation in [0, 90, 180, 270]:
                self.controller.step(
                    action="Teleport",
                    position=pos,
                    rotation={'x': 0, 'y': rotation, 'z': 0}
                )
                
                frame = self.controller.last_event.frame
                if self._is_looking_at_wall(frame):
                    continue
                
                # Check for ANY visible object
                has_object = False
                for obj in self.controller.last_event.metadata['objects']:
                    if (obj.get('visible', False) and 
                        obj['objectType'] not in ['Wall', 'Floor', 'Ceiling']):
                        has_object = True
                        break
                
                if has_object:
                    emergency_goals.append({
                        'position': pos,
                        'rotation': {'x': 0, 'y': rotation, 'z': 0}
                    })
        
        # Restore position
        self.controller.step(
            action="Teleport",
            position=current_state['position'],
            rotation=current_state['rotation']
        )
        
        return emergency_goals

    def _get_object_based_goals(self, reachable_positions, current_pos):
        """Find positions with good views of objects"""
        object_goals = []
        
        # Get all objects in scene
        event = self.controller.last_event
        objects = event.metadata['objects']
        
        # Define interesting object types
        interesting_objects = [
            'Television', 'Laptop', 'Book', 'CellPhone', 'AlarmClock',
            'Apple', 'Bowl', 'Cup', 'Plate', 'Pot', 'Pan',
            'Microwave', 'Toaster', 'CoffeeMachine', 'Sink',
            'HousePlant', 'Painting', 'Mirror', 'Lamp',
            'Sofa', 'Chair', 'Bed', 'DiningTable', 'Desk'
        ]
        
        # Filter for visible, interesting objects
        target_objects = [
            obj for obj in objects
            if obj['objectType'] in interesting_objects and obj.get('visible', False)
        ]
        
        # Store current state
        current_state = {
            'position': current_pos,
            'rotation': event.metadata['agent']['rotation']
        }
        
        # Test positions for object visibility
        for pos in reachable_positions:
            distance = self._calculate_distance(current_pos, pos)
            
            # Skip if too close or too far
            if distance < 2.0 or distance > 10.0:
                continue
            
            # Test multiple rotations at this position
            for rotation in [0, 90, 180, 270]:
                self.controller.step(
                    action="Teleport",
                    position=pos,
                    rotation={'x': 0, 'y': rotation, 'z': 0}
                )
                
                # Check what objects are visible from this viewpoint
                visible_objects = self._get_visible_objects_info()
                
                if visible_objects['count'] >= 2 and visible_objects['diversity'] >= 2:
                    # Capture the view for quality check
                    frame = self.controller.last_event.frame
                    
                    # Check visual quality
                    if self._check_visual_quality(frame):
                        object_goals.append({
                            'position': pos,
                            'rotation': {'x': 0, 'y': rotation, 'z': 0},
                            'visible_objects': visible_objects,
                            'distance': distance,
                            'frame': frame
                        })
        
        self.controller.step(
            action="Teleport",
            position=current_state['position'],
            rotation=current_state['rotation']
        )
        
        return object_goals

    def _get_visible_objects_info(self):
        """Get information about currently visible objects"""
        event = self.controller.last_event
        objects = event.metadata['objects']
        
        visible_objects = []
        object_types = set()
        
        for obj in objects:
            if obj.get('visible', False) and obj['objectType'] not in ['Wall', 'Floor', 'Ceiling']:
                # Check if object is in central area of view
                if 'screenPosition' in obj:
                    x, y = obj['screenPosition']['x'], obj['screenPosition']['y']
                    # Object should be somewhat centered (not at edges)
                    if 0.2 < x < 0.8 and 0.2 < y < 0.8:
                        visible_objects.append(obj)
                        object_types.add(obj['objectType'])
        
        return {
            'count': len(visible_objects),
            'diversity': len(object_types),
            'objects': visible_objects,
            'types': list(object_types)
        }

    def _check_visual_quality(self, image):
        """Check if image has good visual features"""
        import cv2
        
        # Convert to numpy array if needed
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Multiple quality metrics
        
        # 1. Texture richness using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_score = laplacian.var()
        
        # 2. Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 3. Color diversity
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        hist = hist.flatten() / hist.sum()
        color_diversity = -np.sum(hist * np.log(hist + 1e-7))  # Entropy
        
        # 4. Contrast
        contrast = gray.std()
        
        # Combined quality check
        return (
            texture_score > 200 and 
            edge_density > 0.08 and 
            color_diversity > 2.5 and
            contrast > 30
        )

    def _score_goal_position(self, goal_data, current_pos):
        """Score a potential goal position based on multiple factors"""
        score = 0
        
        # Distance factor (prefer moderate distances)
        distance = goal_data['distance']
        if 3 < distance < 7:
            score += 20
        elif 2 < distance < 10:
            score += 10
        
        # Object visibility factor
        visible_info = goal_data['visible_objects']
        score += visible_info['count'] * 5
        score += visible_info['diversity'] * 10
        
        # Prefer certain object types
        priority_objects = ['Television', 'Laptop', 'Painting', 'Mirror', 'Book']
        for obj_type in visible_info['types']:
            if obj_type in priority_objects:
                score += 15
        
        # Visual complexity bonus
        frame = goal_data['frame']
        if self._check_visual_quality(frame):
            score += 25
        
        return score

    def _set_goal_from_position(self, position, rotation):
        """Set goal image from a specific position and rotation"""
        # Store current state
        current_state = self.controller.last_event.metadata['agent']
        
        # Teleport to goal position
        self.controller.step(
            action="Teleport",
            position=position,
            rotation=rotation
        )
        
        # Capture goal image
        goal_event = self.controller.last_event
        self.goal_image = goal_event.frame
        self.goal_position = np.array([position['x'], position['y'], position['z']])
        
        # Teleport back
        self.controller.step(
            action="Teleport",
            position=current_state['position'],
            rotation=current_state['rotation']
        )

    # def _has_sufficient_visual_features(self, image):
    #     import cv2
    #     gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        
    #     # Ti'nh feature dung` Laplacian variance
    #     laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
    #     # Calculate edge density
    #     edges = cv2.Canny(gray, 50, 150)
    #     edge_density = np.sum(edges > 0) / edges.size
        
    #     return laplacian_var > 100 and edge_density > 0.05

    # def _is_easily_accessible(self, goal_pos):
    #     """Check if position is in open area, not corner or behind obstacles"""
    #     check_offsets = [(0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)]
    #     accessible_count = 0
        
    #     for dx, dz in check_offsets:
    #         test_pos = {
    #             'x': goal_pos['x'] + dx,
    #             'y': goal_pos['y'], 
    #             'z': goal_pos['z'] + dz
    #         }
            
    #         # Try to move to nearby position
    #         result = self.controller.step(
    #             action="Teleport",
    #             position=test_pos
    #         )
            
    #         if result.metadata["lastActionSuccess"]:
    #             accessible_count += 1
        
    #     self.controller.step(action="Teleport", position=goal_pos)
        
    #     # At least 3 out of 4 surrounding positions should be accessible
    #     return accessible_count >= 3

    def _calculate_reward(self, event, obs) -> float:
        reward = 0.0
        reward -= 0.01  # step penalty
        
        # Heavy penalty for being stuck
        if self.stuck_counter > 0:
            reward -= self.stuck_counter * 0.5  # Increasing penalty
            
        # Penalty for consecutive collisions
        if self.consecutive_collisions > 0:
            reward -= self.consecutive_collisions * 0.3
        
        if self.is_goal_conditioned:
            distance = self._distance_to_goal()
            distance = min(distance, 10.0)

            if hasattr(self, '_prev_distance_to_goal'):
                distance_improvement = self._prev_distance_to_goal - distance
                distance_improvement = np.clip(distance_improvement, -1.0, 1.0)
                
                # Only give distance reward if not stuck
                if self.stuck_counter == 0:
                    reward += distance_improvement * 5.0
                    if distance_improvement > 0.01:
                        reward += 0.5
                        
            self._prev_distance_to_goal = distance
            
            # Progressive rewards only if making progress
            if self.consecutive_collisions == 0:
                if distance < 5.0:
                    reward += 0.2
                if distance < 3.0:
                    reward += 0.3
                if distance < 2.0:
                    reward += 0.5
                if distance < 1.5:
                    reward += 1.0
            
            if distance < self.success_distance:
                reward += 50.0
                print(f"SUCCESS! Distance: {distance:.2f}")
        else:
            # Exploration mode
            current_pos = self._get_agent_position()
            pos_key = (round(current_pos['x'], 1), round(current_pos['z'], 1))
            
            # Only reward exploration if not stuck
            if self.stuck_counter == 0:
                visit_count = self.position_visit_counts.get(pos_key, 0)
                if visit_count == 0:
                    reward += 1.0
                elif visit_count < 3:
                    reward += 0.5 / visit_count
                self.position_visit_counts[pos_key] = visit_count + 1

            elif self.stuck_counter > 10:  # Really stuck
                print("Agent stuck - forcing random teleport")
                reachable = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
                current_pos = self._get_agent_position()
                
                # Find positions within reasonable distance
                nearby_positions = [
                    pos for pos in reachable
                    if 1.0 < self._calculate_distance(current_pos, pos) < 3.0
                ]
                
                if nearby_positions:
                    new_pos = random.choice(nearby_positions)
                    self.controller.step(
                        action="Teleport",
                        position=new_pos,
                        rotation=dict(x=0, y=random.choice([0, 90, 180, 270]), z=0)
                    )
                    self.stuck_counter = 0
                    self.consecutive_collisions = 0
                    self.last_positions.clear
                    
                    reward -= 2.0

        if not event.metadata['lastActionSuccess']:
            if self.consecutive_collisions > 3:
                reward -= 2.0  # Bigger penalty for repeated collisions
            else:
                reward -= 0.5  # Smaller initial penalty
        
        # Bonus for successful movement after collision
        if event.metadata['lastActionSuccess'] and len(self.collision_history) > 0:
            if self.collision_history[-1] == self.last_action:
                reward += 0.5  # Recovered from collision
        
        # Movement bonus (only if not stuck)
        if event.metadata['lastActionSuccess'] and self.stuck_counter == 0:
            if event.metadata['lastAction'] in ['MoveAhead', 'MoveBack']:
                reward += 0.1
        
        return np.clip(reward, -5.0, 5.0)
    
    def _is_done(self, obs) -> bool:
        """Check if episode is done"""
        if self.current_step >= self.max_episode_steps:
            return True
        
        if self.is_goal_conditioned:
            return self._distance_to_goal() < self.success_distance
        
        return False
    
    def _is_success(self, obs) -> bool:
        """Check if episode was successful"""
        if self.is_goal_conditioned:
            distance = self._distance_to_goal()
            success = distance < self.success_distance
            if distance < 2.0 and not success:
                print(f"Near goal but not success: distance={distance:.2f}, threshold={self.success_distance}")
            if success:
                print(f"SUCCESS DETECTED! Distance: {distance:.2f}")
            return success
        else:
            # For exploration, success = visiting many positions
            return len(self.visited_positions) > 10
    
    def _distance_to_goal(self) -> float:
        if self.goal_position is None:
            return float('inf')
        
        current_pos = self._get_agent_position()
        return self._calculate_distance(current_pos, {
            'x': self.goal_position[0],
            'y': self.goal_position[1],
            'z': self.goal_position[2]
        })
    
    def _get_agent_position(self) -> Dict[str, float]:
        try:
            agent_metadata = self.controller.last_event.metadata['agent']
            position = agent_metadata['position']
            print(f"DEBUG: Raw agent position: {position}")
            return position
        except Exception as e:
            print(f"ERROR getting agent position: {e}")
            return {'x': 0.0, 'y': 0.0, 'z': 0.0}
        
    def _calculate_distance(self, pos1, pos2) -> float:
        return np.sqrt(
            (pos1['x'] - pos2['x'])**2 +
            (pos1['z'] - pos2['z'])**2
        )

    def close(self):
        if self.controller:
            self.controller.stop()
    
    def render(self, mode='human'):
        if mode == 'human':
            event = self.controller.last_event
            return np.array(event.frame)
        return None
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")