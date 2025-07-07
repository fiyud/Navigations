# enhanced_ai2thor_env.py - Environment that handles both iTHOR and RoboTHOR scenes

import ai2thor
from ai2thor.controller import Controller
import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Optional, Any
import random
import gym
from gym import spaces

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
        
        # Separate iTHOR and RoboTHOR scenes
        self.scene_names = scene_names if scene_names else []
        self.ithor_scenes = []
        self.robothor_scenes = []
                
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
        
    def _is_robothor_scene(self, scene_name: str) -> bool:
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
        else:
            self.goal_position = None
            self.goal_image = None

        print(f"Episode type: {'GOAL' if self.is_goal_conditioned else 'EXPLORATION'}, "
            f"goal_prob: {self.goal_prob}, random: {random.random()}")

        # Get init observation
        obs = self._get_observation()
        
        for _ in range(self.context_size):
            self.context_buffer.append(obs['rgb'])
        
        return self._format_observation(obs)
    
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
        action_map = {
            0: {"action": "MoveAhead", "moveMagnitude": 0.5},
            1: {"action": "MoveBack", "moveMagnitude": 0.5},
            2: {"action": "RotateLeft", "degrees": 30},   # Smaller rotation
            3: {"action": "RotateRight", "degrees": 30},
            4: {"action": "MoveAhead", "moveMagnitude": 1.0},  # Longer step
            5: {"action": "RotateLeft", "degrees": 60},   # Larger rotation
            6: {"action": "RotateRight", "degrees": 60},
        }
        
        try:
            event = self.controller.step(action=action_map[action])
        except Exception as e:
            print(f"AI2THOR step failed: {e}")
            # Return a failed step
            obs = self._get_observation()
            return self._format_observation(obs), -1.0, True, {'error': str(e)}
        
        self.current_step += 1
        
        # Get observation
        obs = self._get_observation()
        
        # Update context buffer
        self.context_buffer.append(obs['rgb'])
        if len(self.context_buffer) > self.context_size:
            self.context_buffer.pop(0)
        
        reward = self._calculate_reward(event, obs)
        done = self._is_done(obs)

        info = {
            'success': self._is_success(obs),
            'collision': not event.metadata['lastActionSuccess'],
            'step': self.current_step,
            'goal_conditioned': self.is_goal_conditioned,
            'distance_to_goal': self._distance_to_goal() if self.is_goal_conditioned else 0.0,
            'scene_type': 'robothor' if (self.current_scene and self._is_robothor_scene(self.current_scene)) else 'ithor'
        }
        
        return self._format_observation(obs), reward, done, info
    
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
        
        return {
            'rgb': rgb_image,
            'goal_rgb': goal_rgb,
            'position': np.array([
                event.metadata['agent']['position']['x'],
                event.metadata['agent']['position']['y'],
                event.metadata['agent']['position']['z']
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
                'goal_mask': np.array([0.0 if self.is_goal_conditioned else 1.0], dtype=np.float32),
                # 'goal_mask': np.array([1.0 if self.is_goal_conditioned else 0.0], dtype=np.float32),
                'goal_position': self.goal_position if self.goal_position is not None else np.zeros(3, dtype=np.float32),
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
        try:
            reachable_positions = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        except:
            reachable_positions = self._get_robothor_reachable_positions()
        
        if reachable_positions:
            current_pos = self._get_agent_position()
            
            valid_goals = [
                pos for pos in reachable_positions
                if self._calculate_distance(current_pos, pos) > 2.0
            ]
            
            if valid_goals:
                goal_pos = random.choice(valid_goals)
                self.goal_position = np.array([goal_pos['x'], goal_pos['y'], goal_pos['z']])
                
                # Capture goal image
                current_state = self.controller.last_event.metadata['agent']
                self.controller.step(
                    action="Teleport",
                    position=goal_pos,
                    rotation=dict(x=0, y=random.choice([0, 90, 180, 270]), z=0)
                )
                goal_event = self.controller.last_event
                self.goal_image = goal_event.frame
                
                # Teleport back
                self.controller.step(
                    action="Teleport",
                    position=current_state['position'],
                    rotation=current_state['rotation']
                )
    
    def _calculate_reward(self, event, obs) -> float:
        reward = 0.0
        reward -= 0.01  # step penalty
        
        if self.is_goal_conditioned and self.current_step % 50 == 0:
            print(f"Goal distance: {self._distance_to_goal():.2f}, "
                f"Success threshold: {self.success_distance}, "
                f"Agent pos: {self._get_agent_position()}")
    
        if self.is_goal_conditioned:
            distance = self._distance_to_goal()
            
            distance = min(distance, 20.0)

            # Stronger distance-based reward
            if hasattr(self, '_prev_distance_to_goal'):
                distance_improvement = self._prev_distance_to_goal - distance
                distance_improvement = np.clip(distance_improvement, -0.5, 0.5)
                reward += distance_improvement * 10.0  # Increased from 20.0

                if abs(reward) > 50:
                    print(f"Large reward: {reward:.2f}, dist improvement: {distance_improvement:.2f}")

                if self.current_step % 50 == 0:
                    print(f"Step {self.current_step}: distance={distance:.2f}, improvement={distance_improvement:.2f}, reward={reward:.2f}")

            self._prev_distance_to_goal = distance
            
            # Add intermediate rewards
            if distance < 3.0:
                reward += 0.5
            if distance < 1.5:
                reward += 1.0
            
            if distance < self.success_distance:
                reward += 50.0  # Reduced further
                print(f"SUCCESS! Distance: {distance:.2f}")
        else:
            # Exploration mode
            current_pos = self._get_agent_position()
            pos_key = (round(current_pos['x'], 1), round(current_pos['z'], 1))
            
            visit_count = self.position_visit_counts.get(pos_key, 0)
            if visit_count == 0:
                reward += 10.0  # Increased from 5.0
            elif visit_count < 3:
                reward += 3.0 / visit_count  # Increased
            
            self.position_visit_counts[pos_key] = visit_count + 1
        
        # Collision penalty
        if not event.metadata['lastActionSuccess']:
            reward -= 5.0  # Increased from 1.0
        
        # Movement bonus to prevent spinning
        if event.metadata['lastAction'] in ['MoveAhead', 'MoveBack'] and event.metadata['lastActionSuccess']:
            reward += 0.2
        
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
        return self.controller.last_event.metadata['agent']['position']
    
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
    
