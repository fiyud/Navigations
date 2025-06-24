import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from torch.distributions import Categorical
from efficientnet_pytorch import EfficientNet

import sys
sys.path.append(r'/home/tuandang/tuandang/quanganh/visualnav-transformer/train')

from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

class EnhancedNoMaDRL(nn.Module):
    def __init__(
        self,
        action_dim: int = 4,
        encoding_size: int = 256,
        context_size: int = 5,
        mha_num_attention_heads: int = 4,
        mha_num_attention_layers: int = 4,
        mha_ff_dim_factor: int = 4,
        hidden_dim: int = 512,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        use_auxiliary_heads: bool = True,
    ):
        super(EnhancedNoMaDRL, self).__init__()
        
        self.action_dim = action_dim
        self.encoding_size = encoding_size
        self.context_size = context_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.use_auxiliary_heads = use_auxiliary_heads
        
        self.vision_encoder = NoMaD_ViNT(
            obs_encoding_size=encoding_size,
            context_size=context_size,
            mha_num_attention_heads=mha_num_attention_heads,
            mha_num_attention_layers=mha_num_attention_layers,
            mha_ff_dim_factor=mha_ff_dim_factor,
        )
        self.vision_encoder = replace_bn_with_gn(self.vision_encoder)
        
        self.lstm = nn.LSTM(
            input_size=encoding_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0.1 if lstm_num_layers > 1 else 0
        )
        
        self.policy_net = PolicyNetwork(
            input_dim=lstm_hidden_size,
            hidden_dim=hidden_dim,
            action_dim=action_dim
        )
        
        self.value_net = ValueNetwork(
            input_dim=lstm_hidden_size,
            hidden_dim=hidden_dim
        )
        
        self.dist_pred_net = DenseNetwork(embedding_dim=lstm_hidden_size)
        
        if use_auxiliary_heads:
            self.collision_predictor = nn.Sequential(
                nn.Linear(lstm_hidden_size, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            
            self.exploration_predictor = nn.Sequential(
                nn.Linear(lstm_hidden_size, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        self.hidden_state = None
        
        for param in self.lstm.parameters():
            param.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    def forward(self, observations: Dict[str, torch.Tensor], mode: str = "policy", 
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        obs_img = observations['context']
        goal_img = observations['goal_rgb']
        goal_mask = observations['goal_mask']
        
        # Input validation
        for key, value in observations.items():
            if torch.isnan(value).any() or torch.isinf(value).any():
                print(f"WARNING: Invalid values in input {key}")
                observations[key] = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0)
        
        last_obs_frame = obs_img[:, -3:, :, :]
        obsgoal_img = torch.cat([last_obs_frame, goal_img], dim=1)
        
        try:
            # Get vision features with error handling
            vision_features = self.vision_encoder(
                obs_img=obs_img,
                goal_img=obsgoal_img,
                input_goal_mask=goal_mask.long().squeeze(-1)
            )
        except Exception as e:
            print(f"Error in vision encoder: {e}")
            batch_size = obs_img.size(0)
            vision_features = torch.zeros(batch_size, self.encoding_size, device=obs_img.device)
        
        if torch.isnan(vision_features).any():
            print("WARNING: NaN in vision features, replacing with zeros")
            vision_features = torch.nan_to_num(vision_features, nan=0.0)

        # Normalize for anti explode
        vision_features = vision_features / (vision_features.norm(dim=-1, keepdim=True) + 1e-8)

        batch_size = vision_features.size(0)
        vision_features = vision_features.unsqueeze(1)  # Add sequence dimension
        
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size, vision_features.device)
        
        # Clamp hid state to prevent explosion
        h, c = hidden_state
        h = torch.clamp(h, -10, 10)
        c = torch.clamp(c, -10, 10)
        hidden_state = (h, c)

        try:
            lstm_out, new_hidden_state = self.lstm(vision_features, hidden_state)
            lstm_features = lstm_out.squeeze(1)
        except Exception as e:
            print(f"LSTM error: {e}")
            lstm_features = vision_features.squeeze(1)
            new_hidden_state = hidden_state
        
        # Normalize LSTM features
        if torch.isnan(lstm_features).any():
            print("WARNING: NaN in LSTM features")
            lstm_features = torch.zeros_like(lstm_features)
        
        lstm_features = torch.clamp(lstm_features, -10, 10)
        
        self.hidden_state = new_hidden_state
        
        results = {'hidden_state': new_hidden_state}
        
        if mode == "policy" or mode == "all":
            try:
                policy_logits = self.policy_net(lstm_features)
                # Prevent extreme values
                policy_logits = torch.clamp(policy_logits, -10, 10)
                
                if torch.isnan(policy_logits).any():
                    print("WARNING: NaN in policy logits, using uniform distribution")
                    policy_logits = torch.zeros_like(policy_logits)
                
                results['policy_logits'] = policy_logits
                results['action_dist'] = Categorical(logits=policy_logits)
            except Exception as e:
                print(f"Policy network error: {e}")
                # Return uniform distribution
                batch_size = lstm_features.size(0)
                policy_logits = torch.zeros(batch_size, self.action_dim, device=lstm_features.device)
                results['policy_logits'] = policy_logits
                results['action_dist'] = Categorical(logits=policy_logits)
            
        if mode == "value" or mode == "all":
            try:
                values = self.value_net(lstm_features)
                values = torch.clamp(values, -100, 100)
                results['values'] = values
            except Exception as e:
                print(f"Value network error: {e}")
                results['values'] = torch.zeros(batch_size, device=lstm_features.device)
            
        if mode == "distance" or mode == "all":
            try:
                distances = self.dist_pred_net(lstm_features)
                distances = torch.clamp(distances, 0, 100)
                results['distances'] = distances
            except Exception as e:
                print(f"Distance network error: {e}")
                results['distances'] = torch.zeros(batch_size, 1, device=lstm_features.device)
            
        if self.use_auxiliary_heads and (mode == "auxiliary" or mode == "all"):
            try:
                collision_pred = self.collision_predictor(lstm_features)
                exploration_pred = self.exploration_predictor(lstm_features)
                results['collision_prediction'] = torch.clamp(collision_pred, 0, 1)
                results['exploration_prediction'] = torch.clamp(exploration_pred, -10, 10)
            except Exception as e:
                print(f"Auxiliary heads error: {e}")
                
        if mode == "features":
            results['features'] = lstm_features
            
        return results
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        c = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device)
        return (h, c)
    
    def reset_hidden(self):
        self.hidden_state = None
    
    def get_action(self, observations: Dict[str, torch.Tensor], deterministic: bool = False,
                   hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        with torch.no_grad():
            outputs = self.forward(observations, mode="policy", hidden_state=hidden_state)
            action_dist = outputs['action_dist']
            
            if deterministic:
                action = action_dist.probs.argmax(dim=-1)
            else:
                action = action_dist.sample()
                
            log_prob = action_dist.log_prob(action)
        return action, log_prob, outputs['hidden_state']

    def evaluate_actions(self, observations: Dict[str, torch.Tensor], actions: torch.Tensor):
        try:
            outputs = self.forward(observations, mode="all")
            
            action_dist = outputs['action_dist']
            log_probs = action_dist.log_prob(actions)
            entropy = action_dist.entropy()
            values = outputs['values']
            
            # Safety checks
            log_probs = torch.clamp(log_probs, -20, 2)
            entropy = torch.clamp(entropy, 0, 10)
            values = torch.clamp(values, -100, 100)
            
            # Replace any remaining NaN
            log_probs = torch.nan_to_num(log_probs, nan=-10.0)
            entropy = torch.nan_to_num(entropy, nan=0.0)
            values = torch.nan_to_num(values, nan=0.0)
            
            return log_probs, values, entropy
            
        except Exception as e:
            print(f"Error in evaluate_actions: {e}")
            batch_size = actions.size(0)
            device = actions.device
            return (
                torch.zeros(batch_size, device=device),
                torch.zeros(batch_size, device=device),
                torch.ones(batch_size, device=device)
            )
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state with small values"""
        h = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device) * 0.01
        c = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size).to(device) * 0.01
        return (h, c)
    
    def reset_hidden(self):
        """Reset stored hidden state"""
        self.hidden_state = None
        
    def calculate_reward(self, event, obs, info: Dict, env) -> float:
        reward = 0.0
        
        # Step penalty (efficiency)
        reward -= self.step_penalty
        
        if env.is_goal_conditioned:
            distance = env._distance_to_goal()

            if hasattr(env, '_prev_distance_to_goal'):
                distance_improvement = env._prev_distance_to_goal - distance
                reward += distance_improvement * self.distance_weight
            
            env._prev_distance_to_goal = distance

            if distance < env.success_distance:
                reward += self.success_reward
            
            if distance < 5.0:
                reward += 1.0
            if distance < 3.0:
                reward += 2.0
            if distance < 2.0:
                reward += 5.0

        else:
            # Exploration rewards
            current_pos = env._get_agent_position()
            pos_key = (round(current_pos['x'], 1), round(current_pos['z'], 1))
            
            # Coverage reward with diminishing returns
            visit_count = env.position_visit_counts.get(pos_key, 0)
            if visit_count == 0:
                reward += self.exploration_bonus
            elif visit_count < 3:
                reward += self.exploration_bonus / (visit_count + 1)
            
            env.position_visit_counts[pos_key] = visit_count + 1
            
            # Movement reward
            if event.metadata['lastAction'] == 'MoveAhead' and event.metadata['lastActionSuccess']:
                reward += 0.05
        
        # Stage-specific rewards
        if self.stage == 1:
            # Stage 1: Focus on exploration, no collision penalty
            if not event.metadata['lastActionSuccess']:
                reward -= 0.1  # Minimal penalty
        else:
            # Stage 2: Add safety constraints
            if not event.metadata['lastActionSuccess']:
                reward -= self.collision_penalty
        
        return reward


class CurriculumManager:
    def __init__(self, config: Dict):
        self.config = config
        self.current_level = 0
        self.success_threshold = config.get('curriculum_success_threshold', 0.7)
        self.window_size = config.get('curriculum_window_size', 100)
        self.recent_successes = []
        
        # Curriculum levels
        self.levels = [
            {'max_distance': 2.0, 'scene_complexity': 'simple', 'goal_prob': 0.3},
            {'max_distance': 4.0, 'scene_complexity': 'simple', 'goal_prob': 0.5},
            {'max_distance': 6.0, 'scene_complexity': 'medium', 'goal_prob': 0.7},
            {'max_distance': 10.0, 'scene_complexity': 'complex', 'goal_prob': 0.8},
            {'max_distance': float('inf'), 'scene_complexity': 'complex', 'goal_prob': 0.9},
        ]
        
    def get_current_settings(self) -> Dict:
        return self.levels[min(self.current_level, len(self.levels) - 1)]
    
    def update(self, success: bool):
        self.recent_successes.append(success)
        if len(self.recent_successes) > self.window_size:
            self.recent_successes.pop(0)
        
        # Check if we should advance
        if len(self.recent_successes) >= self.window_size:
            success_rate = sum(self.recent_successes) / len(self.recent_successes)
            if success_rate >= self.success_threshold:
                self.advance_level()
                self.recent_successes = []
    
    def advance_level(self):
        if self.current_level < len(self.levels) - 1:
            self.current_level += 1
            print(f"Advancing to curriculum level {self.current_level + 1}")


class EvaluationMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.episodes = []
        self.current_episode = {
            'success': False,
            'collision_count': 0,
            'path_length': 0,
            'reward': 0,
            'visited_positions': set(),
            'distance_to_goal': [],
        }
    
    def step(self, info: Dict, reward: float, position: Tuple):
        self.current_episode['path_length'] += 1
        self.current_episode['reward'] += reward
        self.current_episode['visited_positions'].add(position)
        
        if info.get('collision', False):
            self.current_episode['collision_count'] += 1
        
        if 'distance_to_goal' in info:
            self.current_episode['distance_to_goal'].append(info['distance_to_goal'])
    
    def end_episode(self, success: bool):
        self.current_episode['success'] = success
        self.episodes.append(self.current_episode.copy())
        self.reset()
    
    def compute_metrics(self) -> Dict:
        if not self.episodes:
            return {}
        
        metrics = {
            # Core metrics
            'success_rate': sum(ep['success'] for ep in self.episodes) / len(self.episodes),
            'avg_reward': np.mean([ep['reward'] for ep in self.episodes]),
            'avg_path_length': np.mean([ep['path_length'] for ep in self.episodes]),
            
            # Safety metrics
            'collision_free_success_rate': sum(
                ep['success'] and ep['collision_count'] == 0 
                for ep in self.episodes
            ) / len(self.episodes),
            'avg_collisions_per_episode': np.mean([ep['collision_count'] for ep in self.episodes]),
            
            # Efficiency metrics
            'exploration_coverage': np.mean([
                len(ep['visited_positions']) / ep['path_length'] 
                for ep in self.episodes if ep['path_length'] > 0
            ]),
        }
        
        # SPL (Success weighted by Path Length)
        successful_episodes = [ep for ep in self.episodes if ep['success']]
        if successful_episodes:
            metrics['spl'] = np.mean([
                1.0 / max(1, ep['path_length'] / 10)  # Approximate optimal path
                for ep in successful_episodes
            ])
        else:
            metrics['spl'] = 0.0
        
        return metrics

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int = 4):
        super(PolicyNetwork, self).__init__()
        self.action_dim = action_dim
        
        # Add normalization layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.out_features == self.action_dim:
                # Very small initialization for output layer
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
            else:
                nn.init.xavier_normal_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class DenseNetwork(nn.Module):
    def __init__(self, embedding_dim: int):
        super(DenseNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim 
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim//4),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//4, self.embedding_dim//16),
            nn.ReLU(),
            nn.Linear(self.embedding_dim//16, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape((-1, self.embedding_dim))
        output = self.network(x)
        return output

def prepare_observation(obs: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    torch_obs = {}
    
    required_keys = ['rgb', 'goal_rgb', 'context', 'goal_mask', 'goal_position']
    for key in required_keys:
        if key not in obs:
            raise KeyError(f"Missing required observation key: {key}")
    
    for key, value in obs.items():
        if isinstance(value, torch.Tensor):
            tensor = value.float()
        else:
            tensor = torch.from_numpy(value).float()
        
        if key in ['rgb', 'goal_rgb', 'context']:
            tensor = tensor / 255.0
        torch_obs[key] = tensor.to(device)
        
        if torch_obs[key].dim() == 3:
            torch_obs[key] = torch_obs[key].unsqueeze(0)
        elif torch_obs[key].dim() == 1:
            torch_obs[key] = torch_obs[key].unsqueeze(0)
    
    return torch_obs