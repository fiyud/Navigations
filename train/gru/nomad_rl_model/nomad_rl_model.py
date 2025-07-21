import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from torch.distributions import Categorical
from efficientnet_pytorch import EfficientNet

import sys
sys.path.append(r'/home/tuandang/tuandang/quanganh/visualnav-transformer/train')

from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

class NoMaDRL(nn.Module):
    def __init__(
        self,
        action_dim: int = 4,
        encoding_size: int = 256,
        context_size: int = 5,
        mha_num_attention_heads: int = 4,
        mha_num_attention_layers: int = 4,
        mha_ff_dim_factor: int = 4,
        hidden_dim: int = 512,
    ):
        super(NoMaDRL, self).__init__()
        
        self.action_dim = action_dim
        self.encoding_size = encoding_size
        self.context_size = context_size
        
        self.vision_encoder = NoMaD_ViNT(
            obs_encoding_size=encoding_size,
            context_size=context_size,
            mha_num_attention_heads=mha_num_attention_heads,
            mha_num_attention_layers=mha_num_attention_layers,
            mha_ff_dim_factor=mha_ff_dim_factor,
        )
        self.vision_encoder = replace_bn_with_gn(self.vision_encoder)
        
        self.dist_pred_net = DenseNetwork(embedding_dim=encoding_size)
        self.policy_net = PolicyNetwork(
            input_dim=encoding_size,
            hidden_dim=hidden_dim,
            action_dim=action_dim
        )
        
        self.value_net = ValueNetwork(
            input_dim=encoding_size,
            hidden_dim=hidden_dim
        )
        
    def forward(self, observations: Dict[str, torch.Tensor], mode: str = "policy"):
        obs_img = observations['context']
        goal_img = observations['goal_rgb']
        goal_mask = observations['goal_mask']
        
        last_obs_frame = obs_img[:, -3:, :, :]
        obsgoal_img = torch.cat([last_obs_frame, goal_img], dim=1)
        
        # print(f"obs_img shape: {obs_img.shape}")
        # print(f"goal_img shape: {goal_img.shape}")
        # print(f"obsgoal_img shape: {obsgoal_img.shape}")
        
        vision_features = self.vision_encoder(
            obs_img=obs_img,
            goal_img=obsgoal_img,
            input_goal_mask=goal_mask.long().squeeze(-1)
        )
        
        if torch.isnan(vision_features).any():
            print("Warning: NaN detected in vision_features")
        
        results = {}
        
        if mode == "policy" or mode == "all":
            policy_logits = self.policy_net(vision_features)
            if torch.isnan(policy_logits).any():
                print("Warning: NaN detected in policy_logits")
            results['policy_logits'] = policy_logits
            results['action_dist'] = Categorical(logits=policy_logits)
            
        if mode == "value" or mode == "all":
            values = self.value_net(vision_features)
            results['values'] = values
            
        if mode == "distance" or mode == "all":
            distances = self.dist_pred_net(vision_features)
            results['distances'] = distances
            
        if mode == "features":
            results['features'] = vision_features
            
        return results
    
    def get_action(self, observations: Dict[str, torch.Tensor], deterministic: bool = False):
        with torch.no_grad():
            outputs = self.forward(observations, mode="policy")
            action_dist = outputs['action_dist']
            
            if deterministic:
                action = action_dist.probs.argmax(dim=-1)
            else:
                action = action_dist.sample()
                
            log_prob = action_dist.log_prob(action)
            
        return action, log_prob
    
    def evaluate_actions(self, observations: Dict[str, torch.Tensor], actions: torch.Tensor):
        outputs = self.forward(observations, mode="all")
        
        action_dist = outputs['action_dist']
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        values = outputs['values']
        
        return log_probs, values, entropy

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int = 4):  # Changed from 6 to 4
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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

class PPOBuffer:
    def __init__(self, size: int, obs_shape: Dict[str, Tuple], action_dim: int, device: torch.device):
        self.size = size
        self.ptr = 0
        self.full = False
        self.device = device
        
        self.observations = {}
        for key, shape in obs_shape.items():
            self.observations[key] = torch.zeros((size, *shape), dtype=torch.float32, device=device)
        
        self.actions = torch.zeros(size, dtype=torch.long, device=device)
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.values = torch.zeros(size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(size, dtype=torch.float32, device=device)
        self.advantages = torch.zeros(size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(size, dtype=torch.bool, device=device)
    
    def store(self, obs: Dict[str, torch.Tensor], action: torch.Tensor, reward: float, 
              value: torch.Tensor, log_prob: torch.Tensor, done: bool):
        for key, val in obs.items():
            self.observations[key][self.ptr] = val.cpu()
        
        self.actions[self.ptr] = action.cpu()
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value.cpu()
        self.log_probs[self.ptr] = log_prob.cpu()
        self.dones[self.ptr] = torch.tensor(done, dtype=torch.bool, device=self.device)
        
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True
    
    def get(self) -> Dict[str, torch.Tensor]:
        size = self.size if self.full else self.ptr
        
        batch = {
            'observations': {key: val[:size].to(self.device) for key, val in self.observations.items()},
            'actions': self.actions[:size].to(self.device),
            'rewards': self.rewards[:size].to(self.device),
            'values': self.values[:size].to(self.device),
            'log_probs': self.log_probs[:size].to(self.device),
            'advantages': self.advantages[:size].to(self.device),
            'returns': self.returns[:size].to(self.device),
            'dones': self.dones[:size].to(self.device),
        }
        
        return batch
    
    def compute_gae(self, last_value: float, gamma: float = 0.99, lam: float = 0.95):
        size = self.size if self.full else self.ptr
        
        advantages = torch.zeros_like(self.rewards[:size])
        last_advantage = 0
        
        for t in reversed(range(size)):
            if t == size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t].float()
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t].float()
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            advantages[t] = last_advantage = delta + gamma * lam * next_non_terminal * last_advantage
        
        self.advantages[:size] = advantages
        self.returns[:size] = advantages + self.values[:size]
        self.returns[:size] = torch.clamp(self.returns[:size], -100, 100)

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