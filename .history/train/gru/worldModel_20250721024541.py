import torch
from torch import nn
from typing import Dict, List, Optional

class CounterfactualWorldModel(nn.Module):
    def __init__(self, feature_dim: int = 256, action_dim: int = 4, 
                 hidden_dim: int = 512):
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Action-specific predictors
        self.future_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim),
                nn.LayerNorm(feature_dim)
            ) for _ in range(action_dim)
        ])
        
        # Collision predictor
        self.collision_predictor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(action_dim)
        ])
        
        # Information gain predictor
        self.info_gain_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive info gain
        )
        
        # Goal reachability predictor
        self.reachability_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.goal_projection = nn.Linear(1000, feature_dim)

    def forward(self, state: torch.Tensor, goal_features: Optional[torch.Tensor] = None) \
                -> List[Dict[str, torch.Tensor]]:
        """
        Args:
            state: [batch_size, feature_dim]
            goal_features: [batch_size, feature_dim] optional goal features
        Returns:
            counterfactuals: List of dicts for each action
        """
        batch_size = state.size(0)
        
        # Extract features
        features = self.feature_extractor(state)
        
        counterfactuals = []
        
        for action_idx in range(self.action_dim):
            # Predict future state
            future_state = self.future_predictors[action_idx](features)
            
            # Predict collision probability
            collision_prob = self.collision_predictor[action_idx](features)
            
            # Predict information gain
            state_change = torch.cat([state, future_state], dim=-1)
            info_gain = self.info_gain_predictor(state_change)
            
            # Predict goal reachability if goal is provided
            if goal_features is not None:
                if goal_features.dim() > 2:
                    goal_features = goal_features.squeeze(1)
                
                # Ensure goal_features has the right size
                if goal_features.size(-1) != self.feature_dim:
                    if goal_features.size(-1) == 1000:  # Original goal image features
                        projected_goal_features = self.goal_projection(goal_features)
                    else:
                        # If different size, create a projection layer dynamically or handle appropriately
                        projected_goal_features = goal_features
                else:
                    projected_goal_features = goal_features
                    
                goal_distance_features = torch.cat([future_state, projected_goal_features], dim=-1)
                reachability = self.reachability_predictor(goal_distance_features)
            else:
                reachability = torch.zeros(batch_size, 1, device=state.device)
            
            counterfactuals.append({
                'future_state': future_state,
                'collision_prob': collision_prob,
                'info_gain': info_gain,
                'reachability': reachability,
                'action_idx': action_idx
            })
        
        return counterfactuals