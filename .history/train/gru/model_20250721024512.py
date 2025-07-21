import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from torch.distributions import Categorical

import sys
sys.path.append(r'/home/tuandang/tuandang/quanganh/visualnav-transformer/train')
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn

sys.path.append(r'/home/tuandang/tuandang/quanganh/visualnav-transformer/train/gru')
from memory_state import UnifiedSpatialMemoryGraphODE
from worldModel import CounterfactualWorldModel

class UnifiedAdvancedNoMaDRL(nn.Module):
    def __init__(
        self,
        action_dim: int = 4,
        encoding_size: int = 256,
        context_size: int = 5,
        mha_num_attention_heads: int = 4,
        mha_num_attention_layers: int = 4,
        mha_ff_dim_factor: int = 4,
        hidden_dim: int = 512,
        max_nodes: int = 500,
        distance_threshold: float = 1.0,
        use_counterfactuals: bool = True,
        config: Dict = None
    ):
        super().__init__()
        
        self.config = config or {}

        self.action_dim = action_dim
        self.encoding_size = encoding_size
        self.use_counterfactuals = use_counterfactuals
        
        self.vision_encoder = NoMaD_ViNT(
            obs_encoding_size=encoding_size,
            context_size=context_size,
            mha_num_attention_heads=mha_num_attention_heads,
            mha_num_attention_layers=mha_num_attention_layers,
            mha_ff_dim_factor=mha_ff_dim_factor,
        )
        
        self._replace_swish_with_relu()

        # Unified Spatial Memory Graph with Neural ODE/PDE
        self.spatial_memory_graph = UnifiedSpatialMemoryGraphODE(
            feature_dim=encoding_size,
            hidden_dim=hidden_dim,
            visual_dim=encoding_size,
            max_nodes=max_nodes,
            distance_threshold=distance_threshold
        )
        
        # Counterfactual world model (kept separate for modularity)
        if use_counterfactuals:
            self.world_model = CounterfactualWorldModel(
                feature_dim=encoding_size,
                action_dim=action_dim,
                hidden_dim=hidden_dim
            )
        
        # Feature fusion w/ unified memory/graph
        fusion_input_dim = encoding_size * 3  # vision + unified spatial memory
        # if use_counterfactuals:
        #     fusion_input_dim += encoding_size  # counterfactual features
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, encoding_size),
            nn.LayerNorm(encoding_size)
        )
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(encoding_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(encoding_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            # nn.Tanh()  # Output between -1 and 1
        )

        def init_value_weights(m):
            if isinstance(m, nn.Linear):
                if m.out_features == 1:  # Last layer
                    nn.init.constant_(m.weight, 0.0)
                    nn.init.constant_(m.bias, 0.0)
                else:
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

        self.value_net.apply(init_value_weights)

        # Distance prediction (from original NoMaD)
        self.dist_pred_net = nn.Sequential(
            nn.Linear(encoding_size, encoding_size // 4),
            nn.ReLU(),
            nn.Linear(encoding_size // 4, encoding_size // 16),
            nn.ReLU(),
            nn.Linear(encoding_size // 16, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Track time for ODE
        self.last_update_time = 0.0
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def _replace_swish_with_relu(self):
        """Replace Swish activation with ReLU to avoid traceback errors"""
        import torch.nn as nn
        
        def replace_activation(module):
            for name, child in module.named_children():
                # Check for various Swish implementations
                if (hasattr(child, '__class__') and 
                    ('swish' in child.__class__.__name__.lower() or 
                    child.__class__.__name__ == 'MemoryEfficientSwish')):
                    setattr(module, name, nn.ReLU(inplace=True))
                else:
                    replace_activation(child)
        
        # Replace in both encoders
        if hasattr(self.vision_encoder, 'obs_encoder'):
            replace_activation(self.vision_encoder.obs_encoder)
        if hasattr(self.vision_encoder, 'goal_encoder'):
            replace_activation(self.vision_encoder.goal_encoder)

    def forward(self, observations: Dict[str, torch.Tensor], mode: str = "policy",
            time_delta: float = 1.0, prev_action: torch.Tensor = None) -> Dict[str, Any]:
        """
        Unified forward pass with spatial memory graph and counterfactuals
        
        Args:
            observations: Dictionary containing:
                - context: [batch_size, 3*context_size, H, W]
                - goal_rgb: [batch_size, 3, H, W]
                - goal_mask: [batch_size, 1]
                - goal_position: [batch_size, 3]
            mode: One of "policy", "value", "distance", "features", or "all"
            time_delta: Time since last update (for ODE evolution)
        """
        batch_size = observations['rgb'].size(0)
    
    # Initialize goal_features early
        goal_features = None
        if 'goal_image' in observations:
            goal_features = observations['goal_image']
        
        # Extract visual features
        visual_features = self._extract_visual_features(observations)
        
        # Update spatial memory graph
        if hasattr(self, 'spatial_memory_graph') and self.spatial_memory_graph is not None:
            # Extract position (you may need to adjust this based on your observation format)
            if 'position' in observations:
                position = observations['position']
            else:
                # Default position if not available
                position = torch.zeros(batch_size, 3, device=visual_features.device)
            
            graph_state, graph_features = self.spatial_memory_graph(
                visual_features, position, time_delta
            )
            
            # Combine visual and graph features
            combined_features = torch.cat([visual_features, graph_features], dim=-1)
            combined_features = self.feature_fusion(combined_features)
        else:
            combined_features = visual_features
        
        # Use counterfactual world model if available and enabled
        if hasattr(self, 'world_model') and self.world_model is not None and self.use_counterfactuals:
            counterfactuals = self.world_model(
                state=combined_features,
                goal_features=goal_features  # Now properly initialized
            )
            
            # Aggregate counterfactual information
            counterfactual_features = self._aggregate_counterfactuals(counterfactuals)
            
            # Combine with existing features
            combined_features = torch.cat([combined_features, counterfactual_features], dim=-1)
            combined_features = self.counterfactual_fusion(combined_features)
        
        if mode == "features":
            return {
                'features': combined_features,
                'goal_features': goal_features
        }
        obs_img = observations['context']
        goal_img = observations['goal_rgb']
        goal_mask = observations['goal_mask']
        position = observations.get('goal_position', torch.zeros(obs_img.size(0), 3))
        
        # Vision encoding (original NoMaD)
        last_obs_frame = obs_img[:, -3:, :, :]
        obsgoal_img = torch.cat([last_obs_frame, goal_img], dim=1)
        
        vision_features = self.vision_encoder(
            obs_img=obs_img,
            goal_img=obsgoal_img,
            input_goal_mask=goal_mask.long().squeeze(-1)
        )
        
        print(f"Goal mask value: {goal_mask[0].item()}")

        if prev_action is None:
            prev_action = torch.zeros(obs_img.size(0), self.action_dim, device=obs_img.device)
        
        # Update spatial features call to include action
        spatial_features, spatial_info = self.spatial_memory_graph(
            observation=vision_features,
            position=position,
            action=prev_action,  # Add this
            goal_features=goal_features,
            time_delta=time_delta
        )
        


        # The condition might be inverted
        # if (goal_mask < 0.5).any():  # Goal mask = 0 means goal-conditioned
        #     goal_features = self.vision_encoder.goal_encoder(obsgoal_img)
        #     print(f"Goal conditioned episode - goal features shape: {goal_features.shape}")
        # else:
        #     goal_features = None
        #     print("Exploration episode - no goal features")
        
 
        goal_mask_value = goal_mask[0].item() if goal_mask.numel() > 0 else -1
        print(f"Goal mask value: {goal_mask_value}, shape: {goal_mask.shape}")

        batch_size = obs_img.size(0)
        goal_features_list = []
        
        for b in range(batch_size):
            if goal_mask[b].item() < 0.5:  # Goal-conditioned
                # Extract single sample goal image
                single_obsgoal = obsgoal_img[b:b+1]
                single_goal_features = self.vision_encoder.goal_encoder(single_obsgoal)
                goal_features_list.append(single_goal_features)
            else:  # Exploration
                # Use zeros or a learned exploration embedding
                goal_features_list.append(torch.zeros(1, 1000, device=obs_img.device))
        
        # Stack goal features
        goal_features = torch.cat(goal_features_list, dim=0) if goal_features_list else None
        
        # Only print during rollout (batch_size = 1)
        if batch_size == 1:
            if goal_mask[0].item() < 0.5:
                print(f"Goal conditioned episode - goal features shape: {goal_features.shape}")
            else:
                print("Exploration episode - no goal features")

        # Unified Spatial Memory Graph with Neural ODE evolution
        spatial_features, spatial_info = self.spatial_memory_graph(
            observation=vision_features,
            position=position,
            goal_features=goal_features,
            time_delta=time_delta
        )
        
        # Build features list for fusion
        features_to_fuse = [vision_features, spatial_features]

        # Counterfactual reasoning
        if self.use_counterfactuals and (mode == "policy" or mode == "all"):
            counterfactuals = self.world_model(vision_features, goal_features)
            
            # Aggregate counterfactual information
            cf_features = self._aggregate_counterfactuals(counterfactuals)
            features_to_fuse.append(cf_features)
        else:
            counterfactuals = None
            # Add zero features to maintain consistent input size
            cf_features = torch.zeros_like(vision_features)
            features_to_fuse.append(cf_features)

        # Ensure all features have the same number of dimensions
        normalized_features = []
        for feat in features_to_fuse:
            if feat.dim() == 3:
                feat = feat.squeeze(1)  # Remove middle dimension if present
            elif feat.dim() == 1:
                feat = feat.unsqueeze(0)  # Add batch dimension if missing
            normalized_features.append(feat)

        # Debug: Print feature dimensions
        # print(f"Feature dimensions: {[f.shape for f in normalized_features]}")

        # Fuse all features
        fused_features = torch.cat(normalized_features, dim=-1)
        final_features = self.feature_fusion(fused_features)
        
        # Generate outputs based on mode
        results = {
            'features': final_features,
            'spatial_info': spatial_info,
            'counterfactuals': counterfactuals,
            'graph_data': self.spatial_memory_graph.get_graph_data()
        }
        
        if mode == "policy" or mode == "all":
            policy_logits = self.policy_net(final_features)
            results['policy_logits'] = policy_logits
            results['action_dist'] = Categorical(logits=policy_logits)

        if mode == "value" or mode == "all":
            raw_values = self.value_net(final_features).squeeze(-1)
            # Scale values to reasonable range
            values = raw_values * 100.0  # Scale tanh output to [-100, 100]
            results['values'] = values
        
        if mode == "distance" or mode == "all":
            distances = self.dist_pred_net(final_features)
            results['distances'] = distances
        
        # Update time for next call
        self.last_update_time += time_delta
        
        return results
    
    def _aggregate_counterfactuals(self, counterfactuals: List[Dict]) -> torch.Tensor:
        """Aggregate counterfactual predictions into features"""
        # Stack counterfactual features
        future_states = torch.stack([cf['future_state'] for cf in counterfactuals], dim=1)
        collision_probs = torch.stack([cf['collision_prob'].squeeze(-1) for cf in counterfactuals], dim=1)
        info_gains = torch.stack([cf['info_gain'].squeeze(-1) for cf in counterfactuals], dim=1)
        reachabilities = torch.stack([cf['reachability'].squeeze(-1) for cf in counterfactuals], dim=1)
        
        # Weight future states by safety and information gain
        safety_weights = 1 - collision_probs
        exploration_weights = info_gains / (info_gains.sum(dim=1, keepdim=True) + 1e-8)
        goal_weights = reachabilities
        
        combined_weights = safety_weights + exploration_weights + goal_weights
        combined_weights = F.softmax(combined_weights, dim=1)
        
        # Weighted average of future states - ensure 2D output
        weighted_future = (future_states * combined_weights.unsqueeze(-1)).sum(dim=1)
        
        # Ensure output is 2D [batch_size, feature_dim]
        if weighted_future.dim() > 2:
            weighted_future = weighted_future.squeeze()
        
        return weighted_future
    
    def get_action(self, observations: Dict[str, torch.Tensor], 
                deterministic: bool = False,
                time_delta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        with torch.no_grad():
            outputs = self.forward(observations, mode="policy", time_delta=time_delta)
            action_dist = outputs['action_dist']
            
            if deterministic:
                action = action_dist.probs.argmax(dim=-1)
            else:
                probs = action_dist.probs
                if outputs.get('spatial_info', {}).get('path_confidence', 1.0).item() < 0.3:
                    # Boost rotation action probabilities
                    if probs.shape[-1] >= 4:  # Ensure we have rotation actions
                        probs[:, 2:4] *= 1.5  # Boost turn left/right
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                
                action = torch.multinomial(probs, 1).squeeze(-1)
            
            log_prob = action_dist.log_prob(action)
            
            # Return additional info for logging/debugging
            extra_info = {
                'num_nodes': outputs['spatial_info']['num_nodes'],
                'num_edges': outputs['spatial_info']['num_edges'],
                'path_confidence': outputs['spatial_info']['path_confidence'],
                'counterfactuals': outputs.get('counterfactuals', None)
            }
        
        return action, log_prob, extra_info
    
    def evaluate_actions(self, observations: Dict[str, torch.Tensor], 
                        actions: torch.Tensor,
                        time_delta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update"""
        outputs = self.forward(observations, mode="all", time_delta=time_delta)
        
        action_dist = outputs['action_dist']
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        values = outputs['values']
        
        return log_probs, values, entropy
    
    def reset_memory(self):
        """Reset spatial memory graph"""
        self.spatial_memory_graph.reset()
        self.last_update_time = 0.0
    
    def get_auxiliary_losses(self, batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses = {}
        
        # Get current outputs first
        current_outputs = self.forward(batch_data['observations'], mode="all")
        
        # Counterfactual consistency loss
        if self.use_counterfactuals and 'next_observations' in batch_data:
            with torch.no_grad():
                # Get actual next features
                next_outputs = self.forward(batch_data['next_observations'], mode="features")
                actual_next_features = next_outputs['features']
            
            counterfactuals = current_outputs['counterfactuals']
            
            if counterfactuals is not None:
                cf_loss = 0.0
                batch_size = batch_data['actions'].size(0)
                
                for b in range(batch_size):
                    action_idx = batch_data['actions'][b].item()
                    predicted_future = counterfactuals[action_idx]['future_state'][b]
                    actual_future = actual_next_features[b]
                    
                    cf_loss += F.mse_loss(predicted_future, actual_future)
                
                losses['counterfactual_loss'] = cf_loss / batch_size
        
        # Spatial coherence loss (optional - ensures smooth evolution)
        if 'spatial_info' in current_outputs:
            graph_data = self.spatial_memory_graph.get_graph_data()
            
            if graph_data.num_nodes > 1:
                # Compute feature smoothness over edges
                edge_index = graph_data.edge_index
                node_features = graph_data.x
                
                if edge_index.size(1) > 0:
                    src_features = node_features[edge_index[0]]
                    dst_features = node_features[edge_index[1]]
                    
                    # Features should be similar for connected nodes
                    smoothness_loss = F.mse_loss(src_features, dst_features)
                    losses['spatial_smoothness_loss'] = smoothness_loss
        
        if hasattr(self.spatial_memory_graph, 'path_score_regularization'):
            losses['path_scorer_reg_loss'] = -0.1 * self.spatial_memory_graph.path_score_regularization

        return losses

    def imagine_and_select_action(self, observations: Dict[str, torch.Tensor], 
                             horizon: int = 15) -> torch.Tensor:
        with torch.no_grad():
            # Get current state
            outputs = self.forward(observations, mode="features")
            current_features = outputs['features']
            
            # Imagine multiple action sequences
            num_samples = 10
            best_return = -float('inf')
            best_action = None
            
            for _ in range(num_samples):
                # Sample action sequence
                actions = []
                imagined_returns = 0
                state = current_features
                
                for t in range(horizon):
                    # Sample action
                    policy_output = self.policy_net(state)
                    action_dist = Categorical(logits=policy_output)
                    action = action_dist.sample()
                    actions.append(action)
                    
                    # Imagine next state (simplified - you'd use RSSM here)
                    # This is a placeholder - implement proper dynamics
                    next_state = state + 0.1 * torch.randn_like(state)
                    
                    # Predict reward
                    reward = self.dist_pred_net(state).squeeze()
                    imagined_returns += (0.99 ** t) * reward
                    
                    state = next_state
                
                # Keep best action
                if imagined_returns > best_return:
                    best_return = imagined_returns
                    best_action = actions[0]
            
            return best_action
        
class UnifiedAdvancedTrainingWrapper:
    def __init__(self, model: UnifiedAdvancedNoMaDRL, config: Dict):
        self.model = model
        self.config = config
        
        # Time tracking for ODE
        self.episode_start_time = 0.0
        self.current_time = 0.0
        self.time_step = config.get('time_step', 0.1)  # Time increment per environment step
    
    def compute_lambda_returns(self, rewards: torch.Tensor, values: torch.Tensor, 
                          dones: torch.Tensor, last_value: float,
                          gamma: float = 0.99, lambda_: float = 0.95) -> torch.Tensor:
        """Compute λ-returns as in DayDreamer Equation (3)"""
        device = rewards.device
        T = len(rewards)
        
        # Initialize returns
        returns = torch.zeros(T, device=device)
        
        # Compute λ-returns backwards
        for t in reversed(range(T)):
            if t == T - 1:
                # V_H^λ = v(s_H)
                returns[t] = last_value
            else:
                # V_t^λ = r_t + γ[(1-λ)v(s_{t+1}) + λV_{t+1}^λ]
                next_value = values[t + 1]
                next_return = returns[t + 1]
                
                returns[t] = rewards[t] + gamma * (1 - dones[t]) * (
                    (1 - lambda_) * next_value + lambda_ * next_return
                )
        
        return returns

    def reset_episode(self):
        """Reset for new episode"""
        # Don't reset the entire graph, just update time
        self.episode_start_time = self.current_time
        # Optionally reset graph only after many episodes
        if hasattr(self, 'episode_count'):
            self.episode_count += 1
            if self.episode_count % 100 == 0:  # Reset every 100 episodes
                print(f"Resetting spatial memory graph after {self.episode_count} episodes")
                self.model.reset_memory()
        else:
            self.episode_count = 1
        
    def get_time_delta(self) -> float:
        """Get time delta since last update"""
        return self.time_step
        
    def step(self, observations: Dict[str, torch.Tensor], 
             deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Execute one step with time tracking"""
        time_delta = self.get_time_delta()
        action, log_prob, extra_info = self.model.get_action(
            observations, 
            deterministic=deterministic,
            time_delta=time_delta
        )
        
        self.current_time += time_delta
        extra_info['current_time'] = self.current_time
        extra_info['episode_time'] = self.current_time - self.episode_start_time
        
        return action, log_prob, extra_info
    
    def compute_losses(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute all losses including auxiliary ones"""
        # Standard PPO losses
        log_probs, values, entropy = self.model.evaluate_actions(
            batch['observations'],
            batch['actions'],
            time_delta=self.time_step
        )
        
        # PPO losses
        advantages = batch['advantages']
        old_log_probs = batch['old_log_probs']
        returns = batch['returns']
        
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config['clip_ratio'], 
                           1 + self.config['clip_ratio']) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = F.mse_loss(values, returns)
        # value_loss = torch.clamp(value_loss, max=100.0)
        entropy_loss = -entropy.mean()
        
        # Auxiliary losses
        aux_losses = self.model.get_auxiliary_losses(batch)
        
        # Combine all losses
        total_loss = (
            policy_loss + 
            self.config['value_coef'] * value_loss +
            self.config['entropy_coef'] * entropy_loss
        )
        
        losses = {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy_loss': entropy_loss,
            'total_loss': total_loss
        }
        
        for name, loss in aux_losses.items():
            coef_name = name.replace('_loss', '_coef')
            coef = self.config.get(coef_name, 0.1)
            total_loss += coef * loss
            losses[name] = loss
        
        losses['total_loss'] = total_loss
        
        return losses

    def compute_lambda_returns(self, rewards, values, dones, gamma=0.99, lambda_=0.95):
        returns = torch.zeros_like(rewards)
        last_return = values[-1]
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                returns[t] = rewards[t] + gamma * (1 - dones[t]) * last_return
            else:
                td_error = rewards[t] + gamma * (1 - dones[t]) * values[t+1] - values[t]
                returns[t] = values[t] + td_error + gamma * lambda_ * (1 - dones[t]) * (returns[t+1] - values[t+1])
        
        return returns

def create_unified_model(config: Dict) -> UnifiedAdvancedNoMaDRL:
    return UnifiedAdvancedNoMaDRL(
        action_dim=config.get('action_dim', 4),
        encoding_size=config['encoding_size'],
        context_size=config['context_size'],
        mha_num_attention_heads=config['mha_num_attention_heads'],
        mha_num_attention_layers=config['mha_num_attention_layers'],
        mha_ff_dim_factor=config['mha_ff_dim_factor'],
        hidden_dim=config['hidden_dim'],
        max_nodes=config.get('max_nodes', 500),
        distance_threshold=config.get('distance_threshold', 1.0),
        use_counterfactuals=config.get('use_counterfactuals', True),
        config=config
    )

def analyze_counterfactuals(model: UnifiedAdvancedNoMaDRL, 
                       observations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    with torch.no_grad():
        outputs = model.forward(observations, mode="all")
        counterfactuals = outputs.get('counterfactuals', None)
        
        if counterfactuals is None:
            return {}
        
        analysis = {
            'action_safety_scores': [],
            'action_info_gains': [],
            'action_reachabilities': [],
            'best_action_safety': None,
            'best_action_exploration': None,
            'best_action_goal': None
        }
        
        for action_idx, cf in enumerate(counterfactuals):
            avg_safety = (1 - cf['collision_prob']).mean().item()
            avg_info_gain = cf['info_gain'].mean().item()
            avg_reachability = cf['reachability'].mean().item()
            
            analysis['action_safety_scores'].append(avg_safety)
            analysis['action_info_gains'].append(avg_info_gain)
            analysis['action_reachabilities'].append(avg_reachability)
        
        # Find best actions for each objective
        analysis['best_action_safety'] = np.argmax(analysis['action_safety_scores'])
        analysis['best_action_exploration'] = np.argmax(analysis['action_info_gains'])
        analysis['best_action_goal'] = np.argmax(analysis['action_reachabilities'])
        
        return analysis

def get_model_summary(model: UnifiedAdvancedNoMaDRL) -> Dict[str, Any]:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    component_params = {}
    for component_name in ['vision_encoder', 'spatial_memory_graph', 'world_model', 
                        'feature_fusion', 'policy_net', 'value_net']:
        if hasattr(model, component_name):
            component = getattr(model, component_name)
            if component is not None:
                params = sum(p.numel() for p in component.parameters())
                component_params[component_name] = params
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'component_parameters': component_params,
        'model_config': {
            'action_dim': model.action_dim,
            'encoding_size': model.encoding_size,
            'use_counterfactuals': model.use_counterfactuals,
            'max_nodes': model.spatial_memory_graph.max_nodes,
            'distance_threshold': model.spatial_memory_graph.distance_threshold
        }
    }

def visualize_spatial_memory_graph(model: UnifiedAdvancedNoMaDRL, save_path: str = None):
    """Visualize the current state of the spatial memory graph"""
    import matplotlib.pyplot as plt
    import networkx as nx
    
    graph_data = model.spatial_memory_graph.get_graph_data()
    
    if graph_data.num_nodes == 0:
        print("Graph is empty")
        return
    
    # Convert to NetworkX for visualization
    G = nx.Graph()
    
    # Add nodes
    positions = {}
    for i in range(graph_data.num_nodes):
        G.add_node(i)
        positions[i] = (graph_data.pos[i, 0].item(), graph_data.pos[i, 2].item())  # x, z
    
    # Add edges
    edge_index = graph_data.edge_index
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        if src < dst:  # Avoid duplicate edges
            G.add_edge(src, dst)
    
    plt.figure(figsize=(10, 10))
    nx.draw(G, positions, 
            node_color='lightblue',
            node_size=500,
            with_labels=True,
            font_size=8,
            edge_color='gray',
            width=2)
    
    plt.title("Spatial Memory Graph")
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    
    plt.close()