import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import math

from torchdiffeq import odeint, odeint_adjoint

# For Graph operations
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, degree, to_dense_adj

class RSSM(nn.Module):
    """Recurrent State-Space Model for better dynamics prediction"""
    def __init__(self, stoch_size=32, deter_size=512, hidden_size=512):
        super().__init__()
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        
        # Prior: p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2 * stoch_size)  # mean and log_std
        )
        
        # Posterior: q(z_t | h_t, x_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_size + hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2 * stoch_size)
        )
        
        # Deterministic state update
        self.cell = nn.GRUCell(stoch_size + hidden_size, deter_size)

class GraphLaplacian(nn.Module):
    """Learnable graph Laplacian for diffusion"""
    def __init__(self, edge_attr_dim: int = 1):
        super().__init__()
        self.edge_weight_net = nn.Sequential(
            nn.Linear(edge_attr_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, 
                num_nodes: Optional[int] = None) -> torch.Tensor:
        """Compute normalized graph Laplacian"""
        if edge_attr is not None:
            edge_weight = self.edge_weight_net(edge_attr).squeeze(-1)
        else:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        
        # Add self-loops
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=num_nodes)
        
        # Compute degree
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=edge_weight.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Normalize
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        
        # Convert to dense Laplacian matrix
        L = torch.eye(num_nodes, device=edge_index.device)
        adj = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes)[0]
        L = L - adj
        
        return L

class RSSM(nn.Module):
    """Recurrent State-Space Model for better dynamics prediction"""
    def __init__(self, stoch_size=32, deter_size=512, hidden_size=512):
        super().__init__()
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        
        # Prior: p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2 * stoch_size)  # mean and log_std
        )
        
        # Posterior: q(z_t | h_t, x_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_size + hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 2 * stoch_size)
        )
        
        # Deterministic state update
        self.cell = nn.GRUCell(stoch_size + hidden_size, deter_size)

class RSSMState:
    """Container for RSSM hidden states"""
    def __init__(self, batch_size, stoch_size=32, deter_size=512, device='cuda'):
        self.stoch = torch.zeros(batch_size, stoch_size, device=device)
        self.deter = torch.zeros(batch_size, deter_size, device=device)
        self.device = device
    
    def detach(self):
        self.stoch = self.stoch.detach()
        self.deter = self.deter.detach()
        return self

class RSSMCore(nn.Module):
    def __init__(self, feature_dim=256, action_dim=4, stoch_size=32, deter_size=512, hidden_size=512, num_classes=32):
        super().__init__()
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.num_classes = num_classes
        self.action_dim = action_dim

        stoch_flat_size = stoch_size * num_classes
        gru_input_size = stoch_flat_size + action_dim  
        
        # Deterministic state model (GRU)
        self.cell = nn.GRUCell(gru_input_size, deter_size)
        
        # Stochastic state model
        self.prior_net = nn.Sequential(
            nn.Linear(deter_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, stoch_size * num_classes)
        )
        
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_size + feature_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, stoch_size * num_classes)
        )
    
    def forward(self, prev_state, prev_action, embed=None):
        """
        Args:
            prev_state: RSSMState
            prev_action: [batch, action_dim]
            embed: [batch, feature_dim] (optional, for posterior)
        Returns:
            post: RSSMState (posterior if embed provided, else prior)
            prior: RSSMState (always prior)
        """
        # Update deterministic state
        if prev_action.dim() == 1:
            prev_action = prev_action.unsqueeze(0)

        print(f"Debug RSSM forward:")
        print(f"  prev_state.stoch shape: {prev_state.stoch.shape}")
        print(f"  prev_action shape: {prev_action.shape}")
        print(f"  prev_state.deter shape: {prev_state.deter.shape}")
        prev_stoch_flat = prev_state.stoch.reshape(prev_state.stoch.shape[0], -1)
        print(f"  prev_stoch_flat shape: {prev_stoch_flat.shape}")
        print(f"  Expected GRU input size: {prev_stoch_flat.shape[1] + prev_action.shape[1]}")

        if prev_state.deter.dim() == 1:
            prev_state.deter = prev_state.deter.unsqueeze(0)

        deter = self.cell(torch.cat([prev_stoch_flat, prev_action], -1), prev_state.deter)
        
        prior_logits = self.prior_net(deter)
        prior_logits = prior_logits.reshape(-1, self.stoch_size, self.num_classes)
        prior_stoch = F.softmax(prior_logits, -1)
        prior_sample = self._sample_discrete(prior_logits)
        
        # Create prior state
        prior = RSSMState(prev_state.stoch.shape[0], self.stoch_size, self.deter_size, prev_state.device)
        prior.stoch = prior_sample
        prior.deter = deter
        
        # Compute posterior if embedding provided
        if embed is not None:
            post_logits = self.posterior_net(torch.cat([deter, embed], -1))
            post_logits = post_logits.reshape(-1, self.stoch_size, self.num_classes)
            post_stoch = F.softmax(post_logits, -1)
            post_sample = self._sample_discrete(post_logits)
            
            post = RSSMState(prev_state.stoch.shape[0], self.stoch_size, self.deter_size, prev_state.device)
            post.stoch = post_sample
            post.deter = deter
            
            return post, prior
        
        return prior, prior
    
    def _sample_discrete(self, logits):
        """Sample discrete latents with straight-through gradients"""
        dist = torch.distributions.OneHotCategorical(logits=logits)
        if self.training:
            sample = dist.sample() + dist.probs - dist.probs.detach()
        else:
            sample = dist.probs
        return sample.reshape(logits.shape[0], -1)
    
class SpatialMemoryGraphODE(nn.Module): # da doi qua rssm
    def __init__(self, feature_dim: int = 256, hidden_dim: int = 512, 
                 visual_dim: int = 256, action_dim: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # RSSM Core from DayDreamer
        self.rssm = RSSMCore(
            feature_dim=visual_dim,
            action_dim=action_dim,
            stoch_size=32,
            deter_size=hidden_dim,
            hidden_size=hidden_dim,
            num_classes=32
        )
        
        # Encoder for observations
        self.encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        stoch_flat_size = self.rssm.stoch_size * self.rssm.num_classes  # 1024
        decoder_input_size = stoch_flat_size + self.rssm.deter_size # 1024 + 512 = 1536

        # Decoder (for reconstruction loss)
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, visual_dim)
        )
        
        # Reward predictor
        self.reward_net = nn.Sequential(
            nn.Linear(feature_dim + self.rssm.deter_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Edge predictor for graph connectivity
        self.edge_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Keep existing Laplacian for spatial smoothness
        self.laplacian = GraphLaplacian()
    
    def forward(self, graph_state: 'SpatialMemoryGraphState', 
                observation: torch.Tensor, position: torch.Tensor,
                action: torch.Tensor, rssm_state: RSSMState,
                current_time: float) -> Tuple['SpatialMemoryGraphState', torch.Tensor, RSSMState]:
        
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
            
        obs_embed = self.encoder(observation)
        post, prior = self.rssm(rssm_state, action, obs_embed)
        
        combined_features = torch.cat([
                post.stoch,  # [batch, stoch_size * num_classes]
                post.deter   # Shape [batch, deter_size]
            ], -1)
        
        print(f"Debug decoder input:")
        print(f"  combined_features shape: {combined_features.shape}")
        print(f"  Expected decoder input: {self.decoder[0].in_features}")

        node_features = self.decoder(combined_features)

        print(f"Debug node_features shape: {node_features.shape}")
        print(f"Expected feature_dim: {graph_state.feature_dim if graph_state else 'None'}")
        
        if node_features.dim() == 2 and node_features.size(0) == 1:
            node_features = node_features.squeeze(0)

        # Update graph state
        node_idx = graph_state.find_or_create_node(position, node_features)
        
        # Information injection at current node
        alpha = 0.7
        if node_idx < graph_state.num_nodes:
            new_features = graph_state.node_features.clone()
            new_features[node_idx] = (
                (1 - alpha) * graph_state.node_features[node_idx] + 
                alpha * node_features
            )
            graph_state.node_features = new_features
        
        graph_state.update_edges(self.edge_predictor)
        graph_state.last_update_time = current_time
        
        # Return features at current node
        if graph_state.num_nodes > 0:
            query_features = graph_state.node_features[node_idx]
        else:
            query_features = node_features
        
        return graph_state, query_features, post
    
class SpatialMemoryGraphState:
    def __init__(self, feature_dim: int = 256, max_nodes: int = 500, 
                distance_threshold: float = 1.0, device: torch.device = None):
        self.feature_dim = feature_dim
        self.max_nodes = max_nodes
        self.distance_threshold = distance_threshold
        self.device = device or torch.device('cpu')
        
        # Node attributes
        self.node_features = torch.empty(0, feature_dim, device=self.device)
        self.node_positions = torch.empty(0, 3, device=self.device)
        self.node_timestamps = torch.empty(0, device=self.device)
        
        # Edge attributes
        self.edge_index = torch.empty(2, 0, dtype=torch.long, device=self.device)
        self.edge_attr = torch.empty(0, 1, device=self.device)
        
        # State
        self.num_nodes = 0
        self.last_update_time = 0.0
    
    def find_or_create_node(self, position: torch.Tensor, 
                       features: torch.Tensor) -> int:
        """Find existing node or create new one"""
        if self.num_nodes == 0:
            print(f"Creating first node at position: {position}")
            return self._add_node(position, features)
        
        # Compute distances to existing nodes
        distances = torch.norm(self.node_positions - position.unsqueeze(0), dim=1)
        min_dist, nearest_idx = distances.min(dim=0)
        
        print(f"Min distance to existing nodes: {min_dist.item():.3f}, threshold: {self.distance_threshold}")
        
        if min_dist < self.distance_threshold:
            # print(f"Reusing node {nearest_idx.item()}")
            return nearest_idx.item()
        else:
            print(f"Creating new node, total nodes will be: {self.num_nodes + 1}")
            return self._add_node(position, features)
    
    def _add_node(self, position: torch.Tensor, features: torch.Tensor) -> int:
        if self.num_nodes >= self.max_nodes:
            self._remove_oldest_node()
        
        if features.dim() == 2:
            features = features.squeeze(0)
        
        if position.dim() == 2:
            position = position.squeeze(0)
        
        # Debug print
        print(f"Adding node - features shape: {features.shape}, position shape: {position.shape}")
        print(f"Existing node_features shape: {self.node_features.shape if self.num_nodes > 0 else 'Empty'}")
    
        self.node_features = torch.cat([
            self.node_features, 
            features.unsqueeze(0)
        ], dim=0)
        
        self.node_positions = torch.cat([
            self.node_positions,
            position.unsqueeze(0)
        ], dim=0)
        
        self.node_timestamps = torch.cat([
            self.node_timestamps,
            torch.tensor([self.last_update_time], device=self.device)
        ])
        
        new_idx = self.num_nodes
        self.num_nodes += 1
        
        return new_idx
    
    def _remove_oldest_node(self):
        """Remove the oldest node from the graph"""
        if self.num_nodes == 0:
            return
        
        # Find oldest node
        oldest_idx = self.node_timestamps.argmin()
        
        # Remove node
        keep_mask = torch.ones(self.num_nodes, dtype=torch.bool, device=self.device)
        keep_mask[oldest_idx] = False
        
        self.node_features = self.node_features[keep_mask]
        self.node_positions = self.node_positions[keep_mask]
        self.node_timestamps = self.node_timestamps[keep_mask]
        
        # Update edges (remove edges connected to removed node)
        edge_mask = (self.edge_index[0] != oldest_idx) & (self.edge_index[1] != oldest_idx)
        self.edge_index = self.edge_index[:, edge_mask]
        self.edge_attr = self.edge_attr[edge_mask]
        
        # Adjust indices
        self.edge_index[self.edge_index > oldest_idx] -= 1
        
        self.num_nodes -= 1
    
    def update_edges(self, edge_predictor: nn.Module, 
                    max_edge_distance: float = 5.0):
        """Update graph edges based on current node features"""
        if self.num_nodes < 2:
            return
        
        new_edges = []
        new_attrs = []
        
        # Check all pairs of nodes
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                pos_diff = self.node_positions[j] - self.node_positions[i]
                distance = torch.norm(pos_diff)
                
                if distance < max_edge_distance:
                    # Predict edge probability
                    edge_features = torch.cat([
                        self.node_features[i],
                        self.node_features[j],
                        pos_diff
                    ])
                    
                    edge_prob = edge_predictor(edge_features)
                    
                    if edge_prob > 0.5:
                        # Add bidirectional edge
                        new_edges.append([i, j])
                        new_edges.append([j, i])
                        new_attrs.append(edge_prob)
                        new_attrs.append(edge_prob)
        
        if new_edges:
            self.edge_index = torch.tensor(new_edges, device=self.device).t()
            self.edge_attr = torch.stack(new_attrs)
        else:
            self.edge_index = torch.empty(2, 0, dtype=torch.long, device=self.device)
            self.edge_attr = torch.empty(0, 1, device=self.device)
    
    def to_data(self) -> Data:
        if self.num_nodes == 0:
            return Data(
                x=torch.empty(0, self.feature_dim, device=self.device),
                edge_index=torch.empty(2, 0, dtype=torch.long, device=self.device),
                edge_attr=torch.empty(0, 1, device=self.device),
                pos=torch.empty(0, 3, device=self.device)
            )
        
        return Data(
            x=self.node_features,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            pos=self.node_positions
        )
    
    def reset(self):
        self.node_features = torch.empty(0, self.feature_dim, device=self.device)
        self.node_positions = torch.empty(0, 3, device=self.device)
        self.node_timestamps = torch.empty(0, device=self.device)
        self.edge_index = torch.empty(2, 0, dtype=torch.long, device=self.device)
        self.edge_attr = torch.empty(0, 1, device=self.device)
        self.num_nodes = 0
        self.last_update_time = 0.0


class UnifiedSpatialMemoryGraphODE(nn.Module):
    def __init__(self, feature_dim: int = 256, hidden_dim: int = 512,
                 visual_dim: int = 256, max_nodes: int = 500,
                 distance_threshold: float = 1.0):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_nodes = max_nodes
        self.distance_threshold = distance_threshold
        
        # Spatial Memory Graph ODE
        self.graph_ode = SpatialMemoryGraphODE(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            visual_dim=visual_dim,
            action_dim=6
        )
        
        self.rssm_state = None

        # Query network for reading from graph
        self.query_net = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Path planner on graph
        self.path_scorer = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),  
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        for layer in self.path_scorer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

        self.path_scorer.apply(init_weights)

        # with torch.no_grad():
        #     self.path_scorer[-1].bias.fill_(0.1)  # Small positive bias
        
        self.goal_projection = nn.Linear(1000, feature_dim)

        self.graph_state = None
        self.current_time = 0.0
    
    def reset(self):
        """Reset the graph state"""
        device = next(self.parameters()).device
        self.graph_state = SpatialMemoryGraphState(
            feature_dim=self.feature_dim,
            max_nodes=self.max_nodes,
            distance_threshold=self.distance_threshold,
            device=device  # Pass the correct device
        )
        self.current_time = 0.0
    
    def forward(self, observation: torch.Tensor, position: torch.Tensor,
                goal_features: Optional[torch.Tensor] = None,
                time_delta: float = 1.0, action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process observation and return memory features
        
        Args:
            observation: Current visual features [batch_size, visual_dim]
            position: Current 3D position [batch_size, 3]
            goal_features: Optional goal features [batch_size, feature_dim]
            time_delta: Time since last update
            
        Returns:
            memory_features: Aggregated features from spatial memory graph
            info: Dictionary with additional information
        """
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
            single_sample = True
        else:
            single_sample = observation.size(0) == 1
    
        batch_size = observation.size(0)
        device = observation.device

        if self.rssm_state is None:
            self.rssm_state = RSSMState(
                batch_size, 
                stoch_size=self.graph_ode.rssm.stoch_size,
                deter_size=self.graph_ode.rssm.deter_size,
                device=device
            )

            self.rssm_state.stoch = torch.zeros(
                batch_size, 
                self.graph_ode.rssm.stoch_size * self.graph_ode.rssm.num_classes,
                device=device
            )
        
        if action is None:
            action = torch.zeros(batch_size, self.graph_ode.rssm.action_dim, device=device)
            action[:, 0] = 1.0
        elif action.dim() == 1:
            action = action.unsqueeze(0)

        if position.dim() == 1:
            position = position.unsqueeze(0)

        if self.graph_state is None:
            self.reset()
            self.graph_state.device = device
        
        self.current_time += time_delta
        
        # Process each batch element (usually batch_size=1 for RL)
        all_features = []
        all_info = []
        
        if single_sample:
            obs_single = observation.squeeze(0)
            pos_single = position.squeeze(0)
            action_single = action.squeeze(0)
            
            self.graph_state, node_features, self.rssm_state = self.graph_ode(
                self.graph_state,
                obs_single,
                pos_single,
                action_single,
                self.rssm_state,
                self.current_time
            )
            
            if node_features.dim() == 1:
                node_features = node_features.unsqueeze(0)

        else:
            for b in range(batch_size):
                self.graph_state, node_features, self.rssm_state = self.graph_ode(
                    self.graph_state,
                    observation[b],
                    position[b],
                    action[b],
                    self.rssm_state,
                    self.current_time
                )
                
                # Query graph for relevant information
                if self.graph_state.num_nodes > 0:
                    # Create query from current observation
                    query = self.query_net(observation[b]).unsqueeze(0).unsqueeze(0)
                    
                    # Get all node features
                    keys = values = self.graph_state.node_features.unsqueeze(0)
                    
                    # Attention-based aggregation
                    aggregated_features, attention_weights = self.graph_attention(
                        query, keys, values
                    )
                    aggregated_features = aggregated_features.squeeze(0).squeeze(0)
                    
                    # Plan path if goal is provided
                    path_confidence = torch.tensor(0.0, device=device)
                    if goal_features is not None and self.graph_state.num_nodes > 1:
                        # Find current node
                        distances = torch.norm(
                            self.graph_state.node_positions - position[b].unsqueeze(0), 
                            dim=1
                        )
                        current_idx = distances.argmin()
                        
                        # Score all nodes for goal similarity
                        goal_scores = []
                        projected_goal = self.goal_projection(goal_features[b])
                        
                        print(f"Goal features shape: {goal_features[b].shape}")
                        print(f"Projected goal shape: {projected_goal.shape}")
                        print(f"Number of nodes to score: {self.graph_state.num_nodes}")
                        
                        for node_feat in self.graph_state.node_features:
                            score_input = torch.cat([node_feat, projected_goal])
                            score = self.path_scorer(score_input)
                            goal_scores.append(score)
                        
                        if goal_scores:
                            goal_scores = torch.stack(goal_scores).squeeze(-1)  # Stack properly
                            path_confidence = goal_scores.max()
                            print(f"Max path confidence: {path_confidence.item():.4f}")

                            self.path_score_regularization = torch.mean(torch.abs(goal_scores))

                    info = {
                        'num_nodes': self.graph_state.num_nodes,
                        'num_edges': self.graph_state.edge_index.size(1) // 2,
                        'attention_weights': attention_weights.squeeze(),
                        'path_confidence': path_confidence,
                        'current_node_features': node_features
                    }
                else:
                    aggregated_features = node_features
                    info = {
                        'num_nodes': 1,
                        'num_edges': 0,
                        'attention_weights': torch.ones(1, device=device),
                        'path_confidence': torch.tensor(0.0, device=device),
                        'current_node_features': node_features
                    }
                
                all_features.append(aggregated_features)
                all_info.append(info)
            
        memory_features = torch.stack(all_features)
        
        aggregated_info = {
            'num_nodes': all_info[0]['num_nodes'],
            'num_edges': all_info[0]['num_edges'],
            'attention_weights': all_info[0]['attention_weights'],
            'path_confidence': torch.stack([info['path_confidence'] for info in all_info]),
            'graph_state': self.graph_state
        }
        
        return memory_features, aggregated_info
    
    def get_graph_features(self) -> torch.Tensor:
        if self.graph_state is None or self.graph_state.num_nodes == 0:
            return torch.zeros(1, self.feature_dim, device=self.device)
        return self.graph_state.node_features
    
    def get_graph_data(self) -> Data:
        if self.graph_state is None:
            return Data()
        return self.graph_state.to_data()