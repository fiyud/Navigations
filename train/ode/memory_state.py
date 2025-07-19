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


class SpatialMemoryGraphODE(nn.Module):
    """Neural ODE dynamics for spatial memory graph evolution"""
    
    def __init__(self, feature_dim: int = 256, hidden_dim: int = 512, 
                 visual_dim: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Temporal dynamics network
        self.temporal_dynamics = nn.Sequential(
            nn.Linear(feature_dim + feature_dim, hidden_dim),  # node features + context
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Spatial diffusion network
        self.diffusion_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Ensure positive diffusion coefficients
        )
        
        # Feature encoder for observations
        self.feature_encoder = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
        # Edge predictor for graph connectivity
        self.edge_predictor = nn.Sequential(
            nn.Linear(feature_dim * 2 + 3, hidden_dim),  # two node features + position diff
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Laplacian computer
        self.laplacian = GraphLaplacian()
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(visual_dim, feature_dim),
            nn.LayerNorm(feature_dim)
        )
        
    def ode_func(self, t: torch.Tensor, state: torch.Tensor, 
                laplacian: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        ODE function defining the dynamics
        dh/dt = f(h, L, context)
        """
        batch_size, num_nodes, feature_dim = state.shape
        
        # Ensure context has correct batch dimension
        if context.dim() == 1:
            context = context.unsqueeze(0)
        
        # Temporal evolution
        context_expanded = context.unsqueeze(1).expand(-1, num_nodes, -1)
        temporal_input = torch.cat([state, context_expanded], dim=-1)
        dhdt_temporal = self.temporal_dynamics(temporal_input)
        
        # Spatial diffusion (PDE on graph)
        # D(h) * L * h
        diffusion_coeffs = self.diffusion_net(state)  # [batch, num_nodes, 1]
        
        # Apply Laplacian for each batch
        dhdt_spatial = torch.zeros_like(state)
        for b in range(batch_size):
            # L * h
            Lh = torch.matmul(laplacian, state[b])  # [num_nodes, feature_dim]
            # D * (L * h)
            dhdt_spatial[b] = diffusion_coeffs[b] * Lh
        
        # Combine temporal and spatial dynamics
        return dhdt_temporal - dhdt_spatial  # negative for diffusion
    
    def forward(self, graph_state: 'SpatialMemoryGraphState', 
                observation: torch.Tensor, position: torch.Tensor, 
                current_time: float) -> Tuple['SpatialMemoryGraphState', torch.Tensor]:
        """
        Update graph with new observation and return query features
        """
        obs_features = self.feature_encoder(observation)
        context = self.context_encoder(observation)
        
        # Evolve existing graph to current time
        if graph_state.num_nodes > 0:
            # Prepare state tensor
            state = graph_state.node_features.unsqueeze(0)  # [1, num_nodes, feature_dim]
            
            laplacian = self.laplacian(
                graph_state.edge_index, 
                graph_state.edge_attr,
                num_nodes=graph_state.num_nodes
            )
            
            if current_time > graph_state.last_update_time:
                import time
                start_time = time.time()

                time_points = torch.tensor([graph_state.last_update_time, current_time], 
                                        device=observation.device)
                
                context_batched = context.unsqueeze(0) if context.dim() == 1 else context
                
                evolved_state = odeint(
                    lambda t, h: self.ode_func(t, h, laplacian, context_batched),
                    state,
                    time_points,
                    method='dopri5',
                    rtol=1e-3,
                    atol=1e-4
                )[-1].squeeze(0)  # Take final time point, remove batch dim
                
                ode_time = time.time() - start_time

                if ode_time > 1.0:
                    print(f"ODE Solve took {ode_time:.2f}s")
                graph_state.node_features = evolved_state.detach().clone()
            else:
                evolved_state = graph_state.node_features
        
        # Find or create node for current position
        node_idx = graph_state.find_or_create_node(position, obs_features)
        
        # Information injection at current node
        alpha = 0.7  # injection rate
        if node_idx < graph_state.num_nodes:
            # Clone to avoid in-place modification
            new_features = graph_state.node_features.clone()
            new_features[node_idx] = (
                (1 - alpha) * graph_state.node_features[node_idx] + 
                alpha * obs_features
            )
            graph_state.node_features = new_features
        # Update edges based on evolved features
        graph_state.update_edges(self.edge_predictor)
        
        # Update time
        graph_state.last_update_time = current_time
        
        # Return query features (features at current node)
        if graph_state.num_nodes > 0:
            query_features = graph_state.node_features[node_idx]
        else:
            query_features = obs_features
        
        return graph_state, query_features


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
        """Add a new node to the graph"""
        if self.num_nodes >= self.max_nodes:
            # Remove oldest node
            self._remove_oldest_node()
        
        # Add new node
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
            visual_dim=visual_dim
        )
        
        # Query network for reading from graph
        self.query_net = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # Graph attention for aggregating information
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

        # Initialize with better weights
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
                time_delta: float = 1.0) -> Tuple[torch.Tensor, Dict[str, Any]]:
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
        batch_size = observation.size(0)
        device = observation.device
        
        # Initialize graph state if needed
        if self.graph_state is None:
            self.reset()
            self.graph_state.device = device
        
        self.current_time += time_delta
        
        # Process each batch element (usually batch_size=1 for RL)
        all_features = []
        all_info = []
        
        for b in range(batch_size):
            # Update graph with observation
            self.graph_state, node_features = self.graph_ode(
                self.graph_state,
                observation[b],
                position[b],
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
                # First node - return encoded observation
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
        
        # Stack features
        memory_features = torch.stack(all_features)
        
        # Aggregate info
        aggregated_info = {
            'num_nodes': all_info[0]['num_nodes'],
            'num_edges': all_info[0]['num_edges'],
            'attention_weights': all_info[0]['attention_weights'],
            'path_confidence': torch.stack([info['path_confidence'] for info in all_info]),
            'graph_state': self.graph_state
        }
        
        return memory_features, aggregated_info
    
    def get_graph_features(self) -> torch.Tensor:
        """Get current graph node features"""
        if self.graph_state is None or self.graph_state.num_nodes == 0:
            return torch.zeros(1, self.feature_dim, device=self.device)
        return self.graph_state.node_features
    
    def get_graph_data(self) -> Data:
        """Get graph as PyTorch Geometric Data object"""
        if self.graph_state is None:
            return Data()
        return self.graph_state.to_data()