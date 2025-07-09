"""
Graph Neural Network components for modeling spatial relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.utils import grid

class GraphNeuralNetwork(nn.Module):
    """
    Graph Neural Network for modeling spatial relationships in meteorological data.
    Converts gridded data to graph representation and processes spatial interactions.
    """
    
    def __init__(self,
                 input_channels: int = 256,
                 hidden_channels: int = 256,
                 output_channels: int = 256,
                 num_layers: int = 3,
                 gnn_type: str = "gat",  # "gcn", "gat", "custom"
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 graph_connectivity: str = "grid_8",  # "grid_4", "grid_8", "adaptive"
                 use_edge_features: bool = True):
        """
        Initialize Graph Neural Network.
        
        Args:
            input_channels: Number of input feature channels
            hidden_channels: Number of hidden channels
            output_channels: Number of output channels
            num_layers: Number of GNN layers
            gnn_type: Type of GNN layer ("gcn", "gat", "custom")
            num_heads: Number of attention heads (for GAT)
            dropout: Dropout probability
            graph_connectivity: Graph connectivity pattern
            use_edge_features: Whether to use edge features
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.graph_connectivity = graph_connectivity
        self.use_edge_features = use_edge_features
        
        # Input projection
        self.input_projection = nn.Linear(input_channels, hidden_channels)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(num_layers):
            if gnn_type == "gcn":
                gnn_layer = GCNConv(hidden_channels, hidden_channels)
            elif gnn_type == "gat":
                gnn_layer = GATConv(
                    hidden_channels, hidden_channels // num_heads, 
                    heads=num_heads, dropout=dropout, concat=True
                )
            else:  # custom
                gnn_layer = LightningGNNLayer(
                    hidden_channels, hidden_channels, num_heads, dropout
                )
            
            self.gnn_layers.append(gnn_layer)
            self.norms.append(nn.LayerNorm(hidden_channels))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Output projection
        self.output_projection = nn.Linear(hidden_channels, output_channels)
        
        # Edge feature computation
        if use_edge_features:
            self.edge_feature_net = EdgeFeatureNetwork(hidden_channels)
        
        # Grid-to-graph converter
        self.grid_converter = GridToGraphConverter(graph_connectivity)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        Args:
            x: Input features (batch_size, channels, height, width)
            
        Returns:
            Output features (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.shape
        
        # Convert grid to graph representation
        node_features, edge_index, edge_attr = self.grid_converter(x)
        
        # Project to hidden dimension
        node_features = self.input_projection(node_features)
        
        # Apply GNN layers
        for i, (gnn_layer, norm, dropout) in enumerate(zip(self.gnn_layers, self.norms, self.dropouts)):
            
            # Store residual
            residual = node_features
            
            # Apply GNN layer
            if self.gnn_type in ["gcn", "gat"]:
                if self.use_edge_features and edge_attr is not None:
                    # For layers that support edge features
                    if hasattr(gnn_layer, 'edge_dim'):
                        node_features = gnn_layer(node_features, edge_index, edge_attr)
                    else:
                        node_features = gnn_layer(node_features, edge_index)
                else:
                    node_features = gnn_layer(node_features, edge_index)
            else:  # custom
                node_features = gnn_layer(node_features, edge_index, edge_attr)
            
            # Apply normalization and dropout
            node_features = norm(node_features)
            node_features = F.relu(node_features)
            node_features = dropout(node_features)
            
            # Residual connection
            node_features = node_features + residual
        
        # Project to output dimension
        node_features = self.output_projection(node_features)
        
        # Convert back to grid representation
        output = self.grid_converter.graph_to_grid(
            node_features, batch_size, height, width
        )
        
        return output

class LightningGNNLayer(MessagePassing):
    """
    Custom GNN layer designed for lightning prediction.
    Incorporates meteorological domain knowledge.
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 edge_dim: Optional[int] = None):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.dropout = dropout
        
        # Linear transformations
        self.lin_q = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_k = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_v = nn.Linear(in_channels, out_channels, bias=False)
        
        # Edge feature processing
        if edge_dim is not None:
            self.edge_lin = nn.Linear(edge_dim, out_channels)
        else:
            self.edge_lin = None
        
        # Output projection
        self.lin_out = nn.Linear(out_channels, out_channels)
        
        # Attention scaling
        self.scale = self.head_dim ** -0.5
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.lin_q.weight)
        nn.init.xavier_uniform_(self.lin_k.weight)
        nn.init.xavier_uniform_(self.lin_v.weight)
        nn.init.xavier_uniform_(self.lin_out.weight)
        if self.edge_lin is not None:
            nn.init.xavier_uniform_(self.edge_lin.weight)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None):
        """Forward pass through custom GNN layer."""
        
        # Generate queries, keys, values
        q = self.lin_q(x).view(-1, self.num_heads, self.head_dim)
        k = self.lin_k(x).view(-1, self.num_heads, self.head_dim)
        v = self.lin_v(x).view(-1, self.num_heads, self.head_dim)
        
        # Propagate messages
        out = self.propagate(edge_index, q=q, k=k, v=v, edge_attr=edge_attr)
        
        # Reshape and project output
        out = out.view(-1, self.out_channels)
        out = self.lin_out(out)
        
        return out
    
    def message(self, q_i, k_j, v_j, edge_attr, index, ptr, size_i):
        """Compute messages between nodes."""
        
        # Compute attention scores
        attn = (q_i * k_j).sum(dim=-1) * self.scale  # (num_edges, num_heads)
        
        # Incorporate edge features if available
        if edge_attr is not None and self.edge_lin is not None:
            edge_features = self.edge_lin(edge_attr)
            edge_features = edge_features.view(-1, self.num_heads, self.head_dim)
            attn = attn + (q_i * edge_features).sum(dim=-1) * self.scale
        
        # Apply softmax attention
        attn = F.softmax(attn, dim=0)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        # Apply attention to values
        out = v_j * attn.unsqueeze(-1)
        
        return out

class GridToGraphConverter(nn.Module):
    """
    Converts between grid representation and graph representation.
    """
    
    def __init__(self, connectivity: str = "grid_8"):
        super().__init__()
        self.connectivity = connectivity
    
    def forward(self, grid_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Convert grid to graph representation.
        
        Args:
            grid_features: Grid features (batch_size, channels, height, width)
            
        Returns:
            Tuple of (node_features, edge_index, edge_attr)
        """
        batch_size, channels, height, width = grid_features.shape
        
        # Flatten spatial dimensions to create nodes
        node_features = grid_features.permute(0, 2, 3, 1).reshape(batch_size * height * width, channels)
        
        # Create edge index based on connectivity
        edge_index = self._create_edge_index(height, width, grid_features.device)
        
        # Create edge attributes (spatial distances, gradients, etc.)
        edge_attr = self._create_edge_attributes(grid_features, edge_index)
        
        return node_features, edge_index, edge_attr
    
    def _create_edge_index(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Create edge index for grid connectivity."""
        
        if self.connectivity == "grid_4":
            # 4-connectivity (up, down, left, right)
            return grid(height, width, dtype=torch.long, device=device)
        elif self.connectivity == "grid_8":
            # 8-connectivity (includes diagonals)
            edges = []
            
            for h in range(height):
                for w in range(width):
                    node_idx = h * width + w
                    
                    # All 8 neighbors
                    for dh in [-1, 0, 1]:
                        for dw in [-1, 0, 1]:
                            if dh == 0 and dw == 0:
                                continue
                            
                            nh, nw = h + dh, w + dw
                            if 0 <= nh < height and 0 <= nw < width:
                                neighbor_idx = nh * width + nw
                                edges.append([node_idx, neighbor_idx])
            
            edge_index = torch.tensor(edges, dtype=torch.long, device=device).t()
            return edge_index
        else:
            raise ValueError(f"Unknown connectivity: {self.connectivity}")
    
    def _create_edge_attributes(self, grid_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Create edge attributes based on spatial relationships."""
        
        batch_size, channels, height, width = grid_features.shape
        num_edges = edge_index.shape[1]
        
        # Spatial distance features
        src_indices = edge_index[0]
        dst_indices = edge_index[1]
        
        # Convert linear indices to 2D coordinates
        src_h = src_indices // width
        src_w = src_indices % width
        dst_h = dst_indices // width  
        dst_w = dst_indices % width
        
        # Compute spatial distances
        spatial_dist = torch.sqrt((src_h - dst_h).float() ** 2 + (src_w - dst_w).float() ** 2)
        
        # Compute feature differences (gradients)
        grid_flat = grid_features.permute(0, 2, 3, 1).reshape(-1, channels)
        
        feature_diffs = []
        for b in range(batch_size):
            offset = b * height * width
            src_feats = grid_flat[offset + src_indices]
            dst_feats = grid_flat[offset + dst_indices]
            diff = torch.abs(src_feats - dst_feats).mean(dim=1)
            feature_diffs.append(diff)
        
        feature_diff = torch.stack(feature_diffs, dim=0).mean(dim=0)
        
        # Combine edge attributes
        edge_attr = torch.stack([spatial_dist, feature_diff], dim=1)
        
        return edge_attr
    
    def graph_to_grid(self, node_features: torch.Tensor, batch_size: int, height: int, width: int) -> torch.Tensor:
        """Convert graph representation back to grid."""
        
        channels = node_features.shape[1]
        
        # Reshape back to grid
        grid_features = node_features.view(batch_size, height, width, channels)
        grid_features = grid_features.permute(0, 3, 1, 2)
        
        return grid_features

class EdgeFeatureNetwork(nn.Module):
    """Network for computing rich edge features."""
    
    def __init__(self, node_dim: int):
        super().__init__()
        
        self.node_dim = node_dim
        
        # Edge feature computation
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 2, node_dim),  # 2 nodes + 2 spatial features
            nn.ReLU(),
            nn.Linear(node_dim, node_dim // 2),
            nn.ReLU(),
            nn.Linear(node_dim // 2, node_dim // 4)
        )
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, spatial_features: torch.Tensor) -> torch.Tensor:
        """Compute edge features from node features and spatial relationships."""
        
        src_features = node_features[edge_index[0]]
        dst_features = node_features[edge_index[1]]
        
        # Concatenate source, destination, and spatial features
        edge_input = torch.cat([src_features, dst_features, spatial_features], dim=1)
        
        # Compute edge features
        edge_features = self.edge_mlp(edge_input)
        
        return edge_features

class LightningGNN(GraphNeuralNetwork):
    """
    Specialized GNN for lightning prediction with meteorological domain knowledge.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Lightning-specific components
        self.wind_shear_processor = WindShearProcessor(self.hidden_channels)
        self.thermal_gradient_processor = ThermalGradientProcessor(self.hidden_channels)
        self.moisture_convergence_processor = MoistureConvergenceProcessor(self.hidden_channels)
    
    def forward(self, x: torch.Tensor, meteorological_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Enhanced forward pass with meteorological processing."""
        
        # Standard GNN processing
        gnn_output = super().forward(x)
        
        # Apply meteorological processors if context is available
        if meteorological_context is not None:
            wind_features = self.wind_shear_processor(meteorological_context)
            thermal_features = self.thermal_gradient_processor(meteorological_context)
            moisture_features = self.moisture_convergence_processor(meteorological_context)
            
            # Combine with GNN output
            enhanced_output = gnn_output + wind_features + thermal_features + moisture_features
            return enhanced_output
        
        return gnn_output

class WindShearProcessor(nn.Module):
    """Process wind shear patterns for lightning prediction."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.processor = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.processor(x)

class ThermalGradientProcessor(nn.Module):
    """Process thermal gradients for lightning prediction."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.processor = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.processor(x)

class MoistureConvergenceProcessor(nn.Module):
    """Process moisture convergence patterns for lightning prediction."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.processor = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.processor(x)