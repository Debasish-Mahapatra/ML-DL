"""
Lightweight Transformer components optimized for memory efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from einops import rearrange

class LightweightTransformer(nn.Module):
    """
    Memory-efficient transformer for capturing mesoscale patterns and temporal dependencies.
    Uses linear attention for reduced memory complexity.
    """
    
    def __init__(self,
                 input_channels: int = 256,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 attention_type: str = "linear",  # "linear", "standard", "performer"
                 max_seq_length: int = 1024,
                 use_spatial_encoding: bool = True):
        """
        Initialize Lightweight Transformer.
        
        Args:
            input_channels: Number of input feature channels
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            attention_type: Type of attention mechanism
            max_seq_length: Maximum sequence length for positional encoding
            use_spatial_encoding: Whether to use spatial positional encoding
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.use_spatial_encoding = use_spatial_encoding
        
        # Input projection
        self.input_projection = nn.Linear(input_channels, hidden_dim)
        
        # Positional encoding
        if use_spatial_encoding:
            self.spatial_encoding = SpatialPositionalEncoding(hidden_dim, max_seq_length)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            LightweightTransformerLayer(
                hidden_dim, num_heads, dropout, attention_type
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, input_channels)
        
        # Layer normalization
        self.final_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through lightweight transformer.
        
        Args:
            x: Input features (batch_size, channels, height, width)
            
        Returns:
            Output features (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.shape
        
        # Reshape to sequence format: (batch_size, seq_length, channels)
        x_seq = rearrange(x, 'b c h w -> b (h w) c')
        
        # Project to hidden dimension
        x_seq = self.input_projection(x_seq)
        
        # Add spatial positional encoding
        if self.use_spatial_encoding:
            x_seq = self.spatial_encoding(x_seq, height, width)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x_seq = layer(x_seq)
        
        # Final normalization
        x_seq = self.final_norm(x_seq)
        
        # Project back to original channels
        x_seq = self.output_projection(x_seq)
        
        # Reshape back to grid format
        output = rearrange(x_seq, 'b (h w) c -> b c h w', h=height, w=width)
        
        # Residual connection
        return output + x

class LightweightTransformerLayer(nn.Module):
    """Individual transformer layer with efficient attention."""
    
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 dropout: float,
                 attention_type: str):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attention_type = attention_type
        
        # Attention mechanism
        if attention_type == "linear":
            self.attention = LinearAttention(hidden_dim, num_heads, dropout)
        elif attention_type == "performer":
            self.attention = PerformerAttention(hidden_dim, num_heads, dropout)
        else:  # standard
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(hidden_dim, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer layer."""
        
        # Self-attention with residual connection
        if self.attention_type in ["linear", "performer"]:
            attn_output = self.attention(x)
        else:  # standard attention
            attn_output, _ = self.attention(x, x, x)
        
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x

class LinearAttention(nn.Module):
    """
    Linear attention mechanism with O(n) complexity instead of O(n²).
    Based on "Transformers are RNNs" paper.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        
        # Linear projections
        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)
        
        # Feature map for linearization
        self.feature_map = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear attention."""
        batch_size, seq_length, _ = x.shape
        
        # Generate Q, K, V
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', 
                           qkv=3, h=self.num_heads, d=self.head_dim)
        
        # Apply feature map to linearize attention
        q = self.feature_map(q) + 1e-6
        k = self.feature_map(k) + 1e-6
        
        # Linear attention computation
        # Instead of softmax(QK^T)V, compute Q(K^TV)
        kv = torch.einsum('bhnd,bhnf->bhdf', k, v)
        normalizer = torch.einsum('bhnd,bhd->bhn', q, k.sum(dim=2))
        
        out = torch.einsum('bhnd,bhdf->bhnf', q, kv)
        out = out / (normalizer.unsqueeze(-1) + 1e-6)
        
        # Reshape and project output
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        return out

class PerformerAttention(nn.Module):
    """
    Performer attention using random feature approximation.
    Provides linear complexity while maintaining performance.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, num_features: int = 64):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_features = num_features
        
        # Linear projections
        self.to_qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, hidden_dim)
        
        # Random features for approximation
        self.register_buffer('random_features', torch.randn(num_features, self.head_dim))
        
    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random feature map."""
        # x: (batch, heads, seq, head_dim)
        x_proj = torch.einsum('...d,fd->...f', x, self.random_features) / math.sqrt(self.num_features)
        return torch.exp(x_proj - x.norm(dim=-1, keepdim=True))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Performer attention."""
        batch_size, seq_length, _ = x.shape
        
        # Generate Q, K, V
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d',
                           qkv=3, h=self.num_heads, d=self.head_dim)
        
        # Apply feature maps
        q_prime = self.feature_map(q)
        k_prime = self.feature_map(k)
        
        # Performer attention computation
        kv = torch.einsum('bhnd,bhnf->bhdf', k_prime, v)
        normalizer = torch.einsum('bhnd,bhd->bhn', q_prime, k_prime.sum(dim=2))
        
        out = torch.einsum('bhnd,bhdf->bhnf', q_prime, kv)
        out = out / (normalizer.unsqueeze(-1) + 1e-6)
        
        # Reshape and project output
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        return out

class SpatialPositionalEncoding(nn.Module):
    """
    2D spatial positional encoding for grid-based data.
    """
    
    def __init__(self, hidden_dim: int, max_length: int = 1024):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Learnable spatial embeddings
        self.x_embedding = nn.Embedding(max_length, hidden_dim // 2)
        self.y_embedding = nn.Embedding(max_length, hidden_dim // 2)
        
    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Add spatial positional encoding."""
        batch_size, seq_length, hidden_dim = x.shape
        
        # Generate spatial coordinates
        y_coords = torch.arange(height, device=x.device).repeat_interleave(width)
        x_coords = torch.arange(width, device=x.device).repeat(height)
        
        # Get embeddings
        x_embed = self.x_embedding(x_coords)
        y_embed = self.y_embedding(y_coords)
        
        # Combine spatial embeddings
        spatial_embed = torch.cat([x_embed, y_embed], dim=-1)
        spatial_embed = spatial_embed.unsqueeze(0).expand(batch_size, -1, -1)
        
        return x + spatial_embed

class FeedForwardNetwork(nn.Module):
    """Feed-forward network for transformer."""
    
    def __init__(self, hidden_dim: int, dropout: float, expansion_factor: int = 4):
        super().__init__()
        
        expanded_dim = hidden_dim * expansion_factor
        
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, expanded_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expanded_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SpatialTransformer(LightweightTransformer):
    """
    Specialized transformer for spatial pattern recognition in meteorological data.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Spatial pattern recognition components
        self.mesoscale_attention = MesoscaleAttention(self.hidden_dim, self.num_heads)
        self.local_pattern_extractor = LocalPatternExtractor(self.hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced forward pass with spatial pattern processing."""
        
        # Standard transformer processing
        transformer_output = super().forward(x)
        
        # Apply mesoscale attention
        mesoscale_features = self.mesoscale_attention(transformer_output)
        
        # Extract local patterns
        local_features = self.local_pattern_extractor(transformer_output)
        
        # Combine features
        enhanced_output = transformer_output + mesoscale_features + local_features
        
        return enhanced_output

class MesoscaleAttention(nn.Module):
    """Attention mechanism focused on mesoscale meteorological patterns."""
    
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Multi-scale convolutions for mesoscale pattern detection
        self.mesoscale_convs = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim // 4, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 9]  # Different scales
        ])
        
        # Attention weights for different scales
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, 4, 1),
            nn.Softmax(dim=1)
        )
        
        # Final projection
        self.output_proj = nn.Conv2d(hidden_dim, hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply mesoscale attention."""
        
        # Extract features at different scales
        scale_features = []
        for conv in self.mesoscale_convs:
            scale_feat = conv(x)
            scale_features.append(scale_feat)
        
        # Concatenate scale features
        multi_scale = torch.cat(scale_features, dim=1)
        
        # Compute attention weights for each scale
        scale_weights = self.scale_attention(multi_scale)
        
        # Apply weighted combination
        weighted_features = sum(w.unsqueeze(2).unsqueeze(3) * feat 
                               for w, feat in zip(scale_weights.split(1, dim=1), scale_features))
        
        # Final projection
        output = self.output_proj(weighted_features)
        
        return output

class LocalPatternExtractor(nn.Module):
    """Extract local meteorological patterns relevant for lightning."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Local pattern detection
        self.local_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim//4),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 1)
        )
        
        # Pattern enhancement
        self.pattern_enhance = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim//2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim//2, hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and enhance local patterns."""
        
        # Extract local patterns
        local_patterns = self.local_conv(x)
        
        # Enhance important patterns
        enhancement_weights = self.pattern_enhance(local_patterns)
        enhanced_patterns = local_patterns * enhancement_weights
        
        return enhanced_patterns