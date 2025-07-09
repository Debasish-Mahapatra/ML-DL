"""
CNN Encoder for CAPE (Convective Available Potential Energy) data.
"""

import torch
import torch.nn as nn
from typing import List, Tuple
from einops import rearrange

class CAPEEncoder(nn.Module):
    """
    2D CNN encoder for CAPE meteorological data at 25km resolution.
    Extracts multi-scale features for lightning prediction.
    """
    
    def __init__(self,
                 in_channels: int = 1,
                 channels: List[int] = [32, 64, 128, 256],
                 kernel_sizes: List[int] = [7, 5, 3, 3],
                 activation: str = "relu",
                 dropout: float = 0.1,
                 use_batch_norm: bool = True,
                 use_residual: bool = True):
        """
        Initialize CAPE encoder.
        
        Args:
            in_channels: Number of input channels (1 for CAPE)
            channels: List of channel dimensions for each layer
            kernel_sizes: List of kernel sizes for each layer
            activation: Activation function name
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.channels = channels
        self.use_residual = use_residual
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build encoder layers
        self.layers = nn.ModuleList()
        
        # First layer
        first_layer = self._make_conv_block(
            in_channels, channels[0], kernel_sizes[0], 
            use_batch_norm, dropout
        )
        self.layers.append(first_layer)
        
        # Subsequent layers
        for i in range(1, len(channels)):
            layer = self._make_conv_block(
                channels[i-1], channels[i], kernel_sizes[i],
                use_batch_norm, dropout
            )
            self.layers.append(layer)
        
        # Feature dimension after encoding
        self.output_channels = channels[-1]
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_conv_block(self, 
                        in_ch: int, 
                        out_ch: int, 
                        kernel_size: int,
                        use_batch_norm: bool, 
                        dropout: float) -> nn.Module:
        """Create a convolutional block with optional batch norm and dropout."""
        layers = []
        
        # Convolution
        padding = kernel_size // 2
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=not use_batch_norm))
        
        # Batch normalization
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        
        # Activation
        layers.append(self.activation)
        
        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        return nn.Sequential(*layers)
    
    def _make_residual_block(self, channels: int, kernel_size: int) -> nn.Module:
        """Create a residual block."""
        padding = kernel_size // 2
        
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(channels),
            self.activation,
            nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(channels)
        )
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CAPE encoder.
        
        Args:
            x: CAPE data tensor (batch_size, 1, height, width)
            
        Returns:
            Encoded features (batch_size, output_channels, height, width)
        """
        features = x
        
        # Pass through encoder layers
        for layer in self.layers:
            features = layer(features)
        
        return features

class EnhancedCAPEEncoder(CAPEEncoder):
    """
    Enhanced CAPE encoder with attention and multi-scale features.
    """
    
    def __init__(self, 
                 *args,
                 use_attention: bool = True,
                 attention_heads: int = 8,
                 multiscale_features: bool = True,
                 **kwargs):
        """
        Initialize enhanced CAPE encoder.
        
        Args:
            use_attention: Whether to use spatial attention
            attention_heads: Number of attention heads
            multiscale_features: Whether to collect multi-scale features
        """
        super().__init__(*args, **kwargs)
        
        self.use_attention = use_attention
        self.multiscale_features = multiscale_features
        
        # Spatial attention module
        if use_attention:
            self.attention = SpatialAttention(self.output_channels, attention_heads)
        
        # Multi-scale feature collection
        if multiscale_features:
            self.multiscale_projections = nn.ModuleList([
                nn.Conv2d(ch, self.output_channels // 4, 1) 
                for ch in self.channels[:-1]
            ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Enhanced forward pass with attention and multi-scale features."""
        features = x
        multiscale_feats = []
        
        # Pass through encoder layers and collect multi-scale features
        for i, layer in enumerate(self.layers):
            features = layer(features)
            
            # Collect intermediate features for multi-scale fusion
            if self.multiscale_features and i < len(self.layers) - 1:
                projected = self.multiscale_projections[i](features)
                # Resize to match final feature size
                if projected.shape[-2:] != features.shape[-2:]:
                    projected = nn.functional.interpolate(
                        projected, size=features.shape[-2:], 
                        mode='bilinear', align_corners=False
                    )
                multiscale_feats.append(projected)
        
        # Apply spatial attention
        if self.use_attention:
            features = self.attention(features)
        
        # Concatenate multi-scale features
        if self.multiscale_features and multiscale_feats:
            final_multiscale = torch.cat(multiscale_feats, dim=1)
            features = torch.cat([features, final_multiscale], dim=1)
        
        return features

class SpatialAttention(nn.Module):
    """Spatial attention mechanism for feature enhancement."""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.to_out = nn.Conv2d(channels, channels, 1)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        
        # Generate query, key, value
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', 
                           heads=self.num_heads, qkv=3)
        
        # Attention computation
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', h=h, w=w)
        
        # Final projection
        out = self.to_out(out)
        
        return x + out  # Residual connection
