"""
Efficient ConvNet for spatial processing, replacing Pyramid GNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class EfficientConvNet(nn.Module):
    """
    Efficient ConvNet for processing large spatial domains (710x710).
    Replaces PyramidGraphNeuralNetwork with much faster convolution operations.
    """
    
    def __init__(self,
                 input_channels: int = 256,
                 hidden_channels: int = 64,
                 output_channels: int = 64,
                 num_layers: int = 3,
                 kernel_sizes: List[int] = [3, 5, 7],
                 dropout: float = 0.1,
                 use_multiscale: bool = True,
                 use_attention: bool = True):
        """
        Initialize Efficient ConvNet.
        
        Args:
            input_channels: Number of input feature channels
            hidden_channels: Number of hidden channels
            output_channels: Number of output channels
            num_layers: Number of convolution layers
            kernel_sizes: Kernel sizes for multi-scale processing
            dropout: Dropout probability
            use_multiscale: Whether to use multi-scale feature fusion
            use_attention: Whether to use channel attention
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.num_layers = num_layers
        self.use_multiscale = use_multiscale
        self.use_attention = use_attention
        
        # Input projection
        self.input_proj = nn.Conv2d(input_channels, hidden_channels, 1, bias=False)
        self.input_norm = nn.BatchNorm2d(hidden_channels)
        
        # Multi-scale convolution branches
        if use_multiscale:
            self.multiscale_convs = nn.ModuleList()
            for kernel_size in kernel_sizes:
                branch = self._make_conv_branch(hidden_channels, kernel_size, num_layers)
                self.multiscale_convs.append(branch)
            
            # Fusion of multi-scale features
            self.scale_fusion = nn.Conv2d(
                hidden_channels * len(kernel_sizes), 
                hidden_channels, 1, bias=False
            )
            self.fusion_norm = nn.BatchNorm2d(hidden_channels)
        else:
            # Single scale processing
            self.conv_layers = self._make_conv_branch(hidden_channels, 3, num_layers)
        
        # Channel attention
        if use_attention:
            self.channel_attention = ChannelAttention(hidden_channels)
        
        # Output projection
        self.output_proj = nn.Conv2d(hidden_channels, output_channels, 1)
        
        # Residual connection
        if input_channels != output_channels:
            self.residual_proj = nn.Conv2d(input_channels, output_channels, 1)
        else:
            self.residual_proj = None
        
        print(f"   ✓ EfficientConvNet initialized: {input_channels}→{output_channels} channels")
    
    def _make_conv_branch(self, channels: int, kernel_size: int, num_layers: int) -> nn.Module:
        """Create a convolution branch with specified kernel size."""
        layers = []
        padding = kernel_size // 2
        
        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1)
            ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Efficient ConvNet.
        
        Args:
            x: Input features (batch_size, channels, height, width)
            
        Returns:
            Output features (batch_size, output_channels, height, width)
        """
        # Store input for residual connection
        residual_input = x
        
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.relu(x, inplace=True)
        
        # Multi-scale processing
        if self.use_multiscale:
            scale_outputs = []
            for conv_branch in self.multiscale_convs:
                scale_out = conv_branch(x)
                scale_outputs.append(scale_out)
            
            # Fuse multi-scale features
            x = torch.cat(scale_outputs, dim=1)
            x = self.scale_fusion(x)
            x = self.fusion_norm(x)
            x = F.relu(x, inplace=True)
        else:
            x = self.conv_layers(x)
        
        # Channel attention
        if self.use_attention:
            x = self.channel_attention(x)
        
        # Output projection
        x = self.output_proj(x)
        
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(residual_input)
        else:
            residual = residual_input
        
        x = x + residual
        
        return x

class ChannelAttention(nn.Module):
    """Channel attention mechanism for feature enhancement."""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention."""
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        attention = self.sigmoid(avg_out + max_out)
        
        return x * attention

class MultiScaleConvNet(EfficientConvNet):
    """
    Multi-scale ConvNet with dilated convolutions for larger receptive fields.
    """
    
    def __init__(self, *args, dilation_rates: List[int] = [1, 2, 4], **kwargs):
        super().__init__(*args, **kwargs)
        
        self.dilation_rates = dilation_rates
        
        # Replace multi-scale convs with dilated convs
        if self.use_multiscale:
            self.multiscale_convs = nn.ModuleList()
            for dilation in dilation_rates:
                branch = self._make_dilated_branch(self.hidden_channels, dilation)
                self.multiscale_convs.append(branch)
    
    def _make_dilated_branch(self, channels: int, dilation: int) -> nn.Module:
        """Create dilated convolution branch."""
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )