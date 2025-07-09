"""
Fusion module for combining different meteorological data sources.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
import torch.nn.functional as F

class MeteorologicalFusion(nn.Module):
    """
    Fusion module for combining CAPE and ERA5 features.
    Currently handles CAPE-only, expandable for ERA5.
    """
    
    def __init__(self,
                 cape_channels: int = 256,
                 era5_channels: Optional[int] = None,
                 output_channels: int = 256,
                 fusion_method: str = "concatenation",
                 use_attention: bool = True,
                 hidden_dim: int = 512):
        """
        Initialize meteorological fusion.
        
        Args:
            cape_channels: Number of CAPE feature channels
            era5_channels: Number of ERA5 feature channels (None for CAPE-only)
            output_channels: Number of output feature channels
            fusion_method: Method for fusion ('concatenation', 'addition', 'attention')
            use_attention: Whether to use attention-based fusion
            hidden_dim: Hidden dimension for fusion networks
        """
        super().__init__()
        
        self.cape_channels = cape_channels
        self.era5_channels = era5_channels
        self.output_channels = output_channels
        self.fusion_method = fusion_method
        self.use_attention = use_attention
        
        # Determine if we're in CAPE-only mode or CAPE+ERA5 mode
        self.cape_only = era5_channels is None
        
        if self.cape_only:
            # CAPE-only mode: simple projection
            self.cape_projection = nn.Sequential(
                nn.Conv2d(cape_channels, output_channels, 1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
        else:
            # CAPE+ERA5 mode: complex fusion
            self._setup_dual_fusion(hidden_dim)
    
    def _setup_dual_fusion(self, hidden_dim: int):
        """Setup fusion for CAPE+ERA5 mode."""
        
        # Individual projections
        self.cape_projection = nn.Sequential(
            nn.Conv2d(self.cape_channels, hidden_dim // 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        self.era5_projection = nn.Sequential(
            nn.Conv2d(self.era5_channels, hidden_dim // 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        if self.fusion_method == "concatenation":
            fusion_input_dim = hidden_dim
        elif self.fusion_method == "addition":
            fusion_input_dim = hidden_dim // 2
        else:  # attention
            fusion_input_dim = hidden_dim
            self.cross_attention = CrossModalAttention(
                hidden_dim // 2, hidden_dim // 2, hidden_dim // 4
            )
        
        # Final fusion network
        self.fusion_network = nn.Sequential(
            nn.Conv2d(fusion_input_dim, self.output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.output_channels, self.output_channels, 1)
        )
        
        # Attention weights for adaptive fusion
        if self.use_attention:
            self.fusion_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(fusion_input_dim, fusion_input_dim // 8, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(fusion_input_dim // 8, fusion_input_dim, 1),
                nn.Sigmoid()
            )
    
    def forward(self, 
                cape_features: torch.Tensor,
                era5_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through meteorological fusion.
        
        Args:
            cape_features: CAPE features (B, cape_channels, H, W)
            era5_features: ERA5 features (B, era5_channels, H, W) - optional
            
        Returns:
            Fused meteorological features (B, output_channels, H, W)
        """
        
        if self.cape_only or era5_features is None:
            # CAPE-only mode
            return self.cape_projection(cape_features)
        
        # CAPE+ERA5 mode
        cape_proj = self.cape_projection(cape_features)
        era5_proj = self.era5_projection(era5_features)
        
        # Ensure spatial dimensions match
        if cape_proj.shape[-2:] != era5_proj.shape[-2:]:
            era5_proj = F.interpolate(
                era5_proj, size=cape_proj.shape[-2:], 
                mode='bilinear', align_corners=False
            )
        
        # Apply fusion method
        if self.fusion_method == "concatenation":
            fused = torch.cat([cape_proj, era5_proj], dim=1)
        elif self.fusion_method == "addition":
            fused = cape_proj + era5_proj
        else:  # attention
            attended_features = self.cross_attention(cape_proj, era5_proj)
            fused = torch.cat([attended_features, cape_proj, era5_proj], dim=1)
        
        # Apply attention weighting if enabled
        if self.use_attention:
            attention_weights = self.fusion_attention(fused)
            fused = fused * attention_weights
        
        # Final fusion
        output = self.fusion_network(fused)
        
        return output

class CrossModalAttention(nn.Module):
    """Cross-modal attention between CAPE and ERA5 features."""
    
    def __init__(self, 
                 cape_dim: int, 
                 era5_dim: int, 
                 attention_dim: int):
        super().__init__()
        
        self.cape_dim = cape_dim
        self.era5_dim = era5_dim
        self.attention_dim = attention_dim
        
        # Query, Key, Value projections
        self.cape_to_q = nn.Conv2d(cape_dim, attention_dim, 1)
        self.era5_to_k = nn.Conv2d(era5_dim, attention_dim, 1)
        self.era5_to_v = nn.Conv2d(era5_dim, attention_dim, 1)
        
        self.scale = attention_dim ** -0.5
        self.to_out = nn.Conv2d(attention_dim, cape_dim, 1)
    
    def forward(self, cape_features: torch.Tensor, era5_features: torch.Tensor) -> torch.Tensor:
        """Apply cross-modal attention."""
        B, C, H, W = cape_features.shape
        
        # Generate Q, K, V
        q = self.cape_to_q(cape_features).view(B, self.attention_dim, -1)
        k = self.era5_to_k(era5_features).view(B, self.attention_dim, -1)
        v = self.era5_to_v(era5_features).view(B, self.attention_dim, -1)
        
        # Attention computation
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(v, attn.transpose(-2, -1))
        out = out.view(B, self.attention_dim, H, W)
        
        # Project to output
        out = self.to_out(out)
        
        return out

class AdaptiveFusion(nn.Module):
    """Adaptive fusion that learns optimal combination weights."""
    
    def __init__(self, feature_dims: List[int], output_dim: int):
        super().__init__()
        
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        self.num_modalities = len(feature_dims)
        
        # Individual feature processors
        self.processors = nn.ModuleList([
            nn.Conv2d(dim, output_dim, 1) for dim in feature_dims
        ])
        
        # Adaptive weight generation
        total_dim = sum(feature_dims)
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_dim, total_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_dim // 4, self.num_modalities, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, feature_list: List[torch.Tensor]) -> torch.Tensor:
        """Adaptively fuse multiple feature modalities."""
        assert len(feature_list) == self.num_modalities
        
        # Process individual features
        processed = [proc(feat) for proc, feat in zip(self.processors, feature_list)]
        
        # Generate adaptive weights
        concatenated = torch.cat(feature_list, dim=1)
        weights = self.weight_generator(concatenated)  # (B, num_modalities, 1, 1)
        
        # Weighted combination
        fused = sum(w.unsqueeze(2).unsqueeze(3) * feat 
                   for w, feat in zip(weights.split(1, dim=1), processed))
        
        return fused
