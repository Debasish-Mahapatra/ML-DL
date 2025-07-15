"""
Multi-scale fusion module for combining different resolution inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

class MultiScaleFusion(nn.Module):
    """
    Multi-scale fusion module that combines meteorological and terrain features
    at different resolutions and upsamples to target resolution.
    """
    
    def __init__(self,
                 met_channels: int = 256,
                 terrain_channels: int = 64,
                 output_channels: int = 256,
                 upsampling_factor: float = 8.33,  # 25km -> 3km
                 fusion_method: str = "terrain_guided",
                 use_progressive_upsampling: bool = True):
        """
        Initialize multi-scale fusion.
        
        Args:
            met_channels: Number of meteorological feature channels
            terrain_channels: Number of terrain feature channels  
            output_channels: Number of output feature channels
            upsampling_factor: Factor for upsampling meteorological data
            fusion_method: Method for fusion ('terrain_guided', 'concatenation', 'attention')
            use_progressive_upsampling: Whether to use progressive upsampling
        """
        super().__init__()
        
        self.met_channels = met_channels
        self.terrain_channels = terrain_channels
        self.output_channels = output_channels
        self.upsampling_factor = upsampling_factor
        self.fusion_method = fusion_method
        self.use_progressive_upsampling = use_progressive_upsampling
        
        # Terrain-guided upsampling
        if fusion_method == "terrain_guided":
            self.terrain_guided_upsample = TerrainGuidedUpsampling(
                met_channels, terrain_channels, met_channels,
                upsampling_factor, use_progressive_upsampling
            )
            fusion_input_channels = met_channels + terrain_channels
        else:
            # Standard upsampling
            self.upsample = self._create_upsampling_module()
            if fusion_method == "concatenation":
                fusion_input_channels = met_channels + terrain_channels
            else:  # attention
                fusion_input_channels = met_channels + terrain_channels
                self.spatial_attention = SpatialCrossAttention(
                    met_channels, terrain_channels, met_channels // 4
                )
        
        # Final fusion network
        self.fusion_network = nn.Sequential(
            nn.Conv2d(fusion_input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(output_channels, output_channels, 1)
        )
        
        # Residual connection projection if needed
        self.residual_proj = None
        if met_channels != output_channels:
            self.residual_proj = nn.Conv2d(met_channels, output_channels, 1)
    
    def _create_upsampling_module(self):
        """Create standard upsampling module."""
        if self.use_progressive_upsampling:
            return ProgressiveUpsampling(
                self.met_channels, 
                self.met_channels, 
                self.upsampling_factor
            )
        else:
            return nn.Sequential(
                nn.Upsample(scale_factor=self.upsampling_factor, mode='bilinear', align_corners=False),
                nn.Conv2d(self.met_channels, self.met_channels, 3, padding=1),
                nn.BatchNorm2d(self.met_channels),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, 
                met_features: torch.Tensor,
                terrain_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-scale fusion.
        
        Args:
            met_features: Meteorological features (B, met_channels, H_met, W_met) at 25km
            terrain_features: Terrain features (B, terrain_channels, H_terrain, W_terrain) at 3km
            
        Returns:
            Fused features at 3km resolution (B, output_channels, H_terrain, W_terrain)
        """
        target_size = terrain_features.shape[-2:]
        
        # Upsample meteorological features
        if self.fusion_method == "terrain_guided":
            upsampled_met = self.terrain_guided_upsample(met_features, terrain_features)
        else:
            upsampled_met = self.upsample(met_features)
            # Ensure exact target size
            if upsampled_met.shape[-2:] != target_size:
                upsampled_met = F.interpolate(
                    upsampled_met, size=target_size, 
                    mode='bilinear', align_corners=False
                )
        
        # Fusion
        if self.fusion_method in ["terrain_guided", "concatenation"]:
            fused = torch.cat([upsampled_met, terrain_features], dim=1)
        else:  # attention
            attended_met = self.spatial_attention(upsampled_met, terrain_features)
            fused = torch.cat([attended_met, terrain_features], dim=1)
        
        # Apply fusion network
        output = self.fusion_network(fused)
        
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(upsampled_met)
            output = output + residual
        else:
            output = output + upsampled_met
        
        return output

class TerrainGuidedUpsampling(nn.Module):
    """
    Terrain-guided upsampling that uses terrain features to guide
    the upsampling of meteorological data from 25km to 3km.
    """
    
    def __init__(self,
                 met_channels: int,
                 terrain_channels: int,
                 output_channels: int,
                 upsampling_factor: float,
                 use_progressive: bool = True,
                 guidance_weight: float = 0.3):
        super().__init__()
        
        self.met_channels = met_channels
        self.terrain_channels = terrain_channels
        self.output_channels = output_channels
        self.upsampling_factor = upsampling_factor
        self.guidance_weight = guidance_weight
        
        # Progressive upsampling stages
        if use_progressive:
            self.upsampling_stages = self._create_progressive_stages()
        else:
            self.upsampling_stages = self._create_single_stage()
        
        # Terrain guidance network
        self.terrain_guidance = TerrainGuidanceNetwork(
            terrain_channels, met_channels, guidance_weight
        )
        
        # Feature refinement
        self.feature_refine = nn.Sequential(
            nn.Conv2d(met_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    
    def _create_progressive_stages(self):
        """Create progressive upsampling stages."""
        stages = nn.ModuleList()
        
        # Calculate intermediate scales
        total_factor = self.upsampling_factor
        num_stages = 3  # Upsample in 3 stages
        stage_factor = total_factor ** (1.0 / num_stages)
        
        for i in range(num_stages):
            stage = nn.Sequential(
                nn.Upsample(scale_factor=stage_factor, mode='bilinear', align_corners=False),
                nn.Conv2d(self.met_channels, self.met_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.met_channels),
                nn.ReLU(inplace=True)
            )
            stages.append(stage)
        
        return stages
    
    def _create_single_stage(self):
        """Create single-stage upsampling."""
        return nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=self.upsampling_factor, mode='bilinear', align_corners=False),
                nn.Conv2d(self.met_channels, self.met_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.met_channels),
                nn.ReLU(inplace=True)
            )
        ])
    
    def forward(self, 
                met_features: torch.Tensor,
                terrain_features: torch.Tensor) -> torch.Tensor:
        """Apply terrain-guided upsampling."""
        
        # Progressive upsampling with terrain guidance
        upsampled = met_features
        
        for i, stage in enumerate(self.upsampling_stages):
            upsampled = stage(upsampled)
            
            # Apply terrain guidance at each stage
            if i == len(self.upsampling_stages) - 1:  # Final stage
                # Ensure exact size match with terrain
                target_size = terrain_features.shape[-2:]
                if upsampled.shape[-2:] != target_size:
                    upsampled = F.interpolate(
                        upsampled, size=target_size,
                        mode='bilinear', align_corners=False
                    )
                
                # Apply terrain guidance
                upsampled = self.terrain_guidance(upsampled, terrain_features)
        
        # Final feature refinement
        output = self.feature_refine(upsampled)
        
        return output

class TerrainGuidanceNetwork(nn.Module):
    """Network that uses terrain features to guide meteorological upsampling."""
    
    def __init__(self, 
                 terrain_channels: int,
                 met_channels: int,
                 guidance_weight: float = 0.3):
        super().__init__()
        
        self.guidance_weight = guidance_weight
        
        # Terrain-to-guidance projection
        self.terrain_to_guidance = nn.Sequential(
            nn.Conv2d(terrain_channels, met_channels // 2, 3, padding=1),
            nn.BatchNorm2d(met_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(met_channels // 2, met_channels, 1),
            nn.Sigmoid()  # Guidance weights between 0 and 1
        )
        
        # Terrain-informed spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(terrain_channels + met_channels, met_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(met_channels // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
                met_features: torch.Tensor,
                terrain_features: torch.Tensor) -> torch.Tensor:
        """Apply terrain guidance to meteorological features."""
        
        # Generate guidance weights from terrain
        guidance_weights = self.terrain_to_guidance(terrain_features)
        
        # Apply guidance weights
        guided_features = met_features * (1 + self.guidance_weight * guidance_weights)
        
        # Spatial attention based on terrain+met interaction
        combined = torch.cat([terrain_features, guided_features], dim=1)
        spatial_weights = self.spatial_attention(combined)
        
        # Apply spatial attention
        output = guided_features * spatial_weights
        
        return output

class ProgressiveUpsampling(nn.Module):
    """Progressive upsampling with learned interpolation."""
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 total_factor: float,
                 num_stages: int = 3):
        super().__init__()
        
        self.total_factor = total_factor
        self.num_stages = num_stages
        self.stage_factor = total_factor ** (1.0 / num_stages)
        
        # Create upsampling stages
        self.stages = nn.ModuleList()
        
        for i in range(num_stages):
            stage = nn.Sequential(
                nn.Upsample(scale_factor=self.stage_factor, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            )
            self.stages.append(stage)
        
        # Final projection
        self.final_proj = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply progressive upsampling."""
        for stage in self.stages:
            x = stage(x)
        
        x = self.final_proj(x)
        return x

class SpatialCrossAttention(nn.Module):
    """Spatial cross-attention between meteorological and terrain features."""
    
    def __init__(self, 
                 met_channels: int,
                 terrain_channels: int,
                 attention_channels: int):
        super().__init__()
        
        self.attention_channels = attention_channels
        
        # Query from meteorological, Key/Value from terrain
        self.met_to_q = nn.Conv2d(met_channels, attention_channels, 1)
        self.terrain_to_k = nn.Conv2d(terrain_channels, attention_channels, 1)
        self.terrain_to_v = nn.Conv2d(terrain_channels, attention_channels, 1)
        
        self.scale = attention_channels ** -0.5
        self.to_out = nn.Conv2d(attention_channels, met_channels, 1)
        
    def forward(self, 
                met_features: torch.Tensor,
                terrain_features: torch.Tensor) -> torch.Tensor:
        """Apply spatial cross-attention."""
        B, C, H, W = met_features.shape
        
        # Generate Q, K, V
        q = self.met_to_q(met_features).view(B, self.attention_channels, -1)
        k = self.terrain_to_k(terrain_features).view(B, self.attention_channels, -1)
        v = self.terrain_to_v(terrain_features).view(B, self.attention_channels, -1)
        
        # Attention computation
        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(v, attn.transpose(-2, -1))
        out = out.view(B, self.attention_channels, H, W)
        
        # Project to output
        out = self.to_out(out)
        
        return met_features + out  # Residual connection