"""
Multi-Resolution Terrain Processor for Strategy 3 Implementation

This module provides enhanced terrain processing capabilities that work with the
multi-resolution fusion approach. It replaces the simple terrain encoder with
a more sophisticated system that understands how terrain features at different
scales influence lightning formation.

Key improvements:
- Processes terrain at multiple scales simultaneously
- Extracts physically meaningful terrain features (gradients, convergence zones)
- Provides terrain context for meteorological feature distribution
- Maintains terrain information at appropriate scales for fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import numpy as np

class MultiResolutionTerrainProcessor(nn.Module):
    """
    Enhanced terrain processor that works with multi-resolution fusion.
    
    This processor understands that terrain influences lightning at multiple scales:
    - Large-scale terrain patterns (10-25km) influence mesoscale meteorology
    - Medium-scale terrain features (1-10km) create convergence zones
    - Fine-scale terrain details (0.1-1km) affect local convective initiation
    """
    
    def __init__(self,
                 in_channels: int = 1,              # Input elevation data
                 base_channels: int = 64,           # Base number of channels
                 output_channels: int = 128,        # Output channels for fusion
                 scales: List[int] = [1, 3, 9],     # Multi-scale processing kernels
                 target_resolution_km: int = 3,     # Target output resolution
                 input_resolution_km: int = 1,      # Input terrain resolution
                 extract_gradients: bool = True,    # Extract terrain gradients
                 extract_convergence: bool = True,  # Extract convergence zones
                 extract_roughness: bool = True):   # Extract terrain roughness
        """
        Initialize multi-resolution terrain processor.
        
        Args:
            in_channels: Number of input channels (1 for elevation)
            base_channels: Base number of feature channels
            output_channels: Number of output channels for fusion
            scales: List of scales for multi-scale processing
            target_resolution_km: Target resolution for output
            input_resolution_km: Input terrain resolution
            extract_gradients: Whether to extract terrain gradients
            extract_convergence: Whether to extract convergence zones
            extract_roughness: Whether to extract terrain roughness
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.scales = scales
        self.target_resolution_km = target_resolution_km
        self.input_resolution_km = input_resolution_km
        self.extract_gradients = extract_gradients
        self.extract_convergence = extract_convergence
        self.extract_roughness = extract_roughness
        
        # Calculate downsampling factor
        self.downsample_factor = input_resolution_km / target_resolution_km  # 1/3 = 0.33
        
        # Multi-scale terrain feature extractors
        self.scale_extractors = nn.ModuleList()
        for scale in scales:
            extractor = TerrainScaleExtractor(
                in_channels=in_channels,
                out_channels=base_channels,
                scale=scale,
                extract_gradients=extract_gradients,
                extract_convergence=extract_convergence,
                extract_roughness=extract_roughness
            )
            self.scale_extractors.append(extractor)
        
        # Feature combination across scales
        total_scale_channels = len(scales) * base_channels
        self.scale_combiner = nn.Sequential(
            nn.Conv2d(total_scale_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
        
        # Intelligent downsampling to target resolution
        self.intelligent_downsampler = IntelligentTerrainDownsampler(
            in_channels=output_channels,
            out_channels=output_channels,
            downsample_factor=self.downsample_factor
        )
        
        # Final terrain feature refinement
        self.final_refinement = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=1)
        )
        
        print(f"✅ Multi-Resolution Terrain Processor initialized:")
        print(f"   - Input resolution: {input_resolution_km}km")
        print(f"   - Target resolution: {target_resolution_km}km")
        print(f"   - Processing scales: {scales}")
        print(f"   - Feature extraction: gradients={extract_gradients}, convergence={extract_convergence}, roughness={extract_roughness}")
    
    def forward(self, 
                elevation_data: torch.Tensor,    # (B, 1, H_1km, W_1km)
                target_size: Tuple[int, int]     # (H_3km, W_3km)
                ) -> Dict[str, torch.Tensor]:
        """
        Process terrain data at multiple resolutions.
        
        Args:
            elevation_data: Input elevation data at 1km resolution
            target_size: Target spatial size for 3km output
            
        Returns:
            Dictionary containing terrain features at different scales
        """
        # Extract features at multiple scales
        scale_features = []
        for extractor in self.scale_extractors:
            scale_feat = extractor(elevation_data)
            scale_features.append(scale_feat)
        
        # Combine features across scales
        combined_features = torch.cat(scale_features, dim=1)
        combined_features = self.scale_combiner(combined_features)
        
        # Intelligent downsampling to target resolution
        downsampled_features = self.intelligent_downsampler(combined_features, target_size)
        
        # Final refinement
        refined_features = self.final_refinement(downsampled_features)
        
        # Prepare output dictionary
        output = {
            'terrain_features': refined_features,
            'raw_elevation': elevation_data,
            'target_size': target_size,
            'scale_features': scale_features
        }
        
        return output

class TerrainScaleExtractor(nn.Module):
    """
    Extracts terrain features at a specific scale.
    Different scales capture different physical processes.
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 scale: int,
                 extract_gradients: bool,
                 extract_convergence: bool,
                 extract_roughness: bool):
        super().__init__()
        
        self.scale = scale
        self.extract_gradients = extract_gradients
        self.extract_convergence = extract_convergence
        self.extract_roughness = extract_roughness
        
        # Base terrain processing
        self.base_processor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # Feature extractors
        feature_channels = out_channels // 4
        total_channels = feature_channels
        
        if extract_gradients:
            self.gradient_extractor = TerrainGradientExtractor(scale)
            total_channels += feature_channels
        
        if extract_convergence:
            self.convergence_extractor = TerrainConvergenceExtractor(scale)
            total_channels += feature_channels
        
        if extract_roughness:
            self.roughness_extractor = TerrainRoughnessExtractor(scale)
            total_channels += feature_channels
        
        # Combine all features
        self.feature_combiner = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, elevation_data: torch.Tensor) -> torch.Tensor:
        """Extract terrain features at this scale."""
        # Base processing
        base_features = self.base_processor(elevation_data)
        
        # Additional feature extraction
        features_to_combine = [base_features]
        
        if self.extract_gradients:
            gradient_features = self.gradient_extractor(elevation_data)
            gradient_features = F.adaptive_avg_pool2d(gradient_features, base_features.shape[-2:])
            features_to_combine.append(gradient_features)
        
        if self.extract_convergence:
            convergence_features = self.convergence_extractor(elevation_data)
            convergence_features = F.adaptive_avg_pool2d(convergence_features, base_features.shape[-2:])
            features_to_combine.append(convergence_features)
        
        if self.extract_roughness:
            roughness_features = self.roughness_extractor(elevation_data)
            roughness_features = F.adaptive_avg_pool2d(roughness_features, base_features.shape[-2:])
            features_to_combine.append(roughness_features)
        
        # Combine all features
        combined = torch.cat(features_to_combine, dim=1)
        output = self.feature_combiner(combined)
        
        return output

class TerrainGradientExtractor(nn.Module):
    """Extracts terrain gradients that influence orographic lifting."""
    
    def __init__(self, scale: int):
        super().__init__()
        self.scale = scale
        
        # Sobel operators for gradient computation
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        # Feature processor
        self.processor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 3 channels: grad_x, grad_y, grad_mag
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, elevation_data: torch.Tensor) -> torch.Tensor:
        """Extract terrain gradients."""
        # Compute gradients
        grad_x = F.conv2d(elevation_data, self.sobel_x, padding=1)
        grad_y = F.conv2d(elevation_data, self.sobel_y, padding=1)
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        # Combine gradient components
        gradient_features = torch.cat([grad_x, grad_y, grad_magnitude], dim=1)
        
        # Process features
        output = self.processor(gradient_features)
        
        return output

class TerrainConvergenceExtractor(nn.Module):
    """Extracts terrain convergence zones that enhance convective initiation."""
    
    def __init__(self, scale: int):
        super().__init__()
        self.scale = scale
        
        # Laplacian operator for convergence detection
        self.register_buffer('laplacian', torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3))
        
        # Feature processor
        self.processor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, elevation_data: torch.Tensor) -> torch.Tensor:
        """Extract terrain convergence zones."""
        # Compute Laplacian (measures local curvature)
        convergence = F.conv2d(elevation_data, self.laplacian, padding=1)
        
        # Process convergence features
        output = self.processor(convergence)
        
        return output

class TerrainRoughnessExtractor(nn.Module):
    """Extracts terrain roughness that affects boundary layer turbulence."""
    
    def __init__(self, scale: int):
        super().__init__()
        self.scale = scale
        
        # Multi-scale roughness computation
        self.roughness_scales = [3, 5, 7]
        
        # Feature processor
        self.processor = nn.Sequential(
            nn.Conv2d(len(self.roughness_scales), 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, elevation_data: torch.Tensor) -> torch.Tensor:
        """Extract terrain roughness at multiple scales."""
        roughness_features = []
        
        for scale in self.roughness_scales:
            # Compute standard deviation in local neighborhood
            kernel = torch.ones(1, 1, scale, scale, device=elevation_data.device) / (scale * scale)
            local_mean = F.conv2d(elevation_data, kernel, padding=scale//2)
            local_var = F.conv2d(elevation_data**2, kernel, padding=scale//2) - local_mean**2
            local_std = torch.sqrt(local_var + 1e-6)
            roughness_features.append(local_std)
        
        # Combine roughness at different scales
        combined_roughness = torch.cat(roughness_features, dim=1)
        
        # Process roughness features
        output = self.processor(combined_roughness)
        
        return output

class IntelligentTerrainDownsampler(nn.Module):
    """
    Intelligently downsamples terrain features while preserving important characteristics.
    Uses learned attention to focus on terrain features that matter for lightning.
    """
    
    def __init__(self, in_channels: int, out_channels: int, downsample_factor: float):
        super().__init__()
        
        self.downsample_factor = downsample_factor
        
        # Terrain importance attention
        self.importance_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Adaptive pooling with learned weights
        self.adaptive_pooling = nn.AdaptiveAvgPool2d(None)  # Size set dynamically
        
        # Feature refinement after downsampling
        self.post_downsample_refinement = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
    
    def forward(self, terrain_features: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        Intelligently downsample terrain features.
        
        Args:
            terrain_features: Input terrain features at 1km resolution
            target_size: Target size for 3km resolution
            
        Returns:
            Downsampled terrain features at 3km resolution
        """
        # Compute importance weights
        importance = self.importance_attention(terrain_features)
        
        # Apply importance weighting
        weighted_features = terrain_features * importance
        
        # Downsample to target size
        if terrain_features.shape[-2:] != target_size:
            downsampled = F.interpolate(
                weighted_features,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
        else:
            downsampled = weighted_features
        
        # Post-downsampling refinement
        refined = self.post_downsample_refinement(downsampled)
        
        return refined