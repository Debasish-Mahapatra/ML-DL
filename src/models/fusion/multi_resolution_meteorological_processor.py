"""
Multi-Resolution Meteorological Processor for Strategy 3 Implementation

This module provides enhanced meteorological processing that preserves information
at native 25km resolution while preparing it for intelligent combination with
high-resolution terrain data.

Key features:
- Processes meteorological data at native resolution (no upsampling)
- Extracts physically meaningful meteorological patterns
- Prepares meteorological features for distribution to higher resolution
- Maintains consistency with atmospheric physics principles
- Provides context for terrain-meteorological interactions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

class MultiResolutionMeteorologicalProcessor(nn.Module):
    """
    Enhanced meteorological processor that works with multi-resolution fusion.
    
    This processor recognizes that meteorological phenomena occur at their natural
    scales and should be processed accordingly. Instead of artificially upsampling
    meteorological data, it extracts rich feature representations at native resolution
    that can be intelligently distributed to higher resolutions based on terrain
    and other local factors.
    """
    
    def __init__(self,
                 cape_channels: int = 256,          # CAPE encoder output channels
                 era5_channels: int = 256,          # ERA5 encoder output channels (future)
                 output_channels: int = 256,        # Output channels for fusion
                 meteorological_scales: List[int] = [3, 5, 7],  # Multi-scale processing
                 cape_only_mode: bool = True,       # Currently CAPE-only
                 extract_patterns: bool = True,     # Extract meteorological patterns
                 extract_gradients: bool = True,    # Extract meteorological gradients
                 native_resolution_km: int = 25):   # Native meteorological resolution
        """
        Initialize multi-resolution meteorological processor.
        
        Args:
            cape_channels: Number of CAPE feature channels
            era5_channels: Number of ERA5 feature channels (future use)
            output_channels: Number of output channels for fusion
            meteorological_scales: List of scales for pattern extraction
            cape_only_mode: Whether to process only CAPE data
            extract_patterns: Whether to extract meteorological patterns
            extract_gradients: Whether to extract meteorological gradients
            native_resolution_km: Native resolution of meteorological data
        """
        super().__init__()
        
        self.cape_channels = cape_channels
        self.era5_channels = era5_channels
        self.output_channels = output_channels
        self.meteorological_scales = meteorological_scales
        self.cape_only_mode = cape_only_mode
        self.extract_patterns = extract_patterns
        self.extract_gradients = extract_gradients
        self.native_resolution_km = native_resolution_km
        
        # CAPE feature processor (always present)
        self.cape_processor = CAPEFeatureProcessor(
            in_channels=cape_channels,
            out_channels=cape_channels,
            scales=meteorological_scales,
            extract_patterns=extract_patterns,
            extract_gradients=extract_gradients
        )
        
        # ERA5 feature processor (future implementation)
        if not cape_only_mode:
            self.era5_processor = ERA5FeatureProcessor(
                in_channels=era5_channels,
                out_channels=era5_channels,
                scales=meteorological_scales,
                extract_patterns=extract_patterns,
                extract_gradients=extract_gradients
            )
        else:
            self.era5_processor = None
        
        # Meteorological fusion (combines CAPE and ERA5 if available)
        fusion_input_channels = cape_channels
        if not cape_only_mode:
            fusion_input_channels += era5_channels
        
        self.meteorological_fusion = MeteorologicalFeatureFusion(
            input_channels=fusion_input_channels,
            output_channels=output_channels,
            cape_only_mode=cape_only_mode
        )
        
        # Meteorological pattern analyzer
        self.pattern_analyzer = MeteorologicalPatternAnalyzer(
            in_channels=output_channels,
            out_channels=output_channels,
            scales=meteorological_scales
        )
        
        # Distribution preparator (prepares features for spatial distribution)
        self.distribution_preparator = DistributionPreparator(
            in_channels=output_channels,
            out_channels=output_channels,
            native_resolution_km=native_resolution_km
        )
        
        print(f"✅ Multi-Resolution Meteorological Processor initialized:")
        print(f"   - CAPE-only mode: {cape_only_mode}")
        print(f"   - Native resolution: {native_resolution_km}km")
        print(f"   - Processing scales: {meteorological_scales}")
        print(f"   - Pattern extraction: {extract_patterns}")
        print(f"   - Gradient extraction: {extract_gradients}")
    
    def forward(self, 
                cape_features: torch.Tensor,                    # (B, cape_channels, H_25km, W_25km)
                era5_features: Optional[torch.Tensor] = None    # (B, era5_channels, H_25km, W_25km)
                ) -> Dict[str, torch.Tensor]:
        """
        Process meteorological features at native resolution.
        
        Args:
            cape_features: CAPE features at 25km resolution
            era5_features: ERA5 features at 25km resolution (future)
            
        Returns:
            Dictionary containing processed meteorological features
        """
        # Process CAPE features
        processed_cape = self.cape_processor(cape_features)
        
        # Process ERA5 features if available
        if era5_features is not None and self.era5_processor is not None:
            processed_era5 = self.era5_processor(era5_features)
        else:
            processed_era5 = None
        
        # Fuse meteorological features
        fused_meteorological = self.meteorological_fusion(processed_cape, processed_era5)
        
        # Analyze meteorological patterns
        pattern_analyzed = self.pattern_analyzer(fused_meteorological)
        
        # Prepare for distribution to higher resolution
        distribution_ready = self.distribution_preparator(pattern_analyzed)
        
        # Prepare output dictionary
        output = {
            'meteorological_features': distribution_ready,
            'cape_features': processed_cape,
            'era5_features': processed_era5,
            'fused_features': fused_meteorological,
            'pattern_features': pattern_analyzed,
            'native_resolution_km': self.native_resolution_km
        }
        
        return output

class CAPEFeatureProcessor(nn.Module):
    """
    Processes CAPE features at native resolution to extract lightning-relevant patterns.
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 scales: List[int],
                 extract_patterns: bool,
                 extract_gradients: bool):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scales = scales
        self.extract_patterns = extract_patterns
        self.extract_gradients = extract_gradients
        
        # Base CAPE processing
        self.base_processor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale CAPE pattern extraction
        if extract_patterns:
            self.pattern_extractors = nn.ModuleList()
            for scale in scales:
                extractor = CAPEPatternExtractor(
                    in_channels=out_channels,
                    out_channels=out_channels // len(scales),
                    kernel_size=scale
                )
                self.pattern_extractors.append(extractor)
        
        # CAPE gradient extraction
        if extract_gradients:
            self.gradient_extractor = CAPEGradientExtractor(
                in_channels=out_channels,
                out_channels=out_channels // 4
            )
        
        # Feature combination
        total_channels = out_channels
        if extract_patterns:
            total_channels += out_channels
        if extract_gradients:
            total_channels += out_channels // 4
        
        self.feature_combiner = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, cape_features: torch.Tensor) -> torch.Tensor:
        """Process CAPE features."""
        # Base processing
        base_features = self.base_processor(cape_features)
        features_to_combine = [base_features]
        
        # Pattern extraction
        if self.extract_patterns:
            pattern_features = []
            for extractor in self.pattern_extractors:
                pattern_feat = extractor(base_features)
                pattern_features.append(pattern_feat)
            combined_patterns = torch.cat(pattern_features, dim=1)
            features_to_combine.append(combined_patterns)
        
        # Gradient extraction
        if self.extract_gradients:
            gradient_features = self.gradient_extractor(base_features)
            features_to_combine.append(gradient_features)
        
        # Combine all features
        combined = torch.cat(features_to_combine, dim=1)
        output = self.feature_combiner(combined)
        
        return output

class CAPEPatternExtractor(nn.Module):
    """Extracts CAPE patterns at specific scales relevant for convective initiation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        
        self.kernel_size = kernel_size
        
        # Pattern extraction convolution
        self.pattern_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Pattern enhancement
        self.pattern_enhancer = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, cape_features: torch.Tensor) -> torch.Tensor:
        """Extract CAPE patterns at this scale."""
        patterns = self.pattern_conv(cape_features)
        enhanced_patterns = self.pattern_enhancer(patterns)
        return enhanced_patterns

class CAPEGradientExtractor(nn.Module):
    """Extracts CAPE gradients that indicate convective boundaries."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # Gradient computation kernels
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
        
        # Gradient processing
        self.gradient_processor = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, kernel_size=3, padding=1, bias=False),  # 3x: grad_x, grad_y, grad_mag
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, cape_features: torch.Tensor) -> torch.Tensor:
        """Extract CAPE gradients."""
        # Apply Sobel operators to each channel
        grad_x_features = []
        grad_y_features = []
        grad_mag_features = []
        
        for i in range(cape_features.shape[1]):
            channel = cape_features[:, i:i+1]
            grad_x = F.conv2d(channel, self.sobel_x, padding=1)
            grad_y = F.conv2d(channel, self.sobel_y, padding=1)
            grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
            
            grad_x_features.append(grad_x)
            grad_y_features.append(grad_y)
            grad_mag_features.append(grad_mag)
        
        # Combine gradients
        all_grad_x = torch.cat(grad_x_features, dim=1)
        all_grad_y = torch.cat(grad_y_features, dim=1)
        all_grad_mag = torch.cat(grad_mag_features, dim=1)
        
        gradient_features = torch.cat([all_grad_x, all_grad_y, all_grad_mag], dim=1)
        
        # Process gradients
        output = self.gradient_processor(gradient_features)
        
        return output

class ERA5FeatureProcessor(nn.Module):
    """
    Processes ERA5 features at native resolution (future implementation).
    """
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 scales: List[int],
                 extract_patterns: bool,
                 extract_gradients: bool):
        super().__init__()
        
        # Placeholder for future ERA5 implementation
        self.processor = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        print("⚠️  ERA5 Feature Processor is placeholder - will be implemented when ERA5 data is available")
    
    def forward(self, era5_features: torch.Tensor) -> torch.Tensor:
        """Process ERA5 features (placeholder)."""
        return self.processor(era5_features)

class MeteorologicalFeatureFusion(nn.Module):
    """
    Fuses different meteorological features at native resolution.
    """
    
    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 cape_only_mode: bool):
        super().__init__()
        
        self.cape_only_mode = cape_only_mode
        
        if cape_only_mode:
            # Simple processing for CAPE-only mode
            self.fusion = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(output_channels, output_channels, kernel_size=1)
            )
        else:
            # More complex fusion for CAPE + ERA5
            self.fusion = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(output_channels, output_channels, kernel_size=1)
            )
    
    def forward(self, 
                cape_features: torch.Tensor,
                era5_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fuse meteorological features."""
        if self.cape_only_mode or era5_features is None:
            # CAPE-only processing
            return self.fusion(cape_features)
        else:
            # Combine CAPE and ERA5 features
            combined = torch.cat([cape_features, era5_features], dim=1)
            return self.fusion(combined)

class MeteorologicalPatternAnalyzer(nn.Module):
    """
    Analyzes meteorological patterns to identify convective-favorable conditions.
    """
    
    def __init__(self, in_channels: int, out_channels: int, scales: List[int]):
        super().__init__()
        
        self.scales = scales
        
        # Multi-scale pattern analysis
        self.scale_analyzers = nn.ModuleList()
        for scale in scales:
            analyzer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // len(scales), kernel_size=scale, padding=scale//2, bias=False),
                nn.BatchNorm2d(out_channels // len(scales)),
                nn.ReLU(inplace=True)
            )
            self.scale_analyzers.append(analyzer)
        
        # Pattern integration
        self.pattern_integrator = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
    
    def forward(self, meteorological_features: torch.Tensor) -> torch.Tensor:
        """Analyze meteorological patterns."""
        # Multi-scale analysis
        scale_patterns = []
        for analyzer in self.scale_analyzers:
            pattern = analyzer(meteorological_features)
            scale_patterns.append(pattern)
        
        # Combine patterns
        combined_patterns = torch.cat(scale_patterns, dim=1)
        
        # Integrate patterns
        integrated_patterns = self.pattern_integrator(combined_patterns)
        
        # Residual connection
        return integrated_patterns + meteorological_features

class DistributionPreparator(nn.Module):
    """
    Prepares meteorological features for spatial distribution to higher resolution.
    """
    
    def __init__(self, in_channels: int, out_channels: int, native_resolution_km: int):
        super().__init__()
        
        self.native_resolution_km = native_resolution_km
        
        # Distribution context encoder
        self.context_encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Distribution weight generator
        self.weight_generator = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, meteorological_features: torch.Tensor) -> torch.Tensor:
        """Prepare meteorological features for distribution."""
        # Encode distribution context
        context = self.context_encoder(meteorological_features)
        
        # Generate distribution weights
        weights = self.weight_generator(context)
        
        # Apply weights to features
        prepared_features = context * weights
        
        return prepared_features