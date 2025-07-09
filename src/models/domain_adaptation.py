"""
Domain adaptation module for generalizing across different geographical regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class DomainAdapter(nn.Module):
    """
    Domain adaptation module that helps the model generalize from Odisha to other regions
    like Bihar without retraining the entire model.
    """
    
    def __init__(self,
                 terrain_features: int = 64,
                 meteorological_features: int = 256,
                 terrain_adaptation_dim: int = 64,
                 meteorological_adaptation_dim: int = 32,
                 dropout: float = 0.1):
        """
        Initialize domain adapter.
        
        Args:
            terrain_features: Number of terrain feature channels
            meteorological_features: Number of meteorological feature channels
            terrain_adaptation_dim: Dimension for terrain adaptation
            meteorological_adaptation_dim: Dimension for meteorological adaptation
            dropout: Dropout probability
        """
        super().__init__()
        
        self.terrain_features = terrain_features
        self.meteorological_features = meteorological_features
        
        # Terrain-specific adaptation (80% of adaptation capacity)
        self.terrain_adapter = TerrainDomainAdapter(
            terrain_features, terrain_adaptation_dim, dropout
        )
        
        # Meteorological pattern adaptation (20% of adaptation capacity)
        self.meteorological_adapter = MeteorologicalDomainAdapter(
            meteorological_features, meteorological_adaptation_dim, dropout
        )
        
        # Feature integration
        self.feature_integrator = nn.Sequential(
            nn.Conv2d(
                meteorological_features + terrain_adaptation_dim + meteorological_adaptation_dim,
                meteorological_features, 3, padding=1, bias=False
            ),
            nn.BatchNorm2d(meteorological_features),
            nn.ReLU(inplace=True)
        )
        
        # Gating mechanism to control adaptation strength
        self.adaptation_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(meteorological_features, meteorological_features // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(meteorological_features // 8, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self,
                fused_features: torch.Tensor,
                terrain_features: torch.Tensor,
                meteorological_features: torch.Tensor) -> torch.Tensor:
        """
        Apply domain adaptation.
        
        Args:
            fused_features: Main fused features from multi-scale fusion
            terrain_features: Terrain features for adaptation
            meteorological_features: Meteorological features for adaptation
            
        Returns:
            Domain-adapted features
        """
        
        # Terrain-specific adaptation
        terrain_adaptation = self.terrain_adapter(terrain_features)
        
        # Meteorological pattern adaptation
        met_adaptation = self.meteorological_adapter(meteorological_features)
        
        # Resize adaptation features to match fused features
        target_size = fused_features.shape[-2:]
        
        if terrain_adaptation.shape[-2:] != target_size:
            terrain_adaptation = F.interpolate(
                terrain_adaptation, size=target_size,
                mode='bilinear', align_corners=False
            )
        
        if met_adaptation.shape[-2:] != target_size:
            met_adaptation = F.interpolate(
                met_adaptation, size=target_size,
                mode='bilinear', align_corners=False
            )
        
        # Combine all features
        combined_features = torch.cat([
            fused_features, terrain_adaptation, met_adaptation
        ], dim=1)
        
        # Integrate features
        integrated = self.feature_integrator(combined_features)
        
        # Compute adaptation gate
        gate = self.adaptation_gate(integrated)
        
        # Apply gated adaptation
        adapted_features = fused_features * (1 - gate) + integrated * gate
        
        return adapted_features

class TerrainDomainAdapter(nn.Module):
    """Adapts to terrain-specific characteristics of different regions."""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        
        # Terrain characteristic extractors
        self.elevation_processor = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 4, 3, padding=1),
            nn.BatchNorm2d(output_dim // 4),
            nn.ReLU(inplace=True)
        )
        
        self.slope_processor = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 4, 3, padding=1),
            nn.BatchNorm2d(output_dim // 4),
            nn.ReLU(inplace=True)
        )
        
        self.roughness_processor = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 4, 3, padding=1),
            nn.BatchNorm2d(output_dim // 4),
            nn.ReLU(inplace=True)
        )
        
        self.aspect_processor = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 4, 3, padding=1),
            nn.BatchNorm2d(output_dim // 4),
            nn.ReLU(inplace=True)
        )
        
        # Feature fusion
        self.terrain_fusion = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, 1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
    def forward(self, terrain_features: torch.Tensor) -> torch.Tensor:
        """Process terrain features for domain adaptation."""
        
        # Extract different terrain characteristics
        elevation_feat = self.elevation_processor(terrain_features)
        slope_feat = self.slope_processor(terrain_features)
        roughness_feat = self.roughness_processor(terrain_features)
        aspect_feat = self.aspect_processor(terrain_features)
        
        # Combine terrain characteristics
        combined = torch.cat([elevation_feat, slope_feat, roughness_feat, aspect_feat], dim=1)
        
        # Fuse terrain features
        adapted_terrain = self.terrain_fusion(combined)
        
        return adapted_terrain

class MeteorologicalDomainAdapter(nn.Module):
    """Adapts to meteorological pattern differences between regions."""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        
        # Meteorological pattern processors
        self.seasonal_processor = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 2, 3, padding=1),
            nn.BatchNorm2d(output_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        self.regional_processor = nn.Sequential(
            nn.Conv2d(input_dim, output_dim // 2, 5, padding=2),
            nn.BatchNorm2d(output_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        # Adaptive pattern fusion
        self.pattern_fusion = nn.Sequential(
            nn.Conv2d(output_dim, output_dim, 1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        # Regional climate adaptation
        self.climate_adapter = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_dim, output_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, meteorological_features: torch.Tensor) -> torch.Tensor:
        """Process meteorological features for domain adaptation."""
        
        # Extract seasonal patterns
        seasonal_feat = self.seasonal_processor(meteorological_features)
        
        # Extract regional patterns
        regional_feat = self.regional_processor(meteorological_features)
        
        # Combine patterns
        combined_patterns = torch.cat([seasonal_feat, regional_feat], dim=1)
        
        # Fuse patterns
        fused_patterns = self.pattern_fusion(combined_patterns)
        
        # Apply climate adaptation
        climate_weights = self.climate_adapter(meteorological_features)
        adapted_met = fused_patterns * climate_weights
        
        return adapted_met

class AdaptationController(nn.Module):
    """
    Controls the strength of domain adaptation based on feature similarity
    between source (Odisha) and target (e.g., Bihar) domains.
    """
    
    def __init__(self, feature_dim: int):
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Domain similarity estimator
        self.similarity_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feature_dim, feature_dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Adaptation strength controller
        self.strength_controller = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptation strength based on domain similarity.
        
        Args:
            source_features: Features from source domain (Odisha)
            target_features: Features from target domain (Bihar)
            
        Returns:
            Adaptation strength weights
        """
        # Compute feature difference
        feature_diff = torch.abs(source_features - target_features)
        
        # Estimate domain similarity (high similarity = low adaptation needed)
        similarity = self.similarity_estimator(feature_diff)
        
        # Compute adaptation strength (inverse of similarity)
        adaptation_strength = (1.0 - similarity) * torch.sigmoid(self.strength_controller)
        
        return adaptation_strength

class FastDomainAdapter(nn.Module):
    """
    Lightweight domain adapter for quick fine-tuning on new regions.
    Contains minimal parameters for fast adaptation.
    """
    
    def __init__(self, feature_dim: int, adaptation_dim: int = 16):
        super().__init__()
        
        # Minimal adaptation parameters
        self.adaptation_weights = nn.Parameter(torch.ones(1, feature_dim, 1, 1))
        self.adaptation_bias = nn.Parameter(torch.zeros(1, feature_dim, 1, 1))
        
        # Quick terrain adjustment
        self.terrain_adjustment = nn.Sequential(
            nn.Conv2d(feature_dim, adaptation_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(adaptation_dim, feature_dim, 1),
            nn.Tanh()  # Bounded adjustment
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply fast domain adaptation."""
        
        # Scale and shift features
        adjusted = features * self.adaptation_weights + self.adaptation_bias
        
        # Apply terrain-specific adjustment
        terrain_adj = self.terrain_adjustment(features)
        final_features = adjusted + 0.1 * terrain_adj  # Small adjustment
        
        return final_features

def create_domain_adapter(config: dict, adaptation_type: str = "full") -> nn.Module:
    """
    Factory function to create different types of domain adapters.
    
    Args:
        config: Adapter configuration
        adaptation_type: Type of adapter ("full", "fast", "minimal")
        
    Returns:
        Domain adapter module
    """
    if adaptation_type == "full":
        return DomainAdapter(
            terrain_features=config.get('terrain_features', 64),
            meteorological_features=config.get('meteorological_features', 256),
            terrain_adaptation_dim=config.get('terrain_adaptation_dim', 64),
            meteorological_adaptation_dim=config.get('meteorological_adaptation_dim', 32),
            dropout=config.get('dropout', 0.1)
        )
    elif adaptation_type == "fast":
        return FastDomainAdapter(
            feature_dim=config.get('meteorological_features', 256),
            adaptation_dim=config.get('adaptation_dim', 16)
        )
    else:  # minimal
        # Return identity for no adaptation
        return nn.Identity()