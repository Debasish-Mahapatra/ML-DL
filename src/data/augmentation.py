"""
Data augmentation for spatial and meteorological data.
"""

import torch
import numpy as np
from typing import Tuple, List, Optional, Union
import random
from torchvision.transforms import functional as TF
import logging

logger = logging.getLogger(__name__)

class SpatialAugmentation:
    """
    Spatial augmentation for meteorological and terrain data.
    Applies same transformations to all spatial inputs to maintain consistency.
    """
    
    def __init__(self,
                 rotation_range: List[float] = [-30, 30],
                 flip_horizontal: float = 0.5,
                 flip_vertical: float = 0.5,
                 scale_range: Optional[List[float]] = None,
                 translation_range: Optional[List[float]] = None):
        """
        Initialize spatial augmentation.
        
        Args:
            rotation_range: Range of rotation angles in degrees [min, max]
            flip_horizontal: Probability of horizontal flip
            flip_vertical: Probability of vertical flip
            scale_range: Range of scaling factors [min, max] (optional)
            translation_range: Range of translation as fraction of image size [min, max] (optional)
        """
        self.rotation_range = rotation_range
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.scale_range = scale_range
        self.translation_range = translation_range
    
    def __call__(self, 
                 cape_tensor: torch.Tensor,
                 lightning_tensor: torch.Tensor,
                 terrain_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply spatial augmentation to all inputs.
        
        Args:
            cape_tensor: CAPE data (channels, height, width)
            lightning_tensor: Lightning data (height, width)
            terrain_tensor: Terrain data (channels, height, width)
            
        Returns:
            Tuple of augmented tensors
        """
        # Generate random parameters for this sample
        angle = random.uniform(self.rotation_range[0], self.rotation_range[1]) if self.rotation_range else 0
        do_hflip = random.random() < self.flip_horizontal
        do_vflip = random.random() < self.flip_vertical
        
        # Apply rotation
        if angle != 0:
            cape_tensor = TF.rotate(cape_tensor, angle, interpolation=TF.InterpolationMode.BILINEAR)
            lightning_tensor = TF.rotate(lightning_tensor.unsqueeze(0), angle, 
                                       interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
            terrain_tensor = TF.rotate(terrain_tensor, angle, interpolation=TF.InterpolationMode.BILINEAR)
        
        # Apply horizontal flip
        if do_hflip:
            cape_tensor = TF.hflip(cape_tensor)
            lightning_tensor = TF.hflip(lightning_tensor.unsqueeze(0)).squeeze(0)
            terrain_tensor = TF.hflip(terrain_tensor)
        
        # Apply vertical flip
        if do_vflip:
            cape_tensor = TF.vflip(cape_tensor)
            lightning_tensor = TF.vflip(lightning_tensor.unsqueeze(0)).squeeze(0)
            terrain_tensor = TF.vflip(terrain_tensor)
        
        return cape_tensor, lightning_tensor, terrain_tensor

class MeteorologicalAugmentation:
    """
    Augmentation specific to meteorological variables.
    Applies realistic perturbations that maintain physical consistency.
    """
    
    def __init__(self,
                 noise_std: float = 0.01,
                 scale_factor: List[float] = [0.95, 1.05],
                 temperature_offset: Optional[List[float]] = None,
                 apply_probability: float = 0.5):
        """
        Initialize meteorological augmentation.
        
        Args:
            noise_std: Standard deviation of Gaussian noise
            scale_factor: Range of multiplicative scaling factors
            temperature_offset: Range of temperature offset in Kelvin (for future use)
            apply_probability: Probability of applying augmentation
        """
        self.noise_std = noise_std
        self.scale_factor = scale_factor
        self.temperature_offset = temperature_offset
        self.apply_probability = apply_probability
    
    def __call__(self, meteorological_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply meteorological augmentation.
        
        Args:
            meteorological_tensor: Meteorological data tensor
            
        Returns:
            Augmented meteorological tensor
        """
        if random.random() > self.apply_probability:
            return meteorological_tensor
        
        augmented_tensor = meteorological_tensor.clone()
        
        # Add Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(augmented_tensor) * self.noise_std
            augmented_tensor = augmented_tensor + noise
        
        # Apply scaling
        if self.scale_factor and len(self.scale_factor) == 2:
            scale = random.uniform(*self.scale_factor)
            augmented_tensor = augmented_tensor * scale
        
        # Ensure non-negative values for CAPE
        augmented_tensor = torch.clamp(augmented_tensor, min=0)
        
        return augmented_tensor

class TemporalAugmentation:
    """
    Temporal augmentation for future sequence-based models.
    Currently placeholder for future implementation.
    """
    
    def __init__(self, 
                 time_shift_range: Optional[List[int]] = None,
                 temporal_dropout: float = 0.0):
        """
        Initialize temporal augmentation.
        
        Args:
            time_shift_range: Range of time shifts in hours
            temporal_dropout: Probability of dropping temporal steps
        """
        self.time_shift_range = time_shift_range
        self.temporal_dropout = temporal_dropout
    
    def __call__(self, temporal_sequence: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal augmentation.
        
        Args:
            temporal_sequence: Temporal sequence (time, channels, height, width)
            
        Returns:
            Augmented temporal sequence
        """
        # Future implementation for temporal sequences
        return temporal_sequence

class PhysicsAwareAugmentation:
    """
    Physics-aware augmentation that maintains physical relationships.
    Ensures augmentations don't violate atmospheric physics.
    """
    
    def __init__(self,
                 max_cape_change: float = 500.0,  # J/kg
                 maintain_gradients: bool = True,
                 elevation_consistency: bool = True):
        """
        Initialize physics-aware augmentation.
        
        Args:
            max_cape_change: Maximum allowed CAPE change in J/kg
            maintain_gradients: Whether to maintain spatial gradients
            elevation_consistency: Whether to maintain elevation-weather relationships
        """
        self.max_cape_change = max_cape_change
        self.maintain_gradients = maintain_gradients
        self.elevation_consistency = elevation_consistency
    
    def __call__(self,
                 cape_tensor: torch.Tensor,
                 terrain_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply physics-aware augmentation.
        
        Args:
            cape_tensor: CAPE data
            terrain_tensor: Terrain data
            
        Returns:
            Tuple of physics-consistent augmented tensors
        """
        # Physics-based perturbations
        augmented_cape = cape_tensor.clone()
        
        if self.elevation_consistency:
            # Modify CAPE based on elevation (higher elevation -> lower CAPE tendency)
            elevation = terrain_tensor[0]  # Assuming first channel is elevation
            elevation_factor = 1.0 - 0.1 * (elevation / elevation.max())
            augmented_cape = augmented_cape * elevation_factor.unsqueeze(0)
        
        # Ensure changes are within physical limits
        cape_change = torch.abs(augmented_cape - cape_tensor)
        mask = cape_change > self.max_cape_change
        augmented_cape[mask] = cape_tensor[mask]
        
        return augmented_cape, terrain_tensor

class AugmentationPipeline:
    """
    Combined augmentation pipeline that applies multiple augmentations in sequence.
    """
    
    def __init__(self,
                 spatial_aug: Optional[SpatialAugmentation] = None,
                 meteorological_aug: Optional[MeteorologicalAugmentation] = None,
                 physics_aug: Optional[PhysicsAwareAugmentation] = None,
                 probability: float = 0.8):
        """
        Initialize augmentation pipeline.
        
        Args:
            spatial_aug: Spatial augmentation instance
            meteorological_aug: Meteorological augmentation instance
            physics_aug: Physics-aware augmentation instance
            probability: Overall probability of applying augmentations
        """
        self.spatial_aug = spatial_aug
        self.meteorological_aug = meteorological_aug
        self.physics_aug = physics_aug
        self.probability = probability
    
    def __call__(self,
                 cape_tensor: torch.Tensor,
                 lightning_tensor: torch.Tensor,
                 terrain_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply full augmentation pipeline.
        
        Args:
            cape_tensor: CAPE data
            lightning_tensor: Lightning data
            terrain_tensor: Terrain data
            
        Returns:
            Tuple of augmented tensors
        """
        if random.random() > self.probability:
            return cape_tensor, lightning_tensor, terrain_tensor
        
        # Apply spatial augmentation first (affects all inputs)
        if self.spatial_aug:
            cape_tensor, lightning_tensor, terrain_tensor = self.spatial_aug(
                cape_tensor, lightning_tensor, terrain_tensor
            )
        
        # Apply meteorological augmentation
        if self.meteorological_aug:
            cape_tensor = self.meteorological_aug(cape_tensor)
        
        # Apply physics-aware augmentation
        if self.physics_aug:
            cape_tensor, terrain_tensor = self.physics_aug(cape_tensor, terrain_tensor)
        
        return cape_tensor, lightning_tensor, terrain_tensor

def create_augmentation_from_config(config: dict) -> Optional[AugmentationPipeline]:
    """
    Create augmentation pipeline from configuration.
    
    Args:
        config: Augmentation configuration dictionary
        
    Returns:
        Augmentation pipeline or None if disabled
    """
    if not config or not config.get('enabled', True):
        return None
    
    # Create individual augmentations
    spatial_aug = None
    if config.get('spatial'):
        spatial_config = config['spatial']
        spatial_aug = SpatialAugmentation(
            rotation_range=spatial_config.get('rotation', {}).get('range', [-30, 30]) if spatial_config.get('rotation', {}).get('enabled', False) else None,
            flip_horizontal=spatial_config.get('flip_horizontal', {}).get('probability', 0.5) if spatial_config.get('flip_horizontal', {}).get('enabled', False) else 0,
            flip_vertical=spatial_config.get('flip_vertical', {}).get('probability', 0.5) if spatial_config.get('flip_vertical', {}).get('enabled', False) else 0
        )
    
    meteorological_aug = None
    if config.get('meteorological'):
        met_config = config['meteorological']
        meteorological_aug = MeteorologicalAugmentation(
            noise_std=met_config.get('noise_std', 0.01),
            scale_factor=met_config.get('scale_factor', [0.95, 1.05])
        )
    
    physics_aug = None
    if config.get('physics_aware'):
        physics_config = config['physics_aware']
        physics_aug = PhysicsAwareAugmentation(
            max_cape_change=physics_config.get('max_cape_change', 500.0),
            maintain_gradients=physics_config.get('maintain_gradients', True),
            elevation_consistency=physics_config.get('elevation_consistency', True)
        )
    
    # Create pipeline
    return AugmentationPipeline(
        spatial_aug=spatial_aug,
        meteorological_aug=meteorological_aug,
        physics_aug=physics_aug,
        probability=config.get('probability', 0.8)
    )
