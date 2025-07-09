"""
Physics-informed loss functions for lightning prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np

class LightningLoss(nn.Module):
    """
    Primary loss function for lightning prediction.
    Handles class imbalance common in lightning data.
    """
    
    def __init__(self,
                 loss_type: str = "focal",
                 pos_weight: Optional[float] = None,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 label_smoothing: float = 0.0):
        """
        Initialize lightning loss.
        
        Args:
            loss_type: Type of loss ('focal', 'bce', 'weighted_bce')
            pos_weight: Weight for positive class (lightning events)
            alpha: Alpha parameter for focal loss
            gamma: Gamma parameter for focal loss
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor(pos_weight))
        else:
            self.pos_weight = None
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute lightning prediction loss.
        
        Args:
            predictions: Model predictions (B, 1, H, W)
            targets: Ground truth lightning (B, H, W)
            
        Returns:
            Loss value
        """
        # Ensure same shape
        if predictions.dim() == 4 and predictions.shape[1] == 1:
            predictions = predictions.squeeze(1)
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        if self.loss_type == "focal":
            return self._focal_loss(predictions, targets)
        elif self.loss_type == "bce":
            return F.binary_cross_entropy(predictions, targets, reduction='mean')
        elif self.loss_type == "weighted_bce":
            return F.binary_cross_entropy(predictions, targets, 
                                        pos_weight=self.pos_weight, reduction='mean')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def _focal_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss for handling class imbalance."""
        
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy(predictions, targets, reduction='none')
        
        # Compute focal weight
        pt = torch.where(targets == 1, predictions, 1 - predictions)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        if self.alpha >= 0:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_loss = alpha_t * focal_weight * bce_loss
        else:
            focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()

class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss that incorporates atmospheric physics constraints.
    """
    
    def __init__(self,
                 charge_separation_weight: float = 0.05,
                 microphysics_weight: float = 0.05,
                 terrain_consistency_weight: float = 0.02,
                 adaptive_weights: bool = True):
        """
        Initialize physics-informed loss.
        
        Args:
            charge_separation_weight: Weight for charge separation constraint
            microphysics_weight: Weight for microphysics constraint
            terrain_consistency_weight: Weight for terrain consistency
            adaptive_weights: Whether to use adaptive weight adjustment
        """
        super().__init__()
        
        self.adaptive_weights = adaptive_weights
        
        # Initialize physics weights
        if adaptive_weights:
            self.charge_separation_weight = nn.Parameter(torch.tensor(charge_separation_weight))
            self.microphysics_weight = nn.Parameter(torch.tensor(microphysics_weight))
            self.terrain_consistency_weight = nn.Parameter(torch.tensor(terrain_consistency_weight))
        else:
            self.register_buffer('charge_separation_weight', torch.tensor(charge_separation_weight))
            self.register_buffer('microphysics_weight', torch.tensor(microphysics_weight))
            self.register_buffer('terrain_consistency_weight', torch.tensor(terrain_consistency_weight))
        
        # Physics constraint implementations
        self.charge_separation = ChargeSeparationConstraint()
        self.microphysics = MicrophysicsConstraint()
        self.terrain_consistency = TerrainConsistencyConstraint()
    
    def forward(self,
                predictions: torch.Tensor,
                cape_data: torch.Tensor,
                terrain_data: Optional[torch.Tensor] = None,
                temperature_data: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute physics-informed loss components.
        
        Args:
            predictions: Lightning predictions (B, 1, H, W)
            cape_data: CAPE data (B, 1, H_cape, W_cape)
            terrain_data: Terrain data (B, 1, H_terrain, W_terrain)
            temperature_data: Temperature data (future use)
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Charge separation constraint
        charge_loss = self.charge_separation(predictions, cape_data)
        losses['charge_separation'] = self.charge_separation_weight * charge_loss
        
        # Microphysics constraint
        if temperature_data is not None:
            micro_loss = self.microphysics(predictions, temperature_data)
            losses['microphysics'] = self.microphysics_weight * micro_loss
        else:
            losses['microphysics'] = torch.tensor(0.0, device=predictions.device)
        
        # Terrain consistency constraint
        if terrain_data is not None:
            terrain_loss = self.terrain_consistency(predictions, terrain_data)
            losses['terrain_consistency'] = self.terrain_consistency_weight * terrain_loss
        else:
            losses['terrain_consistency'] = torch.tensor(0.0, device=predictions.device)
        
        # Total physics loss
        losses['total_physics'] = sum(losses.values())
        
        return losses

class ChargeSeparationConstraint(nn.Module):
    """
    Constraint based on charge separation physics.
    Lightning probability should be low when CAPE is insufficient.
    """
    
    def __init__(self, cape_threshold: float = 1000.0):
        super().__init__()
        self.cape_threshold = cape_threshold
    
    def forward(self, predictions: torch.Tensor, cape_data: torch.Tensor) -> torch.Tensor:
        """
        Penalize high lightning probability when CAPE is low.
        
        Args:
            predictions: Lightning predictions (B, 1, H, W)
            cape_data: CAPE data (B, 1, H_cape, W_cape)
            
        Returns:
            Charge separation constraint loss
        """
        # Resize CAPE to match prediction resolution
        if cape_data.shape[-2:] != predictions.shape[-2:]:
            cape_resized = F.interpolate(
                cape_data, size=predictions.shape[-2:],
                mode='bilinear', align_corners=False
            )
        else:
            cape_resized = cape_data
        
        # Squeeze channel dimension if present
        if predictions.dim() == 4 and predictions.shape[1] == 1:
            predictions = predictions.squeeze(1)
        if cape_resized.dim() == 4 and cape_resized.shape[1] == 1:
            cape_resized = cape_resized.squeeze(1)
        
        # Normalize CAPE (typical range 0-5000 J/kg)
        cape_normalized = torch.clamp(cape_resized / 5000.0, 0, 1)
        
        # Physics constraint: Lightning probability should be proportional to CAPE
        # Penalize high lightning prediction when CAPE is low
        cape_mask = (cape_normalized < (self.cape_threshold / 5000.0)).float()
        constraint_loss = cape_mask * predictions
        
        return constraint_loss.mean()

class MicrophysicsConstraint(nn.Module):
    """
    Constraint based on mixed-phase microphysics.
    Lightning requires presence of ice and liquid water.
    """
    
    def __init__(self, temperature_threshold: float = 273.15):
        super().__init__()
        self.temperature_threshold = temperature_threshold
    
    def forward(self, predictions: torch.Tensor, temperature_data: torch.Tensor) -> torch.Tensor:
        """
        Penalize lightning prediction in purely warm or cold regions.
        
        Args:
            predictions: Lightning predictions (B, 1, H, W)
            temperature_data: Temperature data (B, levels, H, W)
            
        Returns:
            Microphysics constraint loss
        """
        # For now, implement a simple temperature gradient constraint
        # Future: Use actual ice/liquid water content from ERA5
        
        if temperature_data.dim() == 4:
            # Use surface temperature (first level)
            temp_surface = temperature_data[:, 0]
        else:
            temp_surface = temperature_data
        
        # Resize temperature to match predictions
        if temp_surface.shape[-2:] != predictions.shape[-2:]:
            temp_resized = F.interpolate(
                temp_surface.unsqueeze(1), size=predictions.shape[-2:],
                mode='bilinear', align_corners=False
            ).squeeze(1)
        else:
            temp_resized = temp_surface
        
        # Squeeze predictions if needed
        if predictions.dim() == 4 and predictions.shape[1] == 1:
            predictions = predictions.squeeze(1)
        
        # Physics constraint: Lightning less likely in extreme temperatures
        temp_normalized = (temp_resized - 273.15) / 40.0  # Normalize around 0C, 40K range
        
        # Penalize lightning in very cold (< -20C) or very hot (> 40C) conditions
        extreme_temp_mask = ((temp_normalized < -0.5) | (temp_normalized > 1.0)).float()
        constraint_loss = extreme_temp_mask * predictions
        
        return constraint_loss.mean()

class TerrainConsistencyConstraint(nn.Module):
    """
    Constraint for terrain-lightning relationships.
    Ensures physically reasonable terrain effects.
    """
    
    def __init__(self, elevation_scale: float = 1000.0):
        super().__init__()
        self.elevation_scale = elevation_scale
    
    def forward(self, predictions: torch.Tensor, terrain_data: torch.Tensor) -> torch.Tensor:
        """
        Enforce terrain-lightning consistency.
        
        Args:
            predictions: Lightning predictions (B, 1, H, W)
            terrain_data: Terrain elevation (B, 1, H, W)
            
        Returns:
            Terrain consistency constraint loss
        """
        # Resize terrain to match predictions
        if terrain_data.shape[-2:] != predictions.shape[-2:]:
            terrain_resized = F.interpolate(
                terrain_data, size=predictions.shape[-2:],
                mode='bilinear', align_corners=False
            )
        else:
            terrain_resized = terrain_data
        
        # Squeeze dimensions
        if predictions.dim() == 4 and predictions.shape[1] == 1:
            predictions = predictions.squeeze(1)
        if terrain_resized.dim() == 4 and terrain_resized.shape[1] == 1:
            terrain_resized = terrain_resized.squeeze(1)
        
        # Compute terrain gradients (orographic lifting effect)
        terrain_grad_x = torch.gradient(terrain_resized, dim=-1)[0]
        terrain_grad_y = torch.gradient(terrain_resized, dim=-2)[0]
        terrain_grad_magnitude = torch.sqrt(terrain_grad_x**2 + terrain_grad_y**2 + 1e-6)
        
        # Normalize gradient
        terrain_grad_norm = terrain_grad_magnitude / (terrain_grad_magnitude.max() + 1e-6)
        
        # Physics constraint: Lightning more likely with moderate terrain gradients
        # Penalize very smooth or very rough terrain with high lightning probability
        smooth_penalty = (terrain_grad_norm < 0.1).float() * predictions * 0.5
        rough_penalty = (terrain_grad_norm > 0.8).float() * predictions * 0.3
        
        constraint_loss = smooth_penalty + rough_penalty
        
        return constraint_loss.mean()

class AdaptiveLossWeighting(nn.Module):
    """
    Adaptive loss weighting that adjusts physics constraint weights during training.
    """
    
    def __init__(self, 
                 initial_main_weight: float = 1.0,
                 initial_physics_weight: float = 0.1,
                 adaptation_rate: float = 0.01):
        super().__init__()
        
        self.adaptation_rate = adaptation_rate
        
        # Learnable weights
        self.main_weight = nn.Parameter(torch.tensor(initial_main_weight))
        self.physics_weight = nn.Parameter(torch.tensor(initial_physics_weight))
        
        # Loss history for adaptation
        self.register_buffer('main_loss_history', torch.zeros(100))
        self.register_buffer('physics_loss_history', torch.zeros(100))
        self.register_buffer('step_count', torch.tensor(0))
    
    def forward(self, main_loss: torch.Tensor, physics_loss: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute adaptive weighted loss.
        
        Args:
            main_loss: Primary lightning prediction loss
            physics_loss: Physics constraint loss
            
        Returns:
            Tuple of (total_loss, loss_info)
        """
        # Update loss history
        step = self.step_count % 100
        self.main_loss_history[step] = main_loss.detach()
        self.physics_loss_history[step] = physics_loss.detach()
        self.step_count += 1
        
        # Compute weighted loss
        main_weighted = torch.sigmoid(self.main_weight) * main_loss
        physics_weighted = torch.sigmoid(self.physics_weight) * physics_loss
        
        total_loss = main_weighted + physics_weighted
        
        # Adaptation: Increase physics weight if main loss is decreasing faster
        if self.step_count > 50:  # Wait for some history
            main_trend = self.main_loss_history[-10:].mean() - self.main_loss_history[-50:-40].mean()
            physics_trend = self.physics_loss_history[-10:].mean() - self.physics_loss_history[-50:-40].mean()
            
            # If main loss decreases much faster than physics loss, increase physics weight
            if main_trend < physics_trend * 0.5:
                self.physics_weight.data += self.adaptation_rate
            # If physics loss decreases much faster, increase main weight
            elif physics_trend < main_trend * 0.5:
                self.main_weight.data += self.adaptation_rate
        
        loss_info = {
            'main_weighted': main_weighted,
            'physics_weighted': physics_weighted,
            'main_weight': torch.sigmoid(self.main_weight),
            'physics_weight': torch.sigmoid(self.physics_weight)
        }
        
        return total_loss, loss_info

class CompositeLoss(nn.Module):
    """
    Complete loss function combining prediction loss and physics constraints.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Primary prediction loss
        self.prediction_loss = LightningLoss(
            loss_type=config.get('primary', 'focal'),
            alpha=config.get('focal_alpha', 0.25),
            gamma=config.get('focal_gamma', 2.0),
            label_smoothing=config.get('label_smoothing', 0.0)
        )
        
        # Physics-informed loss
        self.physics_loss = PhysicsInformedLoss(
            charge_separation_weight=config.get('charge_separation_weight', 0.05),
            microphysics_weight=config.get('microphysics_weight', 0.05),
            terrain_consistency_weight=config.get('terrain_consistency_weight', 0.02),
            adaptive_weights=config.get('adaptive_weights', True)
        )
        
        # Adaptive weighting
        if config.get('adaptive_weighting', True):
            self.adaptive_weighting = AdaptiveLossWeighting(
                initial_physics_weight=config.get('physics_weight', 0.1)
            )
        else:
            self.adaptive_weighting = None
            self.physics_weight = config.get('physics_weight', 0.1)
    
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                cape_data: torch.Tensor,
                terrain_data: Optional[torch.Tensor] = None,
                temperature_data: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute complete loss with physics constraints.
        
        Returns:
            Dictionary containing all loss components
        """
        # Primary prediction loss
        main_loss = self.prediction_loss(predictions, targets)
        
        # Physics constraint losses
        physics_losses = self.physics_loss(predictions, cape_data, terrain_data, temperature_data)
        total_physics_loss = physics_losses['total_physics']
        
        # Combine losses
        if self.adaptive_weighting is not None:
            total_loss, weighting_info = self.adaptive_weighting(main_loss, total_physics_loss)
        else:
            total_loss = main_loss + self.physics_weight * total_physics_loss
            weighting_info = {'main_weight': 1.0, 'physics_weight': self.physics_weight}
        
        # Prepare output
        loss_dict = {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'physics_loss': total_physics_loss,
            **physics_losses,
            **weighting_info
        }
        
        return loss_dict

def create_loss_function(config: Dict) -> nn.Module:
    """
    Factory function to create loss function from configuration.
    
    Args:
        config: Loss function configuration
        
    Returns:
        Configured loss function
    """
    loss_type = config.get('type', 'composite')
    
    if loss_type == 'simple':
        return LightningLoss(
            loss_type=config.get('primary', 'focal'),
            alpha=config.get('focal_alpha', 0.25),
            gamma=config.get('focal_gamma', 2.0)
        )
    elif loss_type == 'physics':
        return PhysicsInformedLoss(
            charge_separation_weight=config.get('charge_separation_weight', 0.05),
            microphysics_weight=config.get('microphysics_weight', 0.05),
            terrain_consistency_weight=config.get('terrain_consistency_weight', 0.02)
        )
    else:  # composite
        return CompositeLoss(config)
