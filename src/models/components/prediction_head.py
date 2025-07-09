"""
Prediction head for lightning occurrence/probability prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

class PredictionHead(nn.Module):
    """
    Lightning prediction head that outputs probability of lightning occurrence.
    """
    
    def __init__(self,
                 input_channels: int = 256,
                 hidden_dim: int = 128,
                 output_dim: int = 1,
                 activation: str = "sigmoid",
                 use_uncertainty: bool = False,
                 dropout: float = 0.1):
        """
        Initialize prediction head.
        
        Args:
            input_channels: Number of input feature channels
            hidden_dim: Hidden dimension for prediction layers
            output_dim: Output dimension (1 for binary lightning prediction)
            activation: Final activation function
            use_uncertainty: Whether to predict uncertainty estimates
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_uncertainty = use_uncertainty
        
        # Feature processing layers
        self.feature_processor = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            
            nn.Conv2d(hidden_dim, hidden_dim // 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        # Lightning prediction branch
        self.lightning_predictor = nn.Conv2d(hidden_dim // 2, output_dim, 1)
        
        # Uncertainty prediction branch (if enabled)
        if use_uncertainty:
            self.uncertainty_predictor = nn.Conv2d(hidden_dim // 2, output_dim, 1)
        
        # Final activation
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "softmax":
            self.activation = nn.Softmax(dim=1)
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with appropriate strategy for lightning prediction."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize final prediction layer with small weights for stable training
        nn.init.xavier_normal_(self.lightning_predictor.weight, gain=0.1)
        if self.lightning_predictor.bias is not None:
            nn.init.constant_(self.lightning_predictor.bias, -2.0)  # Bias toward no lightning
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through prediction head.
        
        Args:
            x: Input features (batch_size, input_channels, height, width)
            
        Returns:
            Lightning predictions (batch_size, output_dim, height, width)
            If use_uncertainty=True, returns tuple (predictions, uncertainty)
        """
        # Process features
        processed_features = self.feature_processor(x)
        
        # Predict lightning
        lightning_logits = self.lightning_predictor(processed_features)
        lightning_pred = self.activation(lightning_logits)
        
        if self.use_uncertainty:
            # Predict uncertainty
            uncertainty_logits = self.uncertainty_predictor(processed_features)
            uncertainty = torch.sigmoid(uncertainty_logits)  # Uncertainty always between 0 and 1
            
            return lightning_pred, uncertainty
        
        return lightning_pred

class MultiScalePredictionHead(nn.Module):
    """
    Multi-scale prediction head that makes predictions at multiple resolutions
    and combines them for final output.
    """
    
    def __init__(self,
                 input_channels: int = 256,
                 scales: List[int] = [1, 2, 4],
                 hidden_dim: int = 128,
                 output_dim: int = 1):
        """
        Initialize multi-scale prediction head.
        
        Args:
            input_channels: Number of input feature channels
            scales: List of scales for multi-scale prediction
            hidden_dim: Hidden dimension
            output_dim: Output dimension
        """
        super().__init__()
        
        self.scales = scales
        self.num_scales = len(scales)
        
        # Multi-scale feature extractors
        self.scale_extractors = nn.ModuleList()
        for scale in scales:
            if scale == 1:
                extractor = nn.Identity()
            else:
                extractor = nn.Sequential(
                    nn.AvgPool2d(scale, scale),
                    nn.Conv2d(input_channels, input_channels, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False)
                )
            self.scale_extractors.append(extractor)
        
        # Scale-specific prediction heads
        self.scale_predictors = nn.ModuleList([
            PredictionHead(input_channels, hidden_dim, output_dim)
            for _ in scales
        ])
        
        # Scale fusion network
        self.scale_fusion = nn.Sequential(
            nn.Conv2d(self.num_scales * output_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, output_dim, 1),
            nn.Sigmoid()
        )
        
        # Attention weights for different scales
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, self.num_scales, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-scale prediction head."""
        
        # Extract features at different scales
        scale_features = []
        for extractor in self.scale_extractors:
            scale_feat = extractor(x)
            scale_features.append(scale_feat)
        
        # Make predictions at each scale
        scale_predictions = []
        for feat, predictor in zip(scale_features, self.scale_predictors):
            pred = predictor(feat)
            scale_predictions.append(pred)
        
        # Compute attention weights for scales
        scale_weights = self.scale_attention(x)
        
        # Weighted combination of scale predictions
        weighted_preds = sum(w.unsqueeze(2).unsqueeze(3) * pred 
                            for w, pred in zip(scale_weights.split(1, dim=1), scale_predictions))
        
        # Concatenate all predictions for fusion
        all_preds = torch.cat(scale_predictions, dim=1)
        
        # Final fusion
        fused_prediction = self.scale_fusion(all_preds)
        
        # Combine weighted and fused predictions
        final_prediction = 0.7 * weighted_preds + 0.3 * fused_prediction
        
        return final_prediction

class PhysicsInformedHead(PredictionHead):
    """
    Physics-informed prediction head that incorporates atmospheric physics
    constraints during prediction.
    """
    
    def __init__(self, *args, physics_channels: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.physics_channels = physics_channels
        
        # Physics constraint processor
        self.physics_processor = PhysicsConstraintProcessor(
            self.input_channels, physics_channels
        )
        
        # Enhanced feature processor that includes physics
        enhanced_input_channels = self.input_channels + physics_channels
        self.enhanced_processor = nn.Sequential(
            nn.Conv2d(enhanced_input_channels, self.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim // 2, 1)
        )
        
        # Override lightning predictor to use enhanced features
        self.lightning_predictor = nn.Conv2d(self.hidden_dim // 2, self.output_dim, 1)
    
    def forward(self, x: torch.Tensor, cape_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with physics constraints."""
        
        # Extract physics-informed features
        physics_features = self.physics_processor(x, cape_data)
        
        # Combine original features with physics features
        enhanced_features = torch.cat([x, physics_features], dim=1)
        
        # Process enhanced features
        processed = self.enhanced_processor(enhanced_features)
        
        # Make prediction
        lightning_logits = self.lightning_predictor(processed)
        lightning_pred = self.activation(lightning_logits)
        
        return lightning_pred

class PhysicsConstraintProcessor(nn.Module):
    """Process physics constraints for lightning prediction."""
    
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        
        # CAPE-based constraint processor
        self.cape_processor = nn.Sequential(
            nn.Conv2d(1, output_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Feature-based constraint processor
        self.feature_processor = nn.Sequential(
            nn.Conv2d(input_channels, output_channels // 2, 1),
            nn.ReLU(inplace=True)
        )
        
        # Combined constraint processor
        self.combined_processor = nn.Sequential(
            nn.Conv2d(output_channels // 4 + output_channels // 2, output_channels, 3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features: torch.Tensor, cape_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process physics constraints."""
        
        # Process input features
        feature_constraints = self.feature_processor(features)
        
        if cape_data is not None:
            # Process CAPE data
            cape_constraints = self.cape_processor(cape_data)
            
            # Resize CAPE constraints to match feature size if needed
            if cape_constraints.shape[-2:] != feature_constraints.shape[-2:]:
                cape_constraints = F.interpolate(
                    cape_constraints, size=feature_constraints.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            
            # Combine constraints
            combined_input = torch.cat([feature_constraints, cape_constraints], dim=1)
        else:
            # Use only feature constraints
            dummy_cape = torch.zeros_like(feature_constraints[:, :feature_constraints.shape[1]//2])
            combined_input = torch.cat([feature_constraints, dummy_cape], dim=1)
        
        # Process combined constraints
        physics_features = self.combined_processor(combined_input)
        
        return physics_features

class EnsemblePredictionHead(nn.Module):
    """
    Ensemble prediction head that combines multiple prediction heads
    for improved reliability and uncertainty estimation.
    """
    
    def __init__(self,
                 input_channels: int = 256,
                 num_heads: int = 3,
                 hidden_dim: int = 128,
                 output_dim: int = 1):
        super().__init__()
        
        self.num_heads = num_heads
        
        # Multiple prediction heads with slight variations
        self.prediction_heads = nn.ModuleList([
            PredictionHead(
                input_channels, 
                hidden_dim + i * 16,  # Slight variation in hidden dim
                output_dim,
                dropout=0.1 + i * 0.05  # Slight variation in dropout
            ) for i in range(num_heads)
        ])
        
        # Ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ensemble heads.
        
        Returns:
            Tuple of (ensemble_prediction, prediction_uncertainty)
        """
        # Get predictions from all heads
        predictions = []
        for head in self.prediction_heads:
            pred = head(x)
            predictions.append(pred)
        
        # Stack predictions
        all_preds = torch.stack(predictions, dim=0)  # (num_heads, batch, channels, h, w)
        
        # Compute ensemble weights (softmax normalized)
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        # Weighted ensemble prediction
        ensemble_pred = torch.sum(weights.view(-1, 1, 1, 1, 1) * all_preds, dim=0)
        
        # Compute uncertainty as prediction variance
        pred_variance = torch.var(all_preds, dim=0)
        uncertainty = torch.sqrt(pred_variance + 1e-6)
        
        return ensemble_pred, uncertainty
