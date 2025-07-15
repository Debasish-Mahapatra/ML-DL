"""
Main Lightning Prediction Model Architecture - Updated for Multi-Resolution Learning.

This version implements Strategy 3: Multi-Resolution Learning approach that:
1. Keeps meteorological processing at native 25km resolution
2. Adds spatial refinement network using 1km terrain data
3. Final layer: 25km meteorological features + 1km terrain → 3km predictions

Key improvements:
- Eliminates problematic 8.33x upsampling
- Fixes shape mismatch issues
- Physics-based multi-resolution approach
- Preserves information at native resolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
from omegaconf import DictConfig

from .encoders import CAPEEncoder, TerrainEncoder, ERA5Encoder
from .fusion import (
    MeteorologicalFusion,
    MultiResolutionFusion,
    MultiResolutionTerrainProcessor,
    MultiResolutionMeteorologicalProcessor
)
from .components import GraphNeuralNetwork, LightweightTransformer, PredictionHead
from .domain_adaptation import DomainAdapter

class LightningPredictor(nn.Module):
    """
    Complete Lightning Prediction Model with Multi-Resolution Learning.
    
    Updated Architecture Flow:
    Input Data → Encoders → Multi-Resolution Meteorological Processing → 
    Multi-Resolution Terrain Processing → Multi-Resolution Fusion → 
    GNN → Transformer → Prediction Head → Lightning Probability
    
    Key Changes:
    - Replaced MultiScaleFusion with MultiResolutionFusion
    - Added separate processors for meteorological and terrain data
    - Eliminated problematic upsampling
    - Fixed shape mismatch issues
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize Lightning Prediction Model with Multi-Resolution Learning.
        
        Args:
            config: Model configuration containing all hyperparameters
        """
        super().__init__()
        
        self.config = config
        self.model_config = config.model
        
        # Store key parameters
        self.cape_only_mode = True  # Will be False when ERA5 is added
        self.use_domain_adaptation = getattr(config.training, 'domain_adaptation', {}).get('enabled', False)
        
        # Initialize all components
        self._build_encoders()
        self._build_multi_resolution_processors()
        self._build_multi_resolution_fusion()
        self._build_core_processing()
        self._build_prediction_head()
        self._build_domain_adaptation()
        
        # Initialize physics constraints
        self.physics_weight = getattr(config.training, 'physics', {}).get('charge_separation_weight', 0.05)
        
        print(f"✅ Lightning Prediction Model initialized with Multi-Resolution Learning")
        print(f"   - CAPE-only mode: {self.cape_only_mode}")
        print(f"   - Domain adaptation: {self.use_domain_adaptation}")
        print(f"   - Physics constraints: {self.physics_weight > 0}")
        print(f"   - Multi-resolution fusion: ENABLED")
    
    def _build_encoders(self):
        """Build encoder components (unchanged)."""
        encoder_config = self.model_config.encoders
        
        # CAPE Encoder (always present)
        self.cape_encoder = CAPEEncoder(
            in_channels=1,
            channels=encoder_config.cape.channels,
            kernel_sizes=encoder_config.cape.kernel_sizes,
            activation=encoder_config.cape.activation,
            dropout=encoder_config.cape.dropout
        )
        
        # Terrain Encoder (always present)
        self.terrain_encoder = TerrainEncoder(
            in_channels=1,
            embedding_dim=encoder_config.terrain.embedding_dim,
            learnable_downsample=encoder_config.terrain.learnable_downsample
        )
        
        # ERA5 Encoder (future use)
        if not self.cape_only_mode:
            self.era5_encoder = ERA5Encoder(
                in_channels=encoder_config.era5.in_channels,
                pressure_levels=encoder_config.era5.pressure_levels,
                channels=encoder_config.era5.channels,
                kernel_sizes=encoder_config.era5.kernel_sizes,
                activation=encoder_config.era5.activation,
                dropout=encoder_config.era5.dropout
            )
        else:
            self.era5_encoder = None
        
        print(f"   ✓ Encoders built (CAPE: {self.cape_encoder.output_channels}ch, "
              f"Terrain: {encoder_config.terrain.embedding_dim}ch)")
    
    def _build_multi_resolution_processors(self):
        """Build multi-resolution processors for meteorological and terrain data."""
        
        # Multi-resolution meteorological processor
        self.meteorological_processor = MultiResolutionMeteorologicalProcessor(
            cape_channels=self.cape_encoder.output_channels,
            era5_channels=self.era5_encoder.output_channels if self.era5_encoder else None,
            output_channels=self.model_config.fusion.meteorological.hidden_dim,
            cape_only_mode=self.cape_only_mode,
            native_resolution_km=25
        )
        
        # Multi-resolution terrain processor
        self.terrain_processor = MultiResolutionTerrainProcessor(
            in_channels=1,  # Raw elevation data
            output_channels=self.model_config.encoders.terrain.embedding_dim,
            target_resolution_km=3,
            input_resolution_km=1
        )
        
        print(f"   ✓ Multi-resolution processors built")
    
    def _build_multi_resolution_fusion(self):
        """Build multi-resolution fusion module."""
        
        # Multi-resolution fusion (replaces old MultiScaleFusion)
        self.multi_resolution_fusion = MultiResolutionFusion(
            met_channels=self.model_config.fusion.meteorological.hidden_dim,
            terrain_channels=self.model_config.encoders.terrain.embedding_dim,
            output_channels=self.model_config.fusion.meteorological.hidden_dim,
            target_resolution_km=3,
            met_resolution_km=25,
            terrain_resolution_km=1
        )
        
        # Store fusion output channels
        self.fusion_output_channels = self.model_config.fusion.meteorological.hidden_dim
        
        print(f"   ✓ Multi-resolution fusion built (Output: {self.fusion_output_channels}ch)")
    
    def _build_core_processing(self):
        """Build core processing components (unchanged)."""
        
        # Graph Neural Network
        gnn_config = self.model_config.gnn
        self.gnn = GraphNeuralNetwork(
            input_channels=self.fusion_output_channels,
            hidden_channels=gnn_config.hidden_dim,
            output_channels=gnn_config.hidden_dim,
            num_layers=gnn_config.num_layers,
            gnn_type=gnn_config.type,
            num_heads=gnn_config.num_heads,
            dropout=gnn_config.dropout
        )
        
        # Lightweight Transformer
        transformer_config = self.model_config.transformer
        self.transformer = LightweightTransformer(
            input_channels=gnn_config.hidden_dim,
            hidden_dim=transformer_config.hidden_dim,
            num_layers=transformer_config.num_layers,
            num_heads=transformer_config.num_heads,
            dropout=transformer_config.dropout,
            attention_type=transformer_config.attention_type
        )
        
        print(f"   ✓ Core processing built (GNN: {gnn_config.hidden_dim}ch, "
              f"Transformer: {transformer_config.hidden_dim}ch)")
    
    def _build_prediction_head(self):
        """Build prediction head (unchanged)."""
        pred_config = self.model_config.prediction_head
        
        self.prediction_head = PredictionHead(
            input_channels=self.model_config.transformer.hidden_dim,
            hidden_dim=pred_config.hidden_dim,
            output_dim=pred_config.output_dim,
            activation=pred_config.activation
        )
        
        print(f"   ✓ Prediction head built (Output: {pred_config.output_dim}ch)")
    
    def _build_domain_adaptation(self):
        """Build domain adaptation module (unchanged)."""
        if self.use_domain_adaptation:
            adapt_config = self.model_config.domain_adapter
            
            self.domain_adapter = DomainAdapter(
                terrain_features=self.model_config.encoders.terrain.embedding_dim,
                meteorological_features=self.fusion_output_channels,
                terrain_adaptation_dim=adapt_config.terrain_adaptation_dim,
                meteorological_adaptation_dim=adapt_config.meteorological_adaptation_dim,
                dropout=adapt_config.dropout
            )
            print(f"   ✓ Domain adaptation built")
        else:
            self.domain_adapter = None
    
    def forward(self, 
                cape_data: torch.Tensor,
                terrain_data: torch.Tensor,
                era5_data: Optional[torch.Tensor] = None,
                domain_adaptation: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete lightning prediction model with multi-resolution learning.
        
        Args:
            cape_data: CAPE data (B, 1, H_25km, W_25km)
            terrain_data: Terrain data (B, 1, H_1km, W_1km) 
            era5_data: ERA5 data (B, 9, 7, H_25km, W_25km) - future use
            domain_adaptation: Whether to apply domain adaptation
            
        Returns:
            Dictionary containing predictions and intermediate features
        """
        batch_size = cape_data.shape[0]
        
        # Calculate target size for 3km output
        # Use the data config grid sizes
        target_size = (710, 710)  # 3km grid size from your data config
        
        # Step 1: Encode inputs at native resolutions
        cape_features = self.cape_encoder(cape_data)  # Stays at 25km
        
        # ERA5 encoding (future)
        if era5_data is not None and self.era5_encoder is not None:
            era5_features = self.era5_encoder(era5_data)
        else:
            era5_features = None
        
        # Step 2: Multi-resolution processing
        # Process meteorological data at native 25km resolution
        meteorological_output = self.meteorological_processor(cape_features, era5_features)
        processed_meteorological = meteorological_output['meteorological_features']
        
        # Process terrain data intelligently from 1km to 3km
        terrain_output = self.terrain_processor(terrain_data, target_size)
        processed_terrain = terrain_output['terrain_features']
        
        # Step 3: Multi-resolution fusion (NO UPSAMPLING)
        # Combines 25km meteorological + 3km terrain → 3km output
        fused_features = self.multi_resolution_fusion(
            processed_meteorological,  # 25km
            processed_terrain,         # 3km
            target_size               # 3km target
        )
        
        # Step 4: Apply domain adaptation if enabled
        if domain_adaptation and self.domain_adapter is not None:
            adapted_features = self.domain_adapter(
                fused_features, 
                processed_terrain, 
                processed_meteorological
            )
            processing_features = adapted_features
        else:
            processing_features = fused_features
        
        # Step 5: Core processing - GNN for spatial relationships
        gnn_features = self.gnn(processing_features)
        
        # Step 6: Core processing - Transformer for patterns
        transformer_features = self.transformer(gnn_features)
        
        # Step 7: Lightning prediction
        lightning_prediction = self.prediction_head(transformer_features)
        
        # Prepare output dictionary
        output = {
            'lightning_prediction': lightning_prediction,
            'cape_features': cape_features,
            'terrain_features': processed_terrain,
            'meteorological_features': processed_meteorological,
            'fused_features': fused_features,
            'gnn_features': gnn_features,
            'transformer_features': transformer_features,
            'meteorological_output': meteorological_output,
            'terrain_output': terrain_output
        }
        
        if era5_features is not None:
            output['era5_features'] = era5_features
            
        if domain_adaptation and self.domain_adapter is not None:
            output['adapted_features'] = adapted_features
        
        return output
    
    def predict(self, 
                cape_data: torch.Tensor,
                terrain_data: torch.Tensor,
                era5_data: Optional[torch.Tensor] = None,
                return_features: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Simplified prediction interface (unchanged).
        
        Args:
            cape_data: CAPE input data
            terrain_data: Terrain input data  
            era5_data: ERA5 input data (future)
            return_features: Whether to return intermediate features
            
        Returns:
            Lightning predictions or full output dictionary
        """
        self.eval()
        
        with torch.no_grad():
            output = self.forward(cape_data, terrain_data, era5_data)
            
            if return_features:
                return output
            else:
                return output['lightning_prediction']
    
    def enable_era5_mode(self):
        """Enable ERA5 mode (for future use when ERA5 data is available)."""
        if self.era5_encoder is None:
            # Build ERA5 encoder
            encoder_config = self.model_config.encoders
            self.era5_encoder = ERA5Encoder(
                in_channels=encoder_config.era5.in_channels,
                pressure_levels=encoder_config.era5.pressure_levels,
                channels=encoder_config.era5.channels,
                kernel_sizes=encoder_config.era5.kernel_sizes,
                activation=encoder_config.era5.activation,
                dropout=encoder_config.era5.dropout
            )
            
            # Update meteorological processor
            self.meteorological_processor = MultiResolutionMeteorologicalProcessor(
                cape_channels=self.cape_encoder.output_channels,
                era5_channels=self.era5_encoder.output_channels,
                output_channels=self.model_config.fusion.meteorological.hidden_dim,
                cape_only_mode=False,
                native_resolution_km=25
            )
        
        self.cape_only_mode = False
        print("✅ ERA5 mode enabled - model now accepts ERA5 3D data")
    
    def get_model_info(self) -> Dict:
        """Get model information and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'cape_only_mode': self.cape_only_mode,
            'domain_adaptation_enabled': self.use_domain_adaptation,
            'physics_weight': self.physics_weight,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'multi_resolution_learning': True,  # NEW: Indicates updated architecture
            'components': {
                'cape_encoder': self.cape_encoder.output_channels,
                'terrain_encoder': self.model_config.encoders.terrain.embedding_dim,
                'era5_encoder': self.era5_encoder.output_channels if self.era5_encoder else 0,
                'fusion_output': self.fusion_output_channels,
                'gnn_hidden': self.model_config.gnn.hidden_dim,
                'transformer_hidden': self.model_config.transformer.hidden_dim,
                'prediction_output': self.model_config.prediction_head.output_dim
            }
        }
    
    def freeze_encoders(self):
        """Freeze encoder parameters (useful for fine-tuning)."""
        for param in self.cape_encoder.parameters():
            param.requires_grad = False
        for param in self.terrain_encoder.parameters():
            param.requires_grad = False
        if self.era5_encoder is not None:
            for param in self.era5_encoder.parameters():
                param.requires_grad = False
        print("✅ Encoders frozen")
    
    def unfreeze_encoders(self):
        """Unfreeze encoder parameters."""
        for param in self.cape_encoder.parameters():
            param.requires_grad = True
        for param in self.terrain_encoder.parameters():
            param.requires_grad = True
        if self.era5_encoder is not None:
            for param in self.era5_encoder.parameters():
                param.requires_grad = True
        print("✅ Encoders unfrozen")

# Factory classes and utility functions remain unchanged
class LightningPredictorFactory:
    """Factory class for creating Lightning Predictor models with different configurations."""
    
    @staticmethod
    def create_cape_only_model(config: DictConfig) -> LightningPredictor:
        """Create CAPE-only model (current implementation)."""
        model = LightningPredictor(config)
        return model
    
    @staticmethod
    def create_full_model(config: DictConfig) -> LightningPredictor:
        """Create full model with ERA5 support (future)."""
        model = LightningPredictor(config)
        model.enable_era5_mode()
        return model
    
    @staticmethod
    def create_lightweight_model(config: DictConfig) -> LightningPredictor:
        """Create lightweight model for inference."""
        # Modify config for lighter model
        light_config = config.copy()
        light_config.model.gnn.num_layers = 2
        light_config.model.transformer.num_layers = 2
        light_config.model.gnn.hidden_dim = 128
        light_config.model.transformer.hidden_dim = 128
        
        model = LightningPredictor(light_config)
        return model

def create_model_from_config(config: DictConfig, model_type: str = "cape_only") -> LightningPredictor:
    """
    Create model from configuration.
    
    Args:
        config: Model configuration
        model_type: Type of model ("cape_only", "full", "lightweight")
        
    Returns:
        Lightning Predictor model
    """
    factory = LightningPredictorFactory()
    
    if model_type == "cape_only":
        return factory.create_cape_only_model(config)
    elif model_type == "full":
        return factory.create_full_model(config)
    elif model_type == "lightweight":
        return factory.create_lightweight_model(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def load_pretrained_model(checkpoint_path: str, config: DictConfig) -> LightningPredictor:
    """
    Load pretrained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Model configuration
        
    Returns:
        Loaded Lightning Predictor model
    """
    model = LightningPredictor(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"✅ Model loaded from {checkpoint_path}")
    return model

# ModelSummary class remains unchanged
class ModelSummary:
    """Utility class for model analysis and visualization."""
    
    @staticmethod
    def print_model_summary(model: LightningPredictor, input_shapes: Dict[str, Tuple]):
        """Print detailed model summary."""
        print("\n" + "="*80)
        print("LIGHTNING PREDICTION MODEL SUMMARY - MULTI-RESOLUTION LEARNING")
        print("="*80)
        
        # Model info
        info = model.get_model_info()
        print(f"Total Parameters: {info['total_parameters']:,}")
        print(f"Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"Model Size: {info['model_size_mb']:.2f} MB")
        print(f"CAPE-only Mode: {info['cape_only_mode']}")
        print(f"Domain Adaptation: {info['domain_adaptation_enabled']}")
        print(f"Multi-Resolution Learning: {info['multi_resolution_learning']}")
        
        print("\nCOMPONENT ARCHITECTURE:")
        print("-"*40)
        for component, channels in info['components'].items():
            if channels > 0:
                print(f"{component.replace('_', ' ').title()}: {channels} channels")
        
        print("\nINPUT/OUTPUT SHAPES:")
        print("-"*40)
        for input_name, shape in input_shapes.items():
            print(f"{input_name}: {shape}")
        
        # Test forward pass to get output shapes
        model.eval()
        with torch.no_grad():
            # Create dummy inputs
            dummy_inputs = {}
            for name, shape in input_shapes.items():
                dummy_inputs[name] = torch.randn(1, *shape)
            
            output = model(**dummy_inputs)
            print(f"Lightning Prediction Output: {output['lightning_prediction'].shape}")
        
        print("="*80)
        print("✅ MULTI-RESOLUTION LEARNING ACTIVE - NO UPSAMPLING ARTIFACTS")
        print("="*80)

# Example usage remains the same but now uses multi-resolution learning
if __name__ == "__main__":
    # Example configuration (simplified)
    from omegaconf import OmegaConf
    
    config = OmegaConf.create({
        'model': {
            'encoders': {
                'cape': {
                    'channels': [64, 128, 256, 512],
                    'kernel_sizes': [7, 5, 3, 3],
                    'activation': 'relu',
                    'dropout': 0.1
                },
                'terrain': {
                    'embedding_dim': 128,
                    'learnable_downsample': True
                },
                'era5': {
                    'in_channels': 9,
                    'pressure_levels': 7,
                    'channels': [32, 64, 128, 256],
                    'kernel_sizes': [(3,3,3), (3,3,3), (3,3,3), (3,3,3)],
                    'activation': 'relu',
                    'dropout': 0.1
                }
            },
            'fusion': {
                'meteorological': {
                    'hidden_dim': 1024,
                    'fusion_method': 'concatenation'
                }
            },
            'gnn': {
                'hidden_dim': 256,
                'num_layers': 3,
                'type': 'gat',
                'num_heads': 8,
                'dropout': 0.1
            },
            'transformer': {
                'hidden_dim': 256,
                'num_layers': 6,
                'num_heads': 16,
                'dropout': 0.1,
                'attention_type': 'linear'
            },
            'prediction_head': {
                'hidden_dim': 256,
                'output_dim': 1,
                'activation': 'sigmoid'
            },
            'domain_adapter': {
                'terrain_adaptation_dim': 64,
                'meteorological_adaptation_dim': 32,
                'dropout': 0.1
            }
        },
        'training': {
            'domain_adaptation': {'enabled': True},
            'physics': {'charge_separation_weight': 0.05}
        }
    })
    
    # Create model with multi-resolution learning
    model = create_model_from_config(config, "cape_only")
    
    # Print summary
    input_shapes = {
        'cape_data': (1, 85, 85),
        'terrain_data': (1, 2130, 2130)
    }
    
    ModelSummary.print_model_summary(model, input_shapes)