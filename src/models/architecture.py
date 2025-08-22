"""
Main Lightning Prediction Model Architecture.
Orchestrates all components into a complete prediction system.
UPDATED: Added support for a two-stage prediction model where Stage A predicts
         the convective environment and Stage B predicts lightning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union
from omegaconf import DictConfig

from .encoders import CAPEEncoder, TerrainEncoder, ERA5Encoder
from .fusion import MeteorologicalFusion, MultiScaleFusion
# MODIFIED: Added PatchBasedTransformer and new ConvectionHead imports
from .components import (GraphNeuralNetwork, LightweightTransformer, 
                         PredictionHead, EfficientConvNet, PatchBasedTransformer,
                         ConvectionHead)
from .domain_adaptation import DomainAdapter

class LightningPredictor(nn.Module):
    """
    Complete Lightning Prediction Model.
    
    Can operate in single-stage (lightning only) or two-stage 
    (convection -> lightning) mode based on configuration.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize Lightning Prediction Model.
        
        Args:
            config: Model configuration containing all hyperparameters
        """
        super().__init__()
        
        self.config = config
        self.model_config = config.model
        
        # --- NEW: Check if two-stage model is configured ---
        self.two_stage_model = 'convection_head' in self.model_config
        
        # Store key parameters
        self.cape_only_mode = True  # Will be False when ERA5 is added
        self.use_domain_adaptation = getattr(config.training, 'domain_adaptation', {}).get('enabled', False)
        
        # Initialize all components
        self._build_encoders()
        self._build_fusion_modules()
        self._build_core_processing()
        self._build_prediction_heads() # Renamed for clarity
        self._build_domain_adaptation()
        
        # Initialize physics constraints
        self.physics_weight = getattr(config.training, 'physics', {}).get('charge_separation_weight', 0.05)
        
        print(f"✅ Lightning Prediction Model initialized")
        print(f"   - Two-Stage Model: {self.two_stage_model}")
        print(f"   - CAPE-only mode: {self.cape_only_mode}")
        print(f"   - Domain adaptation: {self.use_domain_adaptation}")
        print(f"   - Physics constraints: {self.physics_weight > 0}")
    
    def _build_encoders(self):
        """Build encoder components."""
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
    
    def _build_fusion_modules(self):
        """Build fusion components."""
        fusion_config = self.model_config.fusion
        
        # Meteorological Fusion (combines CAPE + future ERA5)
        era5_channels = None if self.cape_only_mode else self.era5_encoder.output_channels
        
        self.meteorological_fusion = MeteorologicalFusion(
            cape_channels=self.cape_encoder.output_channels,
            era5_channels=era5_channels,
            output_channels=fusion_config.meteorological.hidden_dim,
            fusion_method=fusion_config.meteorological.fusion_method
        )
        
        # Multi-Scale Fusion (combines meteorological + terrain, upsamples 25km→3km)
        self.multiscale_fusion = MultiScaleFusion(
            met_channels=fusion_config.meteorological.hidden_dim,
            terrain_channels=self.terrain_encoder.embedding_dim,
            output_channels=fusion_config.meteorological.hidden_dim,  # Use same as met_channels 
            upsampling_factor=8.33,  # 25km → 3km
            fusion_method="terrain_guided"
        )
        
        # Correct the output channels
        self.fusion_output_channels = fusion_config.meteorological.hidden_dim
        
        print(f"   ✓ Fusion modules built (Output: {self.fusion_output_channels}ch)")
    
    def _build_core_processing(self):
        """Build core processing components (EfficientConvNet + Patch-Based Transformer)."""
        
        # EfficientConvNet for spatial processing
        convnet_config = self.model_config.gnn  # Keep same config section name for compatibility
        self.spatial_processor = EfficientConvNet(
            input_channels=self.fusion_output_channels,
            hidden_channels=convnet_config.hidden_dim,
            output_channels=convnet_config.hidden_dim,
            num_layers=convnet_config.num_layers,
            kernel_sizes=[3, 5, 7],  # Multi-scale kernels
            dropout=convnet_config.dropout,
            use_multiscale=True,
            use_attention=True
        )
        
        # MODIFIED: Patch-Based Transformer with ADAPTIVE patch size
        transformer_config = self.model_config.transformer
        
        # Calculate adaptive patch size based on domain size
        target_lightning_size = tuple(self.config.data.domain.grid_size_3km)
        target_patches_per_dim = getattr(transformer_config, 'target_patches_per_dim', 25)
        max_patch_size = getattr(transformer_config, 'max_patch_size', 32)
        min_patch_size = getattr(transformer_config, 'min_patch_size', 16)
        
        # Adaptive patch size calculation
        avg_domain_size = (target_lightning_size[0] + target_lightning_size[1]) // 2
        adaptive_patch_size = max(min_patch_size, 
                                 min(max_patch_size, 
                                     avg_domain_size // target_patches_per_dim))
        
        print(f"   📐 Adaptive patch size: {adaptive_patch_size}x{adaptive_patch_size} "
              f"({adaptive_patch_size*3}km area) for domain {target_lightning_size}")
        
        self.transformer = PatchBasedTransformer(
            input_channels=convnet_config.hidden_dim,
            hidden_dim=transformer_config.hidden_dim,
            num_layers=transformer_config.num_layers,
            num_heads=transformer_config.num_heads,
            patch_size=adaptive_patch_size,  # ADAPTIVE: Changes with domain size
            dropout=transformer_config.dropout,
            attention_type=transformer_config.attention_type
        )
        
        print(f"   ✓ Core processing built (ConvNet: {convnet_config.hidden_dim}ch, "
              f"Adaptive Patch Transformer: {transformer_config.hidden_dim}ch)")
    
    def _build_prediction_heads(self):
        """Build prediction head(s) based on configuration."""
        # Stage B: Lightning Prediction Head (always present)
        pred_config = self.model_config.prediction_head
        self.prediction_head = PredictionHead(
            input_channels=self.model_config.transformer.hidden_dim,
            hidden_dim=pred_config.hidden_dim,
            output_dim=pred_config.output_dim,
            activation=pred_config.activation
        )
        print(f"   ✓ Lightning head built (Output: {pred_config.output_dim}ch)")

        # --- NEW: Stage A: Convection Prediction Head (optional) ---
        if self.two_stage_model:
            convection_config = self.model_config.convection_head
            self.convection_head = ConvectionHead(
                input_channels=self.model_config.transformer.hidden_dim,
                hidden_dim=convection_config.hidden_dim,
                output_dim=convection_config.output_dim,
                dropout=convection_config.dropout
            )
            print(f"   ✓ Convection head built (Output: {convection_config.output_dim}ch)")
        else:
            self.convection_head = None

    def _build_domain_adaptation(self):
        """Build domain adaptation module."""
        if self.use_domain_adaptation:
            adapt_config = self.model_config.domain_adapter
            
            self.domain_adapter = DomainAdapter(
                terrain_features=self.terrain_encoder.embedding_dim,
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
        Forward pass through complete lightning prediction model.
        
        Args:
            cape_data: CAPE data (B, 1, H_25km, W_25km)
            terrain_data: Terrain data (B, 1, H_1km, W_1km) 
            era5_data: ERA5 data (B, 9, 7, H_25km, W_25km) - future use
            domain_adaptation: Whether to apply domain adaptation
            
        Returns:
            Dictionary containing predictions and intermediate features
        """
        # Steps 1-6: Feature extraction (same for both model types)
        # ... (code is identical to original until after transformer)
        batch_size = cape_data.shape[0]
        target_lightning_size = tuple(self.config.data.domain.grid_size_3km)
        cape_features = self.cape_encoder(cape_data)
        terrain_features = self.terrain_encoder(terrain_data, target_lightning_size)
        era5_features = None
        if era5_data is not None and self.era5_encoder is not None:
            era5_features = self.era5_encoder(era5_data)
        meteorological_features = self.meteorological_fusion(cape_features, era5_features)
        fused_features = self.multiscale_fusion(
            meteorological_features, terrain_features, target_size=target_lightning_size
        )
        if domain_adaptation and self.domain_adapter is not None:
            processing_features = self.domain_adapter(
                fused_features, terrain_features, meteorological_features
            )
        else:
            processing_features = fused_features
        convnet_features = self.spatial_processor(processing_features)
        transformer_features = self.transformer(convnet_features)
        
        # Step 7: Prediction
        # Stage B: Lightning prediction (always happens)
        lightning_prediction = self.prediction_head(transformer_features)
        
        # Prepare output dictionary
        output = {
            'lightning_prediction': lightning_prediction,
            'cape_features': cape_features,
            'terrain_features': terrain_features,
            'meteorological_features': meteorological_features,
            'fused_features': fused_features,
            'convnet_features': convnet_features,
            'transformer_features': transformer_features
        }

        # --- NEW: Stage A: Convection prediction (if enabled) ---
        if self.two_stage_model:
            convection_prediction = self.convection_head(transformer_features)
            output['convection_prediction'] = convection_prediction
        
        if era5_features is not None:
            output['era5_features'] = era5_features
            
        if domain_adaptation and self.domain_adapter is not None and 'adapted_features' not in output:
             output['adapted_features'] = processing_features
        
        return output
    
    def predict(self, 
                cape_data: torch.Tensor,
                terrain_data: torch.Tensor,
                era5_data: Optional[torch.Tensor] = None,
                return_features: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Simplified prediction interface.
        
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
            
            # Update meteorological fusion
            self.meteorological_fusion = MeteorologicalFusion(
                cape_channels=self.cape_encoder.output_channels,
                era5_channels=self.era5_encoder.output_channels,
                output_channels=self.model_config.fusion.meteorological.hidden_dim,
                fusion_method=self.model_config.fusion.meteorological.fusion_method
            )
        
        self.cape_only_mode = False
        print("✅ ERA5 mode enabled - model now accepts ERA5 3D data")
    
    def get_model_info(self) -> Dict:
        """Get model information and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        components_info = {
            'cape_encoder': self.cape_encoder.output_channels,
            'terrain_encoder': self.terrain_encoder.embedding_dim,
            'era5_encoder': self.era5_encoder.output_channels if self.era5_encoder else 0,
            'fusion_output': self.fusion_output_channels,
            'convnet_hidden': self.model_config.gnn.hidden_dim,
            'transformer_hidden': self.model_config.transformer.hidden_dim,
            'lightning_head_output': self.model_config.prediction_head.output_dim
        }
        
        if self.two_stage_model:
            components_info['convection_head_output'] = self.model_config.convection_head.output_dim

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'cape_only_mode': self.cape_only_mode,
            'domain_adaptation_enabled': self.use_domain_adaptation,
            'two_stage_model_enabled': self.two_stage_model,
            'physics_weight': self.physics_weight,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'components': components_info
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

class LightningPredictorFactory:
    """Factory class for creating Lightning Predictor models with different configurations."""
    
    @staticmethod
    def create_cape_only_model(config: DictConfig) -> "LightningPredictor":
        """Create CAPE-only model (current implementation)."""
        model = LightningPredictor(config)
        return model
    
    @staticmethod
    def create_full_model(config: DictConfig) -> "LightningPredictor":
        """Create full model with ERA5 support (future)."""
        model = LightningPredictor(config)
        model.enable_era5_mode()
        return model
    
    @staticmethod
    def create_lightweight_model(config: DictConfig) -> "LightningPredictor":
        """Create lightweight model for inference."""
        # Modify config for lighter model
        light_config = config.copy()
        light_config.model.gnn.num_layers = 2
        light_config.model.transformer.num_layers = 2
        light_config.model.gnn.hidden_dim = 128
        light_config.model.transformer.hidden_dim = 128
        
        model = LightningPredictor(light_config)
        return model

def create_model_from_config(config: DictConfig, model_type: str = "cape_only") -> "LightningPredictor":
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

def load_pretrained_model(checkpoint_path: str, config: DictConfig) -> "LightningPredictor":
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

class ModelSummary:
    """Utility class for model analysis and visualization."""
    
    @staticmethod
    def print_model_summary(model: "LightningPredictor", input_shapes: Dict[str, Tuple]):
        """Print detailed model summary."""
        print("\n" + "="*80)
        print("LIGHTNING PREDICTION MODEL SUMMARY")
        print("="*80)
        
        # Model info
        info = model.get_model_info()
        print(f"Total Parameters: {info['total_parameters']:,}")
        print(f"Trainable Parameters: {info['trainable_parameters']:,}")
        print(f"Model Size: {info['model_size_mb']:.2f} MB")
        print(f"CAPE-only Mode: {info['cape_only_mode']}")
        print(f"Two-Stage Model: {info['two_stage_model_enabled']}")
        print(f"Domain Adaptation: {info['domain_adaptation_enabled']}")
        
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
            if 'convection_prediction' in output:
                print(f"Convection Prediction Output: {output['convection_prediction'].shape}")

        print("="*80)
    
    @staticmethod
    def analyze_memory_usage(model: "LightningPredictor", input_shapes: Dict[str, Tuple]) -> Dict:
        """Analyze model memory usage."""
        import torch.profiler
        
        model.train()
        
        # Create dummy inputs
        dummy_inputs = {}
        for name, shape in input_shapes.items():
            dummy_inputs[name] = torch.randn(1, *shape, requires_grad=True)
        
        # Profile memory usage
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            with_stack=True
        ) as prof:
            output = model(**dummy_inputs)
            loss = output['lightning_prediction'].sum()
            loss.backward()
        
        # Extract memory info
        memory_info = {
            'peak_memory_mb': torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0,
            'profile_summary': prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10)
        }
        
        return memory_info

# Example usage and testing
if __name__ == "__main__":
    # Example configuration (simplified)
    from omegaconf import OmegaConf
    
    config = OmegaConf.create({
        'data': { # Add dummy data config for testing
            'domain': {
                'grid_size_3km': (256, 256)
            }
        },
        'model': {
            'encoders': {
                'cape': {
                    'channels': [32, 64, 128, 256],
                    'kernel_sizes': [7, 5, 3, 3],
                    'activation': 'relu',
                    'dropout': 0.1
                },
                'terrain': {
                    'embedding_dim': 64,
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
                    'hidden_dim': 256,
                    'fusion_method': 'concatenation'
                },
                'multiscale': {
                    'upsampling_factor': 8.33
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
                'num_layers': 4,
                'num_heads': 8,
                'dropout': 0.1,
                'attention_type': 'linear'
            },
            'prediction_head': {
                'hidden_dim': 128,
                'output_dim': 1,
                'activation': 'sigmoid'
            },
            # --- NEW: Add config for the convection head ---
            'convection_head': {
                'hidden_dim': 64,
                'output_dim': 1,
                'dropout': 0.1
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
    
    # Create model
    model = create_model_from_config(config, "cape_only")
    
    # Print summary
    input_shapes = {
        'cape_data': (1, 85, 85),
        'terrain_data': (1, 710, 710)
    }
    
    ModelSummary.print_model_summary(model, input_shapes)
