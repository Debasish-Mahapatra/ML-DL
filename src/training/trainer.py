"""
PyTorch Lightning trainer for the lightning prediction model with DeepSpeed integration.
Complete implementation with all training, validation, test, and prediction methods.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau
from typing import Dict, Any, Optional, Tuple, Union
from omegaconf import DictConfig
import numpy as np
import logging
from pathlib import Path
import json

from ..models.architecture import LightningPredictor
from .loss_functions import CompositeLoss, create_loss_function
from .metrics import LightningMetrics, MetricTracker, compute_class_weights

logger = logging.getLogger(__name__)

class LightningTrainer(pl.LightningModule):
    """
    PyTorch Lightning trainer for lightning prediction with DeepSpeed, physics constraints,
    memory optimization, and domain adaptation support.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize Lightning Trainer with DeepSpeed support.
        
        Args:
            config: Complete training configuration
        """
        super().__init__()
        
        self.config = config
        self.model_config = config.model
        self.training_config = config.training
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Initialize model
        self.model = LightningPredictor(config)
        
        # Initialize loss function
        self.loss_function = create_loss_function(self.training_config.loss)
        
        # Initialize metrics
        self.train_metrics = LightningMetrics(
            threshold=0.5,
            spatial_tolerance=1,
            compute_spatial_metrics=True
        )
        self.val_metrics = LightningMetrics(
            threshold=0.5,
            spatial_tolerance=1,
            compute_spatial_metrics=True
        )
        self.test_metrics = LightningMetrics(
            threshold=0.5,
            spatial_tolerance=1,
            compute_spatial_metrics=True
        )
        
        # Metric tracking
        self.metric_tracker = MetricTracker()
        
        # Training state
        self.automatic_optimization = True
        self.gradient_accumulation_steps = getattr(self.training_config, 'gradient_accumulation_steps', 4)
        self.current_accumulation_step = 0
        
        # Domain adaptation settings
        self.domain_adaptation_enabled = getattr(config.training, 'domain_adaptation', {}).get('enabled', False)
        self.domain_adaptation_warmup = getattr(config.training, 'domain_adaptation', {}).get('warmup_epochs', 10)
        
        # Physics loss weighting
        self.physics_loss_weight = getattr(config.training, 'physics', {}).get('weight', 0.1)
        
        # DeepSpeed settings
        self.deepspeed_enabled = getattr(config.training, 'deepspeed', {}).get('enabled', False)
        
        # Class weights for imbalanced data
        self.class_weights = None
        
        print(f"Lightning Trainer initialized:")
        print(f"  - Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  - Gradient accumulation: {self.gradient_accumulation_steps} steps")
        print(f"  - Domain adaptation: {self.domain_adaptation_enabled}")
        print(f"  - Physics loss weight: {self.physics_loss_weight}")
        print(f"  - DeepSpeed enabled: {self.deepspeed_enabled}")
    
    def forward(self, cape_data: torch.Tensor, terrain_data: torch.Tensor, 
                era5_data: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(cape_data, terrain_data, era5_data, 
                         domain_adaptation=self.domain_adaptation_enabled)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler for DeepSpeed."""
        
        # DeepSpeed handles optimizer configuration, but we need to provide fallback
        
        if self.deepspeed_enabled:
            # Return minimal config - DeepSpeed will override
            optimizer = AdamW(self.parameters(), lr=float(self.training_config.optimizer.lr))
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": CosineAnnealingWarmRestarts(
                        optimizer,  # Use the same optimizer instance
                        T_0=50
                    ),
                    "interval": "epoch"
                }
            }
        
        # Standard configuration for non-DeepSpeed training
        optimizer_config = self.training_config.optimizer
        
        # Create optimizer
        if optimizer_config.type.lower() == "adamw":
            optimizer = AdamW(
                self.parameters(),
                lr=optimizer_config.lr,
                weight_decay=optimizer_config.weight_decay,
                betas=getattr(optimizer_config, 'betas', [0.9, 0.999]),
                eps=getattr(optimizer_config, 'eps', 1e-8)
            )
        elif optimizer_config.type.lower() == "sgd":
            optimizer = SGD(
                self.parameters(),
                lr=optimizer_config.lr,
                weight_decay=optimizer_config.weight_decay,
                momentum=getattr(optimizer_config, 'momentum', 0.9)
            )
        else:
            # Default to AdamW
            optimizer = AdamW(
                self.parameters(),
                lr=optimizer_config.lr,
                weight_decay=getattr(optimizer_config, 'weight_decay', 1e-5)
            )
        
        # Create scheduler
        scheduler_config = self.training_config.scheduler
        
        if scheduler_config.type == "CosineAnnealingWarmRestarts":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=scheduler_config.T_0,
                T_mult=getattr(scheduler_config, 'T_mult', 2),
                eta_min=getattr(scheduler_config, 'eta_min', 1e-6)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        elif scheduler_config.type == "OneCycleLR":
            scheduler = OneCycleLR(
                optimizer,
                max_lr=optimizer_config.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=getattr(scheduler_config, 'pct_start', 0.3)
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                }
            }
        else:
            return {"optimizer": optimizer}
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with memory optimization."""
        
        # Extract data
        cape_data = batch['cape']
        terrain_data = batch['terrain']
        lightning_target = batch['lightning']
        era5_data = batch.get('era5', None)
        
        # Forward pass
        outputs = self.forward(cape_data, terrain_data, era5_data)
        predictions = outputs['lightning_prediction']
        
        # Compute loss
        loss_dict = self.loss_function(
            predictions, 
            lightning_target,
            cape_data=cape_data,
            terrain_data=terrain_data,
        )
        
        total_loss = loss_dict['total_loss']
        
        # Update metrics
        self.train_metrics.update(predictions, lightning_target)
        
        # Log losses
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        for loss_name, loss_value in loss_dict.items():
            if loss_name != 'total_loss':
                self.log(f'train_{loss_name}', loss_value, on_step=False, on_epoch=True)
        
        # Update accumulation step counter
        self.current_accumulation_step += 1
        
        # Clear cache periodically for memory management
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        
        cape_data = batch['cape']
        terrain_data = batch['terrain']
        lightning_target = batch['lightning']
        era5_data = batch.get('era5', None)
        
        # Forward pass
        outputs = self.forward(cape_data, terrain_data, era5_data)
        predictions = outputs['lightning_prediction']
        
        # Compute loss
        loss_dict = self.loss_function(
            predictions, 
            lightning_target,
            cape_data=cape_data,
            terrain_data=terrain_data,
            model_outputs=outputs
        )
        
        total_loss = loss_dict['total_loss']
        
        # Update metrics
        self.val_metrics.update(predictions, lightning_target)
        
        # Log losses
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        for loss_name, loss_value in loss_dict.items():
            if loss_name != 'total_loss':
                self.log(f'val_{loss_name}', loss_value, on_step=False, on_epoch=True)
        
        return total_loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        
        cape_data = batch['cape']
        terrain_data = batch['terrain']
        lightning_target = batch['lightning']
        era5_data = batch.get('era5', None)
        
        # Forward pass
        outputs = self.forward(cape_data, terrain_data, era5_data)
        predictions = outputs['lightning_prediction']
        
        # Compute loss
        loss_dict = self.loss_function(
            predictions, 
            lightning_target,
            cape_data=cape_data,
            terrain_data=terrain_data,
            model_outputs=outputs
        )
        
        total_loss = loss_dict['total_loss']
        
        # Update metrics
        self.test_metrics.update(predictions, lightning_target)
        
        return total_loss
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step for inference."""
        
        cape_data = batch['cape']
        terrain_data = batch['terrain']
        era5_data = batch.get('era5', None)
        
        outputs = self.model(cape_data, terrain_data, era5_data, domain_adaptation=False)
        
        return {
            'predictions': outputs['lightning_prediction'],
            'cape_features': outputs.get('cape_features'),
            'terrain_features': outputs.get('terrain_features'),
            'fused_features': outputs.get('fused_features'),
            'gnn_features': outputs.get('gnn_features'),
            'transformer_features': outputs.get('transformer_features')
        }
    
    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        
        # Log current learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_epoch=True)
        
        # Reset accumulation step counter
        self.current_accumulation_step = 0
        
        # Domain adaptation warmup
        if self.domain_adaptation_enabled and self.current_epoch < self.domain_adaptation_warmup:
            # Reduce domain adaptation influence during warmup
            domain_weight = self.current_epoch / self.domain_adaptation_warmup
            self.log('domain_adaptation_weight', domain_weight, on_epoch=True)
    
    def on_train_epoch_end(self) -> None:
        """Log training metrics at epoch end."""
        
        # Compute and log training metrics
        train_metrics = self.train_metrics.compute()
        
        for metric_name, metric_value in train_metrics.items():
            self.log(f'train_{metric_name}', metric_value, on_epoch=True)
        
        # Reset metrics
        self.train_metrics.reset()
        
        # Reset accumulation step counter
        self.current_accumulation_step = 0
    
    def on_validation_epoch_end(self) -> None:
        """Log validation metrics at epoch end."""
        
        # Compute and log validation metrics
        val_metrics = self.val_metrics.compute()
        
        for metric_name, metric_value in val_metrics.items():
            self.log(f'val_{metric_name}', metric_value, on_epoch=True, prog_bar=(metric_name == 'f1_score'))
        
        # Track metrics
        self.metric_tracker.update(self.current_epoch, val_metrics)
        
        # Reset metrics
        self.val_metrics.reset()
        
        # Log best metrics so far
        best_metrics = self.metric_tracker.get_best_metrics()
        for metric_name, (best_value, best_epoch) in best_metrics.items():
            self.log(f'best_{metric_name}', best_value, on_epoch=True)
    
    def on_test_epoch_end(self) -> Dict[str, float]:
        """Log test metrics at epoch end."""
        
        # Compute and log test metrics
        test_metrics = self.test_metrics.compute()
        
        for metric_name, metric_value in test_metrics.items():
            self.log(f'test_{metric_name}', metric_value, on_epoch=True)
        
        # Reset metrics
        self.test_metrics.reset()
        
        return test_metrics
    
    def on_fit_start(self):
        """Called at the start of training."""
        
        # Log model information
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Log hardware information
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU: {device_name} ({memory_gb:.1f} GB)")
        
        # Initialize class weights if needed
        if self.class_weights is None and hasattr(self.trainer.datamodule, 'get_class_weights'):
            self.class_weights = self.trainer.datamodule.get_class_weights()
            logger.info(f"Class weights initialized: {self.class_weights}")
    
    def on_fit_end(self):
        """Called at the end of training."""
        
        # Log final metrics
        best_metrics = self.metric_tracker.get_best_metrics()
        logger.info("Training completed. Best validation metrics:")
        for metric_name, (best_value, best_epoch) in best_metrics.items():
            logger.info(f"  {metric_name}: {best_value:.4f} (epoch {best_epoch})")
    
    def on_validation_start(self):
        """Called at the start of validation."""
        pass
    
    def on_test_start(self):
        """Called at the start of testing."""
        pass
    
    def on_predict_start(self):
        """Called at the start of prediction."""
        self.model.eval()
    
    def configure_callbacks(self):
        """Configure additional callbacks if needed."""
        return []
    
    def on_before_optimizer_step(self, optimizer):
        """Called before optimizer step."""
        
        # Log gradient norms for debugging
        if self.current_epoch == 0 and self.global_step < 10:
            grad_norm = 0.0
            for p in self.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            self.log('grad_norm', grad_norm, on_step=True)
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Custom optimizer step with gradient accumulation."""
        
        # Handle gradient accumulation
        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
            # Step the optimizer
            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()
        else:
            # Just run the closure to compute gradients
            optimizer_closure()


def create_deepspeed_strategy(config: DictConfig) -> Optional[DeepSpeedStrategy]:
    """Create DeepSpeed strategy from configuration."""
    
    deepspeed_config = config.training.get('deepspeed', {})
    
    if not deepspeed_config.get('enabled', False):
        return None
    
    # Load DeepSpeed config file
    config_path = deepspeed_config.get('config_path', 'config/deepspeed_config.json')
    
    if not Path(config_path).exists():
        logger.warning(f"DeepSpeed config file not found: {config_path}")
        return None
    
    # Load and validate DeepSpeed configuration
    with open(config_path, 'r') as f:
        ds_config = json.load(f)
    
    # Override some settings from training config
    if 'train_micro_batch_size_per_gpu' not in ds_config:
        ds_config['train_micro_batch_size_per_gpu'] = config.data.batch_size
    
    # Create strategy
    strategy = DeepSpeedStrategy(
        stage=deepspeed_config.get('stage', 2),
        offload_optimizer=deepspeed_config.get('offload_optimizer', True),
        offload_parameters=deepspeed_config.get('offload_parameters', True),
        cpu_checkpointing=deepspeed_config.get('cpu_checkpointing', True),
        config=ds_config,
        logging_level=logging.INFO
    )
    
    logger.info(f"✅ DeepSpeed strategy created with stage {deepspeed_config.get('stage', 2)}")
    return strategy


def create_trainer(config: DictConfig, 
                  experiment_name: str,
                  logger_type: str = "tensorboard") -> Tuple[pl.Trainer, LightningTrainer]:
    """
    Create PyTorch Lightning trainer with DeepSpeed support and all callbacks.
    
    Args:
        config: Training configuration
        experiment_name: Name for the experiment
        logger_type: Type of logger ("tensorboard", "wandb")
        
    Returns:
        Tuple of (pl.Trainer, LightningTrainer)
    """
    
    # Initialize model
    lightning_module = LightningTrainer(config)
    
    # Setup logger
    if logger_type == "tensorboard":
        experiment_logger = TensorBoardLogger(
            save_dir="logs",
            name=experiment_name,
            version=None
        )
    elif logger_type == "wandb":
        experiment_logger = WandbLogger(
            project="lightning-prediction",
            name=experiment_name,
            save_dir="logs"
        )
    else:
        experiment_logger = None
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"experiments/{experiment_name}/checkpoints",
        filename="{epoch:02d}-{val_f1_score:.3f}",
        monitor="val_f1_score",
        mode="max",
        save_top_k=int(config.training.get('save_top_k', 3)),
        save_last=True,
        verbose=True,
        save_weights_only=False,  # Save full model for DeepSpeed compatibility
        every_n_epochs=config.training.get('every_n_epochs', 1)
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_f1_score",
        mode="max",
        patience=int(config.training.get('patience', 15)),
        min_delta=float(config.training.get('min_delta', 1e-4)),
        verbose=True
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Create DeepSpeed strategy if enabled
    strategy = create_deepspeed_strategy(config)
    if strategy is None:
        strategy = "auto"  # Use default strategy
        logger.info("Using default PyTorch Lightning strategy")
    
    # Create trainer with DeepSpeed support
    trainer_kwargs = {
        'max_epochs': int(config.training.get('max_epochs', 100)),
        'accelerator': config.training.get('accelerator', 'gpu'),
        'devices': int(config.training.get('devices', 1)),
        'precision': config.training.get('precision', 16),  # Use FP16 for DeepSpeed
        'strategy': strategy,
        'callbacks': callbacks,
        'logger': experiment_logger,
        'deterministic': config.training.get('deterministic', False),
        'log_every_n_steps': int(config.training.get('log_every_n_steps', 50)),
        'val_check_interval': float(config.training.get('val_check_interval', 1.0)),
        'enable_checkpointing': True,
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'gradient_clip_val': config.training.get('max_grad_norm', 1.0),
    }
    
    # Add DeepSpeed-specific settings
    if strategy != "auto":
        trainer_kwargs.update({
            'accumulate_grad_batches': config.training.get('gradient_accumulation_steps', 4),
            'sync_batchnorm': False,  # Not needed for single GPU
        })
    
    trainer = pl.Trainer(**trainer_kwargs)
    
    logger.info(f"✅ Trainer created for experiment: {experiment_name}")
    logger.info(f"   - Strategy: {type(strategy).__name__ if strategy != 'auto' else 'auto'}")
    logger.info(f"   - Precision: {config.training.get('precision', 16)}")
    logger.info(f"   - Max epochs: {config.training.get('max_epochs', 100)}")
    
    return trainer, lightning_module


class DomainAdaptationTrainer(LightningTrainer):
    """Extended trainer for domain adaptation between different regions."""
    
    def __init__(self, config: DictConfig, source_checkpoint: str):
        super().__init__(config)
        
        # Load source model weights
        checkpoint = torch.load(source_checkpoint, map_location='cpu')
        if 'state_dict' in checkpoint:
            self.load_state_dict(checkpoint['state_dict'], strict=False)
        
        # Domain adaptation settings
        self.freeze_backbone_epochs = getattr(config.training, 'domain_adaptation', {}).get('freeze_epochs', 5)
        self.adaptation_lr_multiplier = getattr(config.training, 'domain_adaptation', {}).get('lr_multiplier', 10.0)
        
        logger.info(f"Domain adaptation trainer initialized from {source_checkpoint}")
    
    def freeze_backbone(self):
        """Freeze backbone parameters for domain adaptation."""
        for name, param in self.named_parameters():
            if 'domain_adapter' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for joint training."""
        for param in self.parameters():
            param.requires_grad = True
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer with different learning rates for adaptation."""
        
        # Separate parameters for different learning rates
        backbone_params = []
        adapter_params = []
        
        for name, param in self.named_parameters():
            if 'domain_adapter' in name:
                adapter_params.append(param)
            else:
                backbone_params.append(param)
        
        # Different learning rates
        base_lr = self.training_config.optimizer.lr
        adapter_lr = base_lr * self.adaptation_lr_multiplier
        
        optimizer_config = self.training_config.optimizer
        
        if optimizer_config.type.lower() == "adamw":
            optimizer = AdamW([
                {'params': backbone_params, 'lr': base_lr},
                {'params': adapter_params, 'lr': adapter_lr}
            ], weight_decay=optimizer_config.weight_decay)
        else:
            optimizer = AdamW([
                {'params': backbone_params, 'lr': base_lr},
                {'params': adapter_params, 'lr': adapter_lr}
            ], weight_decay=0.01)
        
        # Use same scheduler configuration as base trainer
        scheduler_config = super().configure_optimizers()
        scheduler_config['optimizer'] = optimizer
        
        return scheduler_config
    
    def on_train_epoch_start(self):
        """Handle backbone freezing for domain adaptation."""
        
        if self.current_epoch < self.freeze_backbone_epochs:
            if self.current_epoch == 0:
                self.freeze_backbone()
                logger.info(f"Backbone frozen for first {self.freeze_backbone_epochs} epochs")
        elif self.current_epoch == self.freeze_backbone_epochs:
            self.unfreeze_backbone()
            logger.info("Backbone unfrozen for joint training")
        
        # Call parent method
        super().on_train_epoch_start()


def create_domain_adaptation_trainer(config: DictConfig,
                                   source_checkpoint: str,
                                   experiment_name: str) -> Tuple[pl.Trainer, DomainAdaptationTrainer]:
    """
    Create domain adaptation trainer for transfer learning.
    
    Args:
        config: Training configuration
        source_checkpoint: Path to source domain checkpoint
        experiment_name: Experiment name
        
    Returns:
        Tuple of (pl.Trainer, DomainAdaptationTrainer)
    """
    
    # Initialize domain adaptation model
    lightning_module = DomainAdaptationTrainer(config, source_checkpoint)
    
    # Use same trainer setup as regular trainer
    trainer, _ = create_trainer(config, experiment_name)
    
    return trainer, lightning_module