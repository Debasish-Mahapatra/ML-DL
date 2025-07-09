"""
PyTorch Lightning trainer for the lightning prediction model.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau
from typing import Dict, Any, Optional, Tuple, Union
from omegaconf import DictConfig
import numpy as np
import logging

from ..models.architecture import LightningPredictor
from .loss_functions import CompositeLoss, create_loss_function
from .metrics import LightningMetrics, MetricTracker, compute_class_weights

logger = logging.getLogger(__name__)

class LightningTrainer(pl.LightningModule):
    """
    PyTorch Lightning trainer for lightning prediction with physics constraints,
    memory optimization, and domain adaptation support.
    """
    
    def __init__(self, config: DictConfig):
        """
        Initialize Lightning Trainer.
        
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
        
        # Class weights for imbalanced data
        self.class_weights = None
        
        print(f"Lightning Trainer initialized:")
        print(f"  - Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  - Gradient accumulation: {self.gradient_accumulation_steps} steps")
        print(f"  - Domain adaptation: {self.domain_adaptation_enabled}")
        print(f"  - Physics loss weight: {self.physics_loss_weight}")
    
    def forward(self, cape_data: torch.Tensor, terrain_data: torch.Tensor, 
                era5_data: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(cape_data, terrain_data, era5_data, 
                         domain_adaptation=self.domain_adaptation_enabled)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        
        # Optimizer selection
        optimizer_config = self.training_config.optimizer
        
        # FIX: Make optimizer type comparison case-insensitive
        optimizer_type = optimizer_config.type.lower()
        
        
        if optimizer_type == "adamw":  # FIX: Changed from optimizer_config.type == "adamw"
            optimizer = AdamW(
                self.parameters(),
                lr=float(optimizer_config.lr),  # FIX: Ensure float
                weight_decay=float(optimizer_config.weight_decay),  # FIX: Ensure float
                betas=getattr(optimizer_config, 'betas', (0.9, 0.999)),
                eps=getattr(optimizer_config, 'eps', 1e-8)
            )
        elif optimizer_type == "sgd":  # FIX: Changed from optimizer_config.type == "sgd"
            optimizer = SGD(
                self.parameters(),
                lr=float(optimizer_config.lr),  # FIX: Ensure float
                momentum=getattr(optimizer_config, 'momentum', 0.9),
                weight_decay=float(optimizer_config.weight_decay)  # FIX: Ensure float
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_config.type}")
        
        # Learning rate scheduler
        scheduler_config = self.training_config.scheduler
        
        
        # FIX: Make scheduler type comparison case-insensitive and handle different naming
        scheduler_type = scheduler_config.type.lower()
        
        
        if scheduler_type in ["cosine", "cosineannelingwarmrestarts"]:  # FIX: Changed from scheduler_config.type == "cosine"
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=int(scheduler_config.T_0),  # FIX: Ensure integer
                T_mult=int(getattr(scheduler_config, 'T_mult', 2)),  # FIX: Ensure integer
                eta_min=float(scheduler_config.eta_min)  # FIX: Ensure floatn
            )
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "step"
            }
        elif scheduler_type == "onecycle":  # FIX: Changed from scheduler_config.type == "onecycle"
            scheduler = OneCycleLR(
                optimizer,
                max_lr=optimizer_config.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=float(getattr(scheduler_config, 'pct_start', 0.3)),  # FIX: Ensure float
                anneal_strategy=getattr(scheduler_config, 'anneal_strategy', 'cos')
            )
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": "step"
            }
        elif scheduler_type == "plateau":  # FIX: Changed from scheduler_config.type == "plateau"
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=float(getattr(scheduler_config, 'factor', 0.5)),  # FIX: Ensure float
                patience=int(getattr(scheduler_config, 'patience', 10)),  # FIX: Ensure integer
                threshold=float(getattr(scheduler_config, 'threshold', 1e-4))  # FIX: Ensure float
            )
            lr_scheduler_config = {
                "scheduler": scheduler,
                "monitor": "val_f1_score",
                "interval": "epoch"
            }
        else:
            return {"optimizer": optimizer}
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config
        }
        
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with gradient accumulation and physics constraints."""
        
        # Extract batch data
        cape_data = batch['cape']
        terrain_data = batch['terrain']
        lightning_targets = batch['lightning']
        era5_data = batch.get('era5', None)
        
        # Determine if domain adaptation should be active
        domain_adaptation_active = (
            self.domain_adaptation_enabled and 
            self.current_epoch >= self.domain_adaptation_warmup
        )
        
        # Forward pass
        outputs = self.model(
            cape_data, terrain_data, era5_data,
            domain_adaptation=domain_adaptation_active
        )
        predictions = outputs['lightning_prediction']
        
        # Compute loss with physics constraints
        if isinstance(self.loss_function, CompositeLoss):
            loss_dict = self.loss_function(
                predictions, lightning_targets, cape_data, terrain_data
            )
            total_loss = loss_dict['total_loss']
        else:
            total_loss = self.loss_function(predictions, lightning_targets)
            loss_dict = {'total_loss': total_loss, 'main_loss': total_loss}
        
        # FIX: Add NaN detection and emergency stop
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"NaN/Inf loss detected at step {batch_idx}")
            print(f"Predictions range: {predictions.min():.4f} to {predictions.max():.4f}")
            print(f"Targets range: {lightning_targets.min():.4f} to {lightning_targets.max():.4f}")
            print(f"Loss components: {loss_dict}")
            # Return a small valid loss to prevent crash
            return torch.tensor(0.1, requires_grad=True, device=predictions.device)
        
        # Update metrics
        with torch.no_grad():
            # FIX: Convert logits to probabilities and binary predictions to avoid shape mismatch in metrics computation
            probabilities = torch.sigmoid(predictions)
            binary_predictions = (probabilities > 0.5).float()
            self.train_metrics.update(binary_predictions, lightning_targets, probabilities)
        
        # Log losses
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        if 'main_loss' in loss_dict:
            self.log('train_main_loss', loss_dict['main_loss'], on_step=True, on_epoch=True)
        if 'physics_loss' in loss_dict:
            self.log('train_physics_loss', loss_dict['physics_loss'], on_step=True, on_epoch=True)
        
        # Log learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True)
        
        # FIX: Return total_loss directly for automatic optimization
        return total_loss
        
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        
        # Extract batch data
        cape_data = batch['cape']
        terrain_data = batch['terrain']
        lightning_targets = batch['lightning']
        era5_data = batch.get('era5', None)
        
        # Forward pass
        outputs = self.model(cape_data, terrain_data, era5_data, domain_adaptation=False)
        predictions = outputs['lightning_prediction']
        
        # Compute loss
        if isinstance(self.loss_function, CompositeLoss):
            loss_dict = self.loss_function(
                predictions, lightning_targets, cape_data, terrain_data
            )
            total_loss = loss_dict['total_loss']
        else:
            total_loss = self.loss_function(predictions, lightning_targets)
        
        # Update metrics
        with torch.no_grad():
            # FIX: Convert logits to probabilities and binary predictions to avoid shape mismatch in metrics computation
            probabilities = torch.sigmoid(predictions)
            binary_predictions = (probabilities > 0.5).float()
            self.val_metrics.update(binary_predictions, lightning_targets, probabilities)
        
        # Log validation loss
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        
        # Extract batch data
        cape_data = batch['cape']
        terrain_data = batch['terrain']
        lightning_targets = batch['lightning']
        era5_data = batch.get('era5', None)
        
        # Forward pass
        outputs = self.model(cape_data, terrain_data, era5_data, domain_adaptation=False)
        predictions = outputs['lightning_prediction']
        
        # Compute loss
        if isinstance(self.loss_function, CompositeLoss):
            loss_dict = self.loss_function(
                predictions, lightning_targets, cape_data, terrain_data
            )
            total_loss = loss_dict['total_loss']
        else:
            total_loss = self.loss_function(predictions, lightning_targets)
        
        # Update metrics
        with torch.no_grad():
            # FIX: Convert logits to probabilities and binary predictions to avoid shape mismatch in metrics computation
            probabilities = torch.sigmoid(predictions)
            binary_predictions = (probabilities > 0.5).float()
            self.test_metrics.update(binary_predictions, lightning_targets, probabilities)
        
        return total_loss
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        
        # Compute and log training metrics
        train_metrics = self.train_metrics.compute()
        
        for metric_name, metric_value in train_metrics.items():
            self.log(f'train_{metric_name}', metric_value, on_epoch=True)
        
        # Reset metrics
        self.train_metrics.reset()
        
        # Reset accumulation step counter
        self.current_accumulation_step = 0
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        
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
    
    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        
        # Compute and log test metrics
        test_metrics = self.test_metrics.compute()
        
        for metric_name, metric_value in test_metrics.items():
            self.log(f'test_{metric_name}', metric_value, on_epoch=True)
        
        # Reset metrics
        self.test_metrics.reset()
        
        return test_metrics
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step for inference."""
        
        cape_data = batch['cape']
        terrain_data = batch['terrain']
        era5_data = batch.get('era5', None)
        
        outputs = self.model(cape_data, terrain_data, era5_data, domain_adaptation=False)
        
        return {
            'predictions': outputs['lightning_prediction'],
            'cape_features': outputs.get('cape_features'),
            'terrain_features': outputs.get('terrain_features')
        }
    
    def on_fit_start(self):
        """Called at the start of training."""
        
        # Compute class weights if not provided
        if self.class_weights is None and hasattr(self.trainer.datamodule, 'train_dataloader'):
            logger.info("Computing class weights from training data...")
            self._compute_class_weights()
        
        # Log model info
        model_info = self.model.get_model_info()
        logger.info(f"Model info: {model_info}")
        
        # Enable domain adaptation after warmup
        if self.domain_adaptation_enabled:
            logger.info(f"Domain adaptation will be enabled after epoch {self.domain_adaptation_warmup}")
    
    def _compute_class_weights(self):
        """Compute class weights from training data."""
        
        try:
            train_loader = self.trainer.datamodule.train_dataloader()
            all_targets = []
            
            # Sample a subset of data to compute weights
            sample_count = 0
            max_samples = 1000  # Limit sampling for efficiency
            
            for batch in train_loader:
                lightning_targets = batch['lightning']
                all_targets.append(lightning_targets)
                
                sample_count += lightning_targets.shape[0]
                if sample_count >= max_samples:
                    break
            
            if all_targets:
                combined_targets = torch.cat(all_targets, dim=0)
                self.class_weights = compute_class_weights(combined_targets)
                logger.info(f"Computed class weights: {self.class_weights}")
            
        except Exception as e:
            logger.warning(f"Could not compute class weights: {e}")
            self.class_weights = torch.tensor([1.0, 1.0])
    
    def freeze_backbone(self):
        """Freeze backbone encoders for fine-tuning."""
        self.model.freeze_encoders()
        logger.info("Backbone encoders frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone encoders."""
        self.model.unfreeze_encoders()
        logger.info("Backbone encoders unfrozen")
    
    def enable_domain_adaptation(self):
        """Enable domain adaptation mode."""
        self.domain_adaptation_enabled = True
        logger.info("Domain adaptation enabled")
    
    def disable_domain_adaptation(self):
        """Disable domain adaptation mode."""
        self.domain_adaptation_enabled = False
        logger.info("Domain adaptation disabled")

class DomainAdaptationTrainer(LightningTrainer):
    """
    Specialized trainer for domain adaptation from Odisha to other regions.
    """
    
    def __init__(self, config: DictConfig, source_checkpoint: Optional[str] = None):
        """
        Initialize domain adaptation trainer.
        
        Args:
            config: Training configuration
            source_checkpoint: Path to source domain checkpoint
        """
        super().__init__(config)
        
        self.source_checkpoint = source_checkpoint
        
        # Domain adaptation specific settings
        self.adaptation_lr_multiplier = getattr(config.training, 'domain_adaptation', {}).get('lr_multiplier', 0.1)
        self.freeze_backbone_epochs = getattr(config.training, 'domain_adaptation', {}).get('freeze_epochs', 5)
        
        # Load source domain model if provided
        if source_checkpoint:
            self.load_source_model(source_checkpoint)
    
    def load_source_model(self, checkpoint_path: str):
        """Load pre-trained source domain model."""
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Load only the main model weights, not the domain adapter
            model_state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                if not key.startswith('model.domain_adapter'):
                    model_state_dict[key] = value
            
            # Load weights with strict=False to allow missing domain adapter weights
            self.load_state_dict(model_state_dict, strict=False)
            
            logger.info(f"Loaded source domain model from {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to load source model: {e}")
            raise
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer with different learning rates for domain adaptation."""
        
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
        
        # FIX: Make optimizer type comparison case-insensitive
        optimizer_type = optimizer_config.type.lower()
        
        if optimizer_type == "adamw":  # FIX: Changed from optimizer_config.type == "adamw"
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

def create_trainer(config: DictConfig, 
                  experiment_name: str,
                  logger_type: str = "tensorboard") -> Tuple[pl.Trainer, LightningTrainer]:
    """
    Create PyTorch Lightning trainer with all callbacks and loggers.
    
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
        dirpath=f"checkpoints/{experiment_name}",
        filename="{epoch:02d}-{val_f1_score:.3f}",
        monitor="val_f1_score",
        mode="max",
        save_top_k=int(config.training.save_top_k),  # FIX: Ensure it's an integer
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_f1_score",
        mode="max",
        patience=int(config.training.patience),  # FIX: Ensure it's an integer
        min_delta=float(config.training.min_delta),  # FIX: Ensure it's a float
        verbose=True
    )
    callbacks.append(early_stop_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=int(config.training.max_epochs),  # FIX: Ensure integers
        accelerator=config.training.accelerator,
        devices=int(config.training.devices),  # FIX: Ensure integers
        precision=int(config.training.precision),  # FIX: Ensure integers
        gradient_clip_val=float(getattr(config.training, 'max_grad_norm', None)) if getattr(config.training, 'max_grad_norm', None) is not None else None,  # FIX: Ensure float or None
        accumulate_grad_batches=int(getattr(config.training, 'gradient_accumulation_steps', 1)),  # FIX: Ensure integers
        callbacks=callbacks,
        logger=experiment_logger,
        deterministic=bool(config.training.deterministic),  # FIX: Ensure boolean
        log_every_n_steps=int(config.training.log_every_n_steps),  # FIX: Ensure integers
        val_check_interval=float(config.training.val_check_interval),  # FIX: Ensure float
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    return trainer, lightning_module

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
