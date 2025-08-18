"""
Main training script for lightning prediction model with enhanced physics debugging.
"""

import os
import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime
import gc

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf

from src.utils.config import get_config
from src.data.data_loader import LightningDataModule
from src.training.trainer import create_trainer, LightningTrainer
from src.models.architecture import create_model_from_config

# MEMORY TRACKING IMPORTS - EXISTING
from src.utils.memory_tracker import (
    MemoryTracker, 
    memory_checkpoint, 
    trace_memory_line, 
    start_global_monitoring, 
    stop_global_monitoring,
    MemoryContext
)

# DEBUG UTILITIES IMPORTS - NEW ADDITION
from src.utils.debug_utils import get_debug_manager, debug_print, is_debug_enabled

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories(experiment_name: str):
    """Create necessary directories for the experiment."""
    
    dirs_to_create = [
        f"experiments/{experiment_name}/logs",
        f"experiments/{experiment_name}/checkpoints", 
        f"experiments/{experiment_name}/results",
        "logs",
        "outputs/predictions",
        "outputs/visualizations",
        "outputs/reports",
        # NEW: Physics debugging directories
        f"experiments/{experiment_name}/physics_debug",
        f"experiments/{experiment_name}/cape_analysis"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created directories for experiment: {experiment_name}")

def validate_config(config):
    """Validate configuration parameters with enhanced physics validation."""
    
    required_keys = [
        "training.experiment_name",
        "training.max_epochs",
        "model.name",
        "data.root_dir"
    ]
    
    for key in required_keys:
        if not OmegaConf.select(config, key):
            raise ValueError(f"Required configuration key missing: {key}")
    
    # Enhanced physics validation
    if hasattr(config.training, 'loss') and hasattr(config.training.loss, 'physics_weight'):
        physics_weight = config.training.loss.physics_weight
        if physics_weight < 0 or physics_weight > 1:
            logger.warning(f"Physics weight {physics_weight} may be outside expected range [0,1]")
    
    logger.info("Configuration validation passed")

def log_physics_configuration(config):
    """Log physics configuration for debugging."""
    
    logger.info("=== PHYSICS CONFIGURATION ===")
    
    loss_config = config.training.loss
    logger.info(f"Main physics weight: {loss_config.physics_weight}")
    logger.info(f"Charge separation weight: {loss_config.charge_separation_weight}")
    logger.info(f"Microphysics weight: {loss_config.microphysics_weight}")
    logger.info(f"Terrain consistency weight: {loss_config.terrain_consistency_weight}")
    
    if hasattr(loss_config, 'cape_gradient_weight'):
        logger.info(f"CAPE gradient weight: {loss_config.cape_gradient_weight}")
    if hasattr(loss_config, 'cape_temporal_weight'):
        logger.info(f"CAPE temporal weight: {loss_config.cape_temporal_weight}")
    
    if hasattr(config.training, 'cape_physics'):
        cape_physics = config.training.cape_physics
        logger.info("CAPE thresholds:")
        logger.info(f"  No lightning: {cape_physics.thresholds.no_lightning} J/kg")
        logger.info(f"  Moderate: {cape_physics.thresholds.moderate} J/kg")
        logger.info(f"  High: {cape_physics.thresholds.high} J/kg")
        logger.info(f"  Saturation: {cape_physics.thresholds.saturation} J/kg")
        logger.info(f"Optimal CAPE: {cape_physics.saturation.optimal_cape} J/kg")
    
    logger.info("=== END PHYSICS CONFIGURATION ===")

def analyze_data_distribution(datamodule, debug_manager):
    """Analyze CAPE and lightning data distributions for physics validation."""
    
    if not debug_manager.cape_analysis:
        return
    
    logger.info("=== CAPE DATA ANALYSIS ===")
    
    try:
        # Get a few batches for analysis
        train_loader = datamodule.train_dataloader()
        cape_values = []
        lightning_values = []
        
        for i, batch in enumerate(train_loader):
            if i >= 5:  # Analyze first 5 batches
                break
            
            cape_data = batch['cape'].cpu().numpy()
            lightning_data = batch['lightning'].cpu().numpy()
            
            cape_values.extend(cape_data.flatten())
            lightning_values.extend(lightning_data.flatten())
        
        cape_values = torch.tensor(cape_values)
        lightning_values = torch.tensor(lightning_values)
        
        # Log CAPE statistics
        logger.info(f"CAPE statistics (J/kg):")
        logger.info(f"  Min: {cape_values.min().item():.1f}")
        logger.info(f"  Max: {cape_values.max().item():.1f}")
        logger.info(f"  Mean: {cape_values.mean().item():.1f}")
        logger.info(f"  Std: {cape_values.std().item():.1f}")
        logger.info(f"  Median: {cape_values.median().item():.1f}")
        
        # Log lightning statistics
        lightning_positive = lightning_values.sum().item()
        lightning_total = len(lightning_values)
        lightning_ratio = lightning_positive / lightning_total
        
        logger.info(f"Lightning statistics:")
        logger.info(f"  Positive samples: {lightning_positive:,}")
        logger.info(f"  Total samples: {lightning_total:,}")
        logger.info(f"  Positive ratio: {lightning_ratio:.6f} ({lightning_ratio*100:.4f}%)")
        
        # CAPE-lightning correlation analysis
        cape_with_lightning = cape_values[lightning_values > 0.5]
        cape_without_lightning = cape_values[lightning_values <= 0.5]
        
        if len(cape_with_lightning) > 0:
            logger.info(f"CAPE with lightning:")
            logger.info(f"  Mean: {cape_with_lightning.mean().item():.1f} J/kg")
            logger.info(f"  Std: {cape_with_lightning.std().item():.1f} J/kg")
        
        if len(cape_without_lightning) > 0:
            logger.info(f"CAPE without lightning:")
            logger.info(f"  Mean: {cape_without_lightning.mean().item():.1f} J/kg")
            logger.info(f"  Std: {cape_without_lightning.std().item():.1f} J/kg")
        
        # Physics regime analysis
        if hasattr(train_loader.dataset, 'config') and hasattr(train_loader.dataset.config.training, 'cape_physics'):
            thresholds = train_loader.dataset.config.training.cape_physics.thresholds
            
            no_lightning_count = (cape_values < thresholds.no_lightning).sum().item()
            moderate_count = ((cape_values >= thresholds.no_lightning) & (cape_values < thresholds.moderate)).sum().item()
            high_count = ((cape_values >= thresholds.moderate) & (cape_values < thresholds.high)).sum().item()
            very_high_count = (cape_values >= thresholds.high).sum().item()
            
            logger.info(f"CAPE regime distribution:")
            logger.info(f"  No lightning regime (<{thresholds.no_lightning}): {no_lightning_count:,} ({no_lightning_count/len(cape_values)*100:.2f}%)")
            logger.info(f"  Moderate regime ({thresholds.no_lightning}-{thresholds.moderate}): {moderate_count:,} ({moderate_count/len(cape_values)*100:.2f}%)")
            logger.info(f"  High regime ({thresholds.moderate}-{thresholds.high}): {high_count:,} ({high_count/len(cape_values)*100:.2f}%)")
            logger.info(f"  Very high regime (>{thresholds.high}): {very_high_count:,} ({very_high_count/len(cape_values)*100:.2f}%)")
    
    except Exception as e:
        logger.warning(f"Could not analyze data distribution: {e}")
    
    logger.info("=== END CAPE DATA ANALYSIS ===")

def analyze_model_predictions(model, datamodule, debug_manager, step_number=None):
    """NEW: Analyze model prediction ranges to understand threshold issues."""
    
    if not debug_manager.verbose_logging and not debug_manager.physics_debug:
        return
    
    step_info = f" at step {step_number}" if step_number else ""
    debug_print(f"=== MODEL PREDICTION ANALYSIS{step_info.upper()} ===", "verbose")
    
    try:
        model.eval()
        with torch.no_grad():
            # Get validation batch for analysis
            val_loader = datamodule.val_dataloader()
            sample_batch = next(iter(val_loader))
            
            # Move to same device as model
            device = next(model.parameters()).device
            cape_data = sample_batch['cape'].to(device)
            terrain_data = sample_batch['terrain'].to(device)
            lightning_targets = sample_batch['lightning'].to(device)
            
            # Forward pass
            outputs = model(cape_data, terrain_data)
            predictions_logits = outputs['lightning_prediction']
            predictions_probs = torch.sigmoid(predictions_logits)
            
            # Analyze prediction statistics
            pred_min = predictions_probs.min().item()
            pred_max = predictions_probs.max().item()
            pred_mean = predictions_probs.mean().item()
            pred_std = predictions_probs.std().item()
            pred_median = predictions_probs.median().item()
            
            # Analyze target statistics for comparison
            target_mean = lightning_targets.mean().item()
            target_sum = lightning_targets.sum().item()
            target_total = lightning_targets.numel()
            
            # Log detailed prediction analysis
            debug_print(f"Model prediction statistics{step_info}:", "verbose")
            debug_print(f"  Probability range: {pred_min:.6f} to {pred_max:.6f}", "verbose")
            debug_print(f"  Mean probability: {pred_mean:.6f}", "verbose")
            debug_print(f"  Std probability: {pred_std:.6f}", "verbose")
            debug_print(f"  Median probability: {pred_median:.6f}", "verbose")
            
            debug_print(f"Target statistics for comparison:", "verbose")
            debug_print(f"  Lightning pixels: {target_sum}/{target_total} ({target_mean*100:.4f}%)", "verbose")
            
            # Threshold analysis
            thresholds_to_test = [0.5, 0.1, 0.05, 0.02, 0.01, 0.005]
            debug_print(f"Threshold analysis:", "verbose")
            
            for thresh in thresholds_to_test:
                pred_positive = (predictions_probs > thresh).sum().item()
                pred_ratio = pred_positive / predictions_probs.numel()
                debug_print(f"  Threshold {thresh:0.3f}: {pred_positive}/{predictions_probs.numel()} pixels ({pred_ratio*100:.4f}%)", "verbose")
            
            # Physics-aware analysis
            if debug_manager.physics_debug:
                debug_print(f"Physics-aware prediction analysis:", "physics")
                
                # Analyze predictions by CAPE regime
                cape_data_cpu = cape_data.cpu()
                predictions_cpu = predictions_probs.cpu()
                
                # Define CAPE thresholds
                cape_thresholds = {
                    'low': 1000.0,
                    'moderate': 2500.0,
                    'high': 4000.0
                }
                
                for regime, threshold in cape_thresholds.items():
                    if regime == 'low':
                        mask = cape_data_cpu < threshold
                        regime_name = f"Low CAPE (<{threshold})"
                    elif regime == 'moderate':
                        mask = (cape_data_cpu >= 1000.0) & (cape_data_cpu < threshold)
                        regime_name = f"Moderate CAPE (1000-{threshold})"
                    else:
                        mask = cape_data_cpu >= threshold
                        regime_name = f"High CAPE (>{threshold})"
                    
                    if mask.any():
                        regime_preds = predictions_cpu[mask]
                        regime_mean = regime_preds.mean().item()
                        regime_max = regime_preds.max().item()
                        regime_pixels = mask.sum().item()
                        
                        debug_print(f"  {regime_name}: {regime_pixels} pixels, mean_pred={regime_mean:.6f}, max_pred={regime_max:.6f}", "physics")
            
        model.train()  # Return to training mode
        
    except Exception as e:
        debug_print(f"Could not analyze model predictions: {e}", "verbose")
        model.train()  # Ensure we return to training mode
    
    debug_print("=== END MODEL PREDICTION ANALYSIS ===", "verbose")

@memory_checkpoint("MAIN_TRAINING")
def main():
    """Main training function with enhanced physics debugging."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train lightning prediction model")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                       help="Path to training configuration file")
    parser.add_argument("--model-config", type=str, default="config/model_config.yaml",
                       help="Path to model configuration file")
    parser.add_argument("--data-config", type=str, default="config/data_config.yaml",
                       help="Path to data configuration file")
    parser.add_argument("--experiment-name", type=str, default=None,
                       help="Experiment name (overrides config)")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                       help="Validate setup without training")
    parser.add_argument("--fast-dev-run", action="store_true",
                       help="Fast development run (single batch)")
    parser.add_argument("--overfit-batches", type=int, default=0,
                       help="Overfit on N batches for debugging")
    # NEW: Physics debugging arguments
    parser.add_argument("--debug-physics", action="store_true",
                       help="Enable detailed physics constraint debugging")
    parser.add_argument("--analyze-cape", action="store_true",
                       help="Analyze CAPE data distribution before training")
    
    args = parser.parse_args()
    
    try:
        # Get configuration - FIXED: Use correct function signature
        config = get_config(args.config.replace('/training_config.yaml', ''))
        
        # Override experiment name if provided
        if args.experiment_name:
            config.training.experiment_name = args.experiment_name
        
        # Override debug settings if specified via command line
        if args.debug_physics:
            if not hasattr(config.training, 'debug'):
                config.training.debug = {}
            config.training.debug.physics_debug = True
            
        if args.analyze_cape:
            if not hasattr(config.training, 'debug'):
                config.training.debug = {}
            config.training.debug.cape_analysis = True
        
        # Initialize debug manager
        debug_manager = get_debug_manager(config.training.debug if hasattr(config.training, 'debug') else {})
        
        if debug_manager.verbose_logging:
            debug_print("Starting lightning prediction training with enhanced physics", "verbose")
        
        # Log physics configuration
        if debug_manager.physics_debug:
            log_physics_configuration(config)
        
        # Set random seed
        seed_everything(config.training.seed, workers=True)
        
        # Validate configuration
        validate_config(config)
        
        # Setup directories
        setup_directories(config.training.experiment_name)
        
        # Initialize memory tracking
        memory_tracker = None
        if debug_manager.memory_tracking:
            memory_tracker = start_global_monitoring()
            debug_print("Memory monitoring started", "memory")
        
        # Initialize data module
        try:
            with MemoryContext("DATAMODULE_INIT"):
                datamodule = LightningDataModule(config)
                datamodule.setup()
            
            if debug_manager.memory_tracking:
                debug_manager.conditional_trace_memory("AFTER_DATAMODULE_SETUP")
            
            logger.info("Data module initialized successfully")
            
        except Exception as e:
            if debug_manager.memory_tracking and memory_tracker:
                memory_tracker.log_current_memory("DATAMODULE_ERROR")
            logger.error(f"Failed to initialize data module: {e}")
            return 1
        
        # Analyze data distribution for physics validation
        if debug_manager.cape_analysis:
            analyze_data_distribution(datamodule, debug_manager)
        
        # Create trainer and model
        try:
            with MemoryContext("TRAINER_MODEL_INIT"):
                trainer, lightning_module = create_trainer(config, config.training.experiment_name)
            
            if debug_manager.memory_tracking:
                debug_manager.conditional_trace_memory("AFTER_TRAINER_MODEL_CREATION")
            
            logger.info("Trainer and model created successfully")
            
            # Log model architecture summary
            if debug_manager.verbose_logging:
                total_params = sum(p.numel() for p in lightning_module.model.parameters())
                trainable_params = sum(p.numel() for p in lightning_module.model.parameters() if p.requires_grad)
                debug_print(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}", "verbose")
            
        except Exception as e:
            if debug_manager.memory_tracking and memory_tracker:
                memory_tracker.log_current_memory("TRAINER_MODEL_ERROR")
            logger.error(f"Failed to create trainer/model: {e}")
            return 1
        
        # NEW: Analyze initial model predictions (before training)
        if debug_manager.verbose_logging or debug_manager.physics_debug:
            analyze_model_predictions(lightning_module, datamodule, debug_manager, step_number="INITIAL")
        
        # Dry run check
        if args.dry_run:
            logger.info("Dry run completed successfully - setup is valid")
            return 0
        
        # Fast development run
        if args.fast_dev_run:
            trainer.fast_dev_run = True
            logger.info("Running in fast development mode")
        
        # Overfit batches for debugging
        if args.overfit_batches > 0:
            trainer.overfit_batches = args.overfit_batches
            logger.info(f"Overfitting on {args.overfit_batches} batches for debugging")
        
        # Training
        try:
            logger.info("Starting training...")
            
            # DEBUG: Test data loading before training
            if debug_manager.verbose_logging:
                debug_print("Testing data loading...", "verbose")
                
            try:
                with MemoryContext("TRAIN_DATALOADER_TEST"):
                    train_loader = datamodule.train_dataloader()
                
                if debug_manager.verbose_logging:
                    debug_print("Train dataloader created successfully", "verbose")
                
                if debug_manager.memory_tracking:
                    debug_manager.conditional_trace_memory("AFTER_DATALOADER_CREATION")
                
                # Try to get first batch
                if debug_manager.batch_info:
                    debug_print("Attempting to get first batch...", "batch")
                    
                    with MemoryContext("FIRST_BATCH_LOAD"):
                        first_batch = next(iter(train_loader))
                    
                    debug_print(f"First batch loaded successfully", "batch")
                    debug_print(f"Batch keys: {list(first_batch.keys())}", "batch")
                    
                    for key, value in first_batch.items():
                        debug_print(f"  {key}: {value.shape} ({value.dtype})", "batch")
                    
                    # NEW: CAPE-specific analysis for first batch
                    if debug_manager.physics_debug and 'cape' in first_batch:
                        cape_batch = first_batch['cape']
                        cape_stats = {
                            'min': cape_batch.min().item(),
                            'max': cape_batch.max().item(),
                            'mean': cape_batch.mean().item(),
                            'std': cape_batch.std().item()
                        }
                        debug_print(f"First batch CAPE stats: {cape_stats}", "physics")
                        
                        if 'lightning' in first_batch:
                            lightning_batch = first_batch['lightning']
                            lightning_pos = (lightning_batch > 0.5).sum().item()
                            lightning_total = lightning_batch.numel()
                            debug_print(f"First batch lightning: {lightning_pos}/{lightning_total} positive", "physics")
                else:
                    # Just test loading without detailed info
                    with MemoryContext("FIRST_BATCH_LOAD"):
                        first_batch = next(iter(train_loader))
                    logger.info("First batch loaded successfully")
                
                if debug_manager.memory_tracking:
                    debug_manager.conditional_trace_memory("AFTER_FIRST_BATCH_TEST")
                    
            except Exception as e:
                logger.error(f"Data loading test failed: {e}")
                import traceback
                logger.error(f"Detailed data loading error: {traceback.format_exc()}")
                return 1
            
            # Start actual training
            if debug_manager.memory_tracking:
                debug_manager.conditional_trace_memory("BEFORE_TRAINING_START")
            
            # NEW: Log training start with physics info
            if debug_manager.physics_debug:
                debug_print("Training started with enhanced CAPE physics constraints", "physics")
            
            trainer.fit(lightning_module, datamodule, ckpt_path=args.resume_from)
            
            if debug_manager.memory_tracking:
                debug_manager.conditional_trace_memory("AFTER_TRAINING_COMPLETE")
            
            # NEW: Analyze final model predictions (after training)
            if debug_manager.verbose_logging or debug_manager.physics_debug:
                analyze_model_predictions(lightning_module, datamodule, debug_manager, step_number="FINAL")
            
        except Exception as e:
            if debug_manager.memory_tracking and memory_tracker:
                memory_tracker.log_current_memory("TRAINING_ERROR")
            logger.error(f"Training failed: {e}")
            import traceback
            logger.error(f"Detailed training error: {traceback.format_exc()}")
            return 1
        
        # Training completed successfully
        logger.info("Training completed successfully!")
        
        if debug_manager.verbose_logging:
            debug_print("Training pipeline completed with enhanced physics", "verbose")
        
        # NEW: Final physics summary
        if debug_manager.physics_debug:
            debug_print("Enhanced CAPE physics training completed successfully", "physics")
            logger.info("Check logs for physics constraint performance during training")
        
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        import traceback
        logger.error(f"Detailed main error: {traceback.format_exc()}")
        return 1
        
    finally:
        # Cleanup
        if 'debug_manager' in locals() and debug_manager and debug_manager.memory_tracking and 'memory_tracker' in locals() and memory_tracker:
            stop_global_monitoring()
            debug_print("Memory monitoring stopped", "memory")
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)