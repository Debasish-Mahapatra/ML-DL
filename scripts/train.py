"""
Main training script for lightning prediction model.
"""

import os
import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime

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
        "outputs/reports"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created directories for experiment: {experiment_name}")

def validate_config(config):
    """Validate configuration parameters."""
    
    required_paths = [
        config.data.root_dir,
        config.data.splits_dir
    ]
    
    for path in required_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")
    
    # Validate GPU availability if specified
    if config.training.accelerator == "gpu" and not torch.cuda.is_available():
        logger.warning("GPU specified but not available, falling back to CPU")
        config.training.accelerator = "cpu"
        config.training.devices = 1
        config.training.precision = 32
    
    # Validate batch size for available memory
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb < 10 and config.data.batch_size > 4:
            logger.warning(f"GPU memory ({gpu_memory_gb:.1f}GB) is limited, reducing batch size")
            config.data.batch_size = min(config.data.batch_size, 2)
    
    logger.info("Configuration validation completed")

def main():
    """Main training function."""
    
    parser = argparse.ArgumentParser(description="Train Lightning Prediction Model")
    parser.add_argument("--config", type=str, default="config", 
                       help="Path to config directory")
    parser.add_argument("--experiment-name", type=str, 
                       default=f"lightning_pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help="Name for this experiment")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--fast-dev-run", action="store_true",
                       help="Run in fast development mode (single batch)")
    parser.add_argument("--overfit-batches", type=float, default=0.0,
                       help="Overfit on a fraction of data for debugging")
    parser.add_argument("--logger", type=str, choices=["tensorboard", "wandb"], 
                       default="tensorboard", help="Logger type")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true",
                       help="Validate setup without training")
    
    args = parser.parse_args()
    
    # Set random seed
    seed_everything(args.seed, workers=True)
    
    # Load configuration
    try:
        config = get_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Validate configuration
    try:
        validate_config(config)
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return 1
    
    # Setup experiment directories
    setup_directories(args.experiment_name)
    
    # Save configuration for this experiment
    config_save_path = f"experiments/{args.experiment_name}/config.yaml"
    OmegaConf.save(config, config_save_path)
    logger.info(f"Saved experiment configuration to {config_save_path}")
    
    # Initialize data module
    try:
        logger.info("Initializing data module...")
        datamodule = LightningDataModule(config)
        datamodule.setup("fit")
        logger.info(f"Data module initialized with {len(datamodule.train_files)} train files")
        
        # Log data statistics
        sample_batch = datamodule.get_sample_batch("train")
        logger.info(f"Sample batch shapes:")
        for key, value in sample_batch.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: {value.shape}")
        
    except Exception as e:
        logger.error(f"Failed to initialize data module: {e}")
        return 1
    
    # Create model and trainer
    try:
        logger.info("Creating trainer and model...")
        trainer, lightning_module = create_trainer(
            config, 
            args.experiment_name,
            args.logger
        )
        
        # Log model information
        model_info = lightning_module.model.get_model_info()
        logger.info(f"Model created:")
        logger.info(f"  Total parameters: {model_info['total_parameters']:,}")
        logger.info(f"  Model size: {model_info['model_size_mb']:.1f} MB")
        logger.info(f"  CAPE-only mode: {model_info['cape_only_mode']}")
        
    except Exception as e:
        logger.error(f"Failed to create trainer/model: {e}")
        return 1
    
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
        logger.info("Testing data loading...")
        try:
            train_loader = datamodule.train_dataloader()
            logger.info("Train dataloader created successfully")
            
            # Try to get first batch
            logger.info("Attempting to get first batch...")
            first_batch = next(iter(train_loader))
            logger.info(f"First batch loaded successfully")
            logger.info(f"Batch keys: {list(first_batch.keys())}")
            for key, value in first_batch.items():
                logger.info(f"  {key}: {value.shape} ({value.dtype})")
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            return 1
        
        # DEBUG: Test model forward pass
        logger.info("Testing model forward pass...")
        try:
            lightning_module.eval()
            with torch.no_grad():
                output = lightning_module(
                    first_batch['cape'],
                    first_batch['terrain'],
                    first_batch.get('era5', None)
                )
                logger.info("Model forward pass successful")
                logger.info(f"Output shape: {output['lightning_prediction'].shape}")
        except Exception as e:
            logger.error(f"Model forward pass failed: {e}")
            return 1
        
        if args.resume_from:
            logger.info(f"Resuming training from {args.resume_from}")
            trainer.fit(lightning_module, datamodule, ckpt_path=args.resume_from)
        else:
            trainer.fit(lightning_module, datamodule)
        
        logger.info("Training completed successfully")
        
        # Save final model info
        final_metrics = lightning_module.metric_tracker.get_best_metrics()
        logger.info("Best validation metrics:")
        for metric_name, (best_value, best_epoch) in final_metrics.items():
            logger.info(f"  {metric_name}: {best_value:.4f} (epoch {best_epoch})")
        
        # Test evaluation if test data available
        if hasattr(datamodule, 'test_files') and datamodule.test_files:
            logger.info("Running final evaluation on test set...")
            test_results = trainer.test(lightning_module, datamodule)
            logger.info(f"Test results: {test_results}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    
    logger.info(f"Experiment {args.experiment_name} completed successfully")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
