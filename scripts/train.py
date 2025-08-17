"""
Main training script for lightning prediction model.
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

@memory_checkpoint("MAIN_TRAINING")
def main():
    """Main training function with conditional debug tracing."""
    
    # Start conditional memory monitoring - MODIFIED
    memory_tracker = None
    debug_manager = None
    
    try:
        # Parse arguments first
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
        parser.add_argument("--debug", action="store_true",
                           help="Enable debug mode (overrides config)")
        
        args = parser.parse_args()
        
        # Override debug mode via command line - NEW
        if args.debug:
            os.environ['LIGHTNING_DEBUG'] = 'true'
            debug_print("Debug mode enabled via command line", "general")
        
        # Load configuration first - MODIFIED
        try:
            config = get_config(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return 1
        
        # Initialize debug manager with config - NEW
        debug_manager = get_debug_manager(config)
        
        # Conditional memory tracking - MODIFIED
        if debug_manager.memory_tracking:
            memory_tracker = start_global_monitoring()
            debug_manager.conditional_trace_memory("START_OF_MAIN")
        
        if debug_manager.verbose_logging:
            debug_print("Starting lightning prediction training pipeline", "verbose")
        
        # Set random seed
        seed_everything(args.seed, workers=True)
        if debug_manager.memory_tracking:
            debug_manager.conditional_trace_memory("AFTER_SEED_SETTING")
        
        # Validate configuration
        try:
            validate_config(config)
            if debug_manager.memory_tracking:
                debug_manager.conditional_trace_memory("AFTER_CONFIG_VALIDATION")
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return 1
        
        # Setup directories
        setup_directories(args.experiment_name)
        if debug_manager.memory_tracking:
            debug_manager.conditional_trace_memory("AFTER_DIRECTORY_SETUP")
        
        # Initialize data module - CORRECTED
        try:
            if debug_manager.verbose_logging:
                debug_print("Initializing data module", "verbose")
            
            with MemoryContext("DATAMODULE_INIT"):
                datamodule = LightningDataModule(config)
            
            if debug_manager.verbose_logging:
                debug_print("Data module initialized, calling setup...", "verbose")
            
            # CRITICAL FIX: Call setup BEFORE trying to use dataloaders
            with MemoryContext("DATAMODULE_SETUP"):
                datamodule.setup("fit")
            
            if debug_manager.verbose_logging:
                debug_print("Data module setup completed successfully", "verbose")
                debug_print(f"Train files loaded: {len(datamodule.train_files) if datamodule.train_files else 'None'}", "verbose")
                debug_print(f"Val files loaded: {len(datamodule.val_files) if datamodule.val_files else 'None'}", "verbose")
            
            if debug_manager.memory_tracking:
                debug_manager.conditional_trace_memory("AFTER_DATAMODULE_INIT_AND_SETUP")
                
        except Exception as e:
            if debug_manager.memory_tracking and memory_tracker:
                memory_tracker.log_current_memory("DATAMODULE_ERROR")
            logger.error(f"Failed to initialize data module: {e}")
            # Add more detailed error info
            import traceback
            logger.error(f"Detailed error: {traceback.format_exc()}")
            return 1
        
        # Create trainer and model
        try:
            if debug_manager.verbose_logging:
                debug_print("Creating trainer and model", "verbose")
            
            with MemoryContext("TRAINER_MODEL_CREATION"):
                trainer, lightning_module = create_trainer(config, args.experiment_name, args.logger)
            
            # Log model info conditionally - MODIFIED
            model_info = lightning_module.model.get_model_info()
            logger.info(f"Model created:")
            logger.info(f"  Total parameters: {model_info['total_parameters']:,}")
            logger.info(f"  Model size: {model_info['model_size_mb']:.1f} MB")
            logger.info(f"  CAPE-only mode: {model_info['cape_only_mode']}")
            
            if debug_manager.memory_tracking:
                debug_manager.conditional_trace_memory("AFTER_MODEL_CREATION")
            
        except Exception as e:
            if debug_manager.memory_tracking and memory_tracker:
                memory_tracker.log_current_memory("TRAINER_MODEL_ERROR")
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
            
            # DEBUG: Test data loading before training - CORRECTED
            if debug_manager.verbose_logging:
                debug_print("Testing data loading...", "verbose")
                
            try:
                # Now datamodule.setup() has been called, so this should work
                with MemoryContext("TRAIN_DATALOADER_TEST"):
                    train_loader = datamodule.train_dataloader()
                
                if debug_manager.verbose_logging:
                    debug_print("Train dataloader created successfully", "verbose")
                
                if debug_manager.memory_tracking:
                    debug_manager.conditional_trace_memory("AFTER_DATALOADER_CREATION")
                
                # Try to get first batch - MODIFIED
                if debug_manager.batch_info:
                    debug_print("Attempting to get first batch...", "batch")
                    
                    with MemoryContext("FIRST_BATCH_LOAD"):
                        first_batch = next(iter(train_loader))
                    
                    debug_print(f"First batch loaded successfully", "batch")
                    debug_print(f"Batch keys: {list(first_batch.keys())}", "batch")
                    
                    for key, value in first_batch.items():
                        debug_print(f"  {key}: {value.shape} ({value.dtype})", "batch")
                else:
                    # Just test loading without detailed info
                    with MemoryContext("FIRST_BATCH_LOAD"):
                        first_batch = next(iter(train_loader))
                    logger.info("First batch loaded successfully")
                
                if debug_manager.memory_tracking:
                    debug_manager.conditional_trace_memory("AFTER_FIRST_BATCH_TEST")
                    
            except Exception as e:
                logger.error(f"Data loading test failed: {e}")
                # Add more detailed error info
                import traceback
                logger.error(f"Detailed data loading error: {traceback.format_exc()}")
                return 1
            
            # Start actual training
            if debug_manager.memory_tracking:
                debug_manager.conditional_trace_memory("BEFORE_TRAINING_START")
            
            trainer.fit(lightning_module, datamodule, ckpt_path=args.resume_from)
            
            if debug_manager.memory_tracking:
                debug_manager.conditional_trace_memory("AFTER_TRAINING_COMPLETE")
            
        except Exception as e:
            if debug_manager.memory_tracking and memory_tracker:
                memory_tracker.log_current_memory("TRAINING_ERROR")
            logger.error(f"Training failed: {e}")
            # Add more detailed error info
            import traceback
            logger.error(f"Detailed training error: {traceback.format_exc()}")
            return 1
        
        # Training completed successfully
        logger.info("Training completed successfully!")
        
        if debug_manager.verbose_logging:
            debug_print("Training pipeline completed", "verbose")
        
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        # Add more detailed error info
        import traceback
        logger.error(f"Detailed main error: {traceback.format_exc()}")
        return 1
        
    finally:
        # Cleanup
        if debug_manager and debug_manager.memory_tracking and memory_tracker:
            stop_global_monitoring()
            debug_print("Memory monitoring stopped", "memory")
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)