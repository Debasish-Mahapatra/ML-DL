"""
Domain adaptation script for transferring from Odisha to other regions.
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
from pytorch_lightning.utilities.seed import seed_everything

from src.utils.config import get_config
from src.data.data_loader import LightningDataModule
from src.training.trainer import create_domain_adaptation_trainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main domain adaptation function."""
    
    parser = argparse.ArgumentParser(description="Domain Adaptation for Lightning Prediction")
    parser.add_argument("--source-checkpoint", type=str, required=True,
                       help="Path to source domain checkpoint (Odisha)")
    parser.add_argument("--target-config", type=str, required=True,
                       help="Configuration for target domain data")
    parser.add_argument("--experiment-name", type=str,
                       default=f"domain_adaptation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help="Experiment name")
    parser.add_argument("--max-epochs", type=int, default=20,
                       help="Maximum training epochs")
    parser.add_argument("--freeze-epochs", type=int, default=5,
                       help="Epochs to freeze backbone")
    parser.add_argument("--adaptation-lr", type=float, default=0.0001,
                       help="Learning rate for adaptation layers")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    seed_everything(args.seed, workers=True)
    
    # Validate source checkpoint
    if not Path(args.source_checkpoint).exists():
        logger.error(f"Source checkpoint not found: {args.source_checkpoint}")
        return 1
    
    # Load target domain configuration
    try:
        config = get_config(args.target_config)
        
        # Override domain adaptation settings
        if not hasattr(config.training, 'domain_adaptation'):
            config.training.domain_adaptation = {}
        
        config.training.domain_adaptation.enabled = True
        config.training.domain_adaptation.freeze_epochs = args.freeze_epochs
        config.training.domain_adaptation.lr_multiplier = 0.1
        config.training.max_epochs = args.max_epochs
        
        logger.info(f"Loaded target domain configuration")
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Setup target domain data
    try:
        logger.info("Setting up target domain data...")
        datamodule = LightningDataModule(config)
        datamodule.setup("fit")
        
        logger.info(f"Target domain data: {len(datamodule.train_files)} train files")
        
    except Exception as e:
        logger.error(f"Failed to setup target domain data: {e}")
        return 1
    
    # Create domain adaptation trainer
    try:
        logger.info("Creating domain adaptation trainer...")
        trainer, lightning_module = create_domain_adaptation_trainer(
            config,
            args.source_checkpoint,
            args.experiment_name
        )
        
        logger.info("Domain adaptation trainer created")
        
    except Exception as e:
        logger.error(f"Failed to create domain adaptation trainer: {e}")
        return 1
    
    # Run domain adaptation training
    try:
        logger.info("Starting domain adaptation training...")
        trainer.fit(lightning_module, datamodule)
        
        logger.info("Domain adaptation completed")
        
        # Test on target domain
        if hasattr(datamodule, 'test_files') and datamodule.test_files:
            logger.info("Evaluating on target domain test set...")
            test_results = trainer.test(lightning_module, datamodule)
            logger.info(f"Target domain test results: {test_results}")
        
    except Exception as e:
        logger.error(f"Domain adaptation training failed: {e}")
        return 1
    
    logger.info(f"Domain adaptation experiment {args.experiment_name} completed")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
