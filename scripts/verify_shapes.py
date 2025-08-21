#!/usr/bin/env python3
"""
Standalone Dataset Shape Verification Script

This script verifies that your dataset configurations and actual data shapes
are correct for the Odisha lightning prediction domain.

Usage:
    python scripts/verify_shapes.py

No init file changes needed - this is a standalone verification tool.
"""

import sys
import os
from pathlib import Path
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configure simple logging for this script
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def print_header(title: str):
    """Print a nice header for sections."""
    print("\n" + "="*70)
    print(f"🔍 {title}")
    print("="*70)

def print_result(success: bool, message: str):
    """Print a result with appropriate emoji."""
    emoji = "✅" if success else "❌"
    print(f"{emoji} {message}")

def verify_dataset_shapes(datamodule) -> bool:
    """
    Verify that all dataset shapes match expected Odisha dimensions.
    
    Args:
        datamodule: Lightning DataModule instance
        
    Returns:
        True if all shapes are correct
    """
    # Expected shapes for Odisha domain based on your data analysis
    expected_shapes = {
        'cape': (19, 26),      # 25km resolution: (lat, lon)
        'lightning': (181, 221), # 3km resolution: (lat, lon)
        'terrain': (553, 660)   # 1km resolution: (lat, lon)
    }
    
    print_header("DATASET SHAPE VERIFICATION")
    
    # Get actual shapes from datasets
    datasets_to_check = []
    if hasattr(datamodule, 'train_dataset') and datamodule.train_dataset is not None:
        datasets_to_check.append(('Training', datamodule.train_dataset))
    if hasattr(datamodule, 'val_dataset') and datamodule.val_dataset is not None:
        datasets_to_check.append(('Validation', datamodule.val_dataset))
    if hasattr(datamodule, 'test_dataset') and datamodule.test_dataset is not None:
        datasets_to_check.append(('Test', datamodule.test_dataset))
    
    if not datasets_to_check:
        print_result(False, "No datasets found! Call datamodule.setup('fit') first.")
        return False
    
    all_correct = True
    
    for dataset_name, dataset in datasets_to_check:
        print(f"\n📊 {dataset_name} Dataset:")
        
        actual_shapes = {
            'cape': dataset.target_cape_shape,
            'lightning': dataset.target_lightning_shape,
            'terrain': dataset.target_terrain_shape
        }
        
        for data_type, expected in expected_shapes.items():
            actual = actual_shapes[data_type]
            is_correct = actual == expected
            
            if is_correct:
                print_result(True, f"{data_type.upper()}: {actual} (correct)")
            else:
                print_result(False, f"{data_type.upper()}: {actual} (expected {expected})")
                all_correct = False
    
    print("\n" + "="*70)
    if all_correct:
        print_result(True, "ALL DATASET SHAPES ARE CORRECT!")
        print("🚀 Ready to proceed with training...")
    else:
        print_result(False, "DATASET SHAPE ERRORS DETECTED!")
        print("🛑 Fix config/simplified/data_config.yaml before training!")
        print("\n📝 Correct config should be:")
        print("  domain:")
        print("    grid_size_25km: [19, 26]    # CAPE")
        print("    grid_size_3km: [181, 221]   # Lightning") 
        print("    grid_size_1km: [553, 660]   # Terrain")
    
    return all_correct

def verify_single_batch_shapes(datamodule) -> bool:
    """
    Load and verify actual tensor shapes from a single batch.
    
    Args:
        datamodule: Lightning DataModule instance
        
    Returns:
        True if batch shapes are correct
    """
    try:
        print_header("SINGLE BATCH SHAPE VERIFICATION")
        
        print("📦 Loading a sample batch...")
        
        # Get a single batch
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        
        # Expected batch shapes (with batch dimension)
        batch_size = batch['cape'].shape[0]
        expected_batch_shapes = {
            'cape': (batch_size, 1, 19, 26),        # (batch, channels, lat, lon)
            'lightning': (batch_size, 181, 221),    # (batch, lat, lon)
            'terrain': (batch_size, 1, 553, 660)    # (batch, channels, lat, lon)
        }
        
        print(f"📊 Batch size: {batch_size}")
        print(f"📊 Actual tensor shapes:")
        
        all_correct = True
        
        for data_type, expected_shape in expected_batch_shapes.items():
            if data_type in batch:
                actual_shape = tuple(batch[data_type].shape)
                is_correct = actual_shape == expected_shape
                
                if is_correct:
                    print_result(True, f"{data_type.upper()}: {actual_shape} (correct)")
                else:
                    print_result(False, f"{data_type.upper()}: {actual_shape} (expected {expected_shape})")
                    all_correct = False
            else:
                print_result(False, f"{data_type.upper()}: Missing from batch")
                all_correct = False
        
        print("\n" + "="*70)
        if all_correct:
            print_result(True, "ALL BATCH SHAPES ARE CORRECT!")
            print("🎯 Data pipeline is working properly!")
        else:
            print_result(False, "BATCH SHAPE ERRORS DETECTED!")
            print("🛑 Check preprocessing.py and config!")
        
        return all_correct
        
    except Exception as e:
        print_result(False, f"Error during batch verification: {e}")
        print(f"📋 Full error: {str(e)}")
        return False

def verify_config_loading() -> bool:
    """Verify that config can be loaded properly."""
    try:
        print_header("CONFIG LOADING VERIFICATION")
        
        from src.utils.config import get_config
        config = get_config()
        
        # Check that domain config exists
        if not hasattr(config, 'data'):
            print_result(False, "Config missing 'data' section")
            return False
            
        if not hasattr(config.data, 'domain'):
            print_result(False, "Config missing 'data.domain' section")
            return False
        
        # Check required domain keys
        required_keys = ['grid_size_25km', 'grid_size_3km', 'grid_size_1km']
        for key in required_keys:
            if not hasattr(config.data.domain, key):
                print_result(False, f"Config missing 'data.domain.{key}'")
                return False
        
        # Show current config values
        print("📝 Current config values:")
        print(f"  grid_size_25km: {config.data.domain.grid_size_25km}")
        print(f"  grid_size_3km: {config.data.domain.grid_size_3km}")
        print(f"  grid_size_1km: {config.data.domain.grid_size_1km}")
        
        print_result(True, "Config loading successful")
        return True
        
    except Exception as e:
        print_result(False, f"Config loading failed: {e}")
        return False

def main():
    """Main verification function."""
    print("🔍 Lightning Prediction Dataset Shape Verification")
    print("📍 Verifying Odisha domain configuration and data shapes...")
    
    # Step 1: Verify config loading
    if not verify_config_loading():
        print("\n❌ VERIFICATION FAILED: Config issues detected!")
        return 1
    
    try:
        # Import after config verification
        from src.utils.config import get_config
        from src.data.data_loader import LightningDataModule
        
        # Load config
        config = get_config()
        
        # Create and setup datamodule
        print("\n📦 Setting up data module...")
        datamodule = LightningDataModule(config)
        datamodule.setup('fit')
        
        # Step 2: Verify dataset shapes
        shapes_ok = verify_dataset_shapes(datamodule)
        
        # Step 3: Verify actual batch shapes  
        batch_ok = verify_single_batch_shapes(datamodule)
        
        # Final result
        print_header("FINAL VERIFICATION RESULT")
        
        if shapes_ok and batch_ok:
            print_result(True, "ALL VERIFICATIONS PASSED!")
            print("🎉 Your dataset configuration is correct!")
            print("🚀 Ready to start training with confidence!")
            return 0
        else:
            print_result(False, "VERIFICATION FAILED!")
            print("🛑 Fix the issues above before training!")
            return 1
            
    except ImportError as e:
        print_result(False, f"Import error: {e}")
        print("📋 Make sure you're running from the project root directory")
        return 1
    except Exception as e:
        print_result(False, f"Unexpected error: {e}")
        logger.exception("Full error details:")
        return 1

if __name__ == "__main__":
    """Run the verification script."""
    exit_code = main()
    sys.exit(exit_code)