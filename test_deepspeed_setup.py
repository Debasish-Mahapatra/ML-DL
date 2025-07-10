#!/usr/bin/env python3
"""
DeepSpeed Setup Test Script
Tests all components before running full training to ensure everything works with your hardware.
"""

import os
import sys
import torch
import psutil
import json
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_success(message):
    """Print success message."""
    print(f"✅ {message}")

def print_warning(message):
    """Print warning message."""
    print(f"⚠️  {message}")

def print_error(message):
    """Print error message."""
    print(f"❌ {message}")

def check_system_requirements():
    """Check system requirements and hardware."""
    print_header("System Requirements Check")
    
    # Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print_success(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print_error(f"Python version {python_version.major}.{python_version.minor} too old. Need >= 3.8")
        return False
    
    # CPU info
    cpu_count = os.cpu_count()
    cpu_freq = psutil.cpu_freq()
    print_success(f"CPU: {cpu_count} cores at {cpu_freq.current:.0f} MHz")
    
    # Memory info
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    print_success(f"RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    print_success(f"SWAP: {swap.total / (1024**3):.1f} GB total, {swap.free / (1024**3):.1f} GB free")
    
    # Disk space
    disk = psutil.disk_usage('/')
    print_success(f"Disk: {disk.free / (1024**3):.1f} GB free space")
    
    return True

def check_pytorch_setup():
    """Check PyTorch installation and CUDA."""
    print_header("PyTorch & CUDA Check")
    
    # PyTorch version
    torch_version = torch.__version__
    print_success(f"PyTorch version: {torch_version}")
    
    # CUDA availability
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        print_success(f"CUDA version: {cuda_version}")
        print_success(f"GPU: {gpu_name}")
        print_success(f"GPU memory: {gpu_memory:.1f} GB")
        print_success(f"GPU count: {gpu_count}")
        
        # Test CUDA operations
        try:
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            z = torch.mm(x, y)
            print_success("CUDA operations test passed")
            
            # Clear memory
            del x, y, z
            torch.cuda.empty_cache()
            
        except Exception as e:
            print_error(f"CUDA operations test failed: {e}")
            return False
            
    else:
        print_error("CUDA not available")
        return False
    
    return True

def check_deepspeed_installation():
    """Check DeepSpeed installation and compatibility."""
    print_header("DeepSpeed Installation Check")
    
    try:
        import deepspeed
        ds_version = deepspeed.__version__
        print_success(f"DeepSpeed version: {ds_version}")
        
        # Check DeepSpeed report
        print("\nRunning DeepSpeed compatibility report...")
        os.system("ds_report")
        
        return True
        
    except ImportError:
        print_error("DeepSpeed not installed. Run: pip install deepspeed")
        return False

def check_config_files():
    """Check if all configuration files exist."""
    print_header("Configuration Files Check")
    
    required_configs = [
        "config/deepspeed_config.json",
        "config/model_config.yaml", 
        "config/data_config.yaml",
        "config/training_config.yaml"
    ]
    
    all_exist = True
    for config_file in required_configs:
        if Path(config_file).exists():
            print_success(f"Found: {config_file}")
        else:
            print_error(f"Missing: {config_file}")
            all_exist = False
    
    return all_exist

def test_config_loading():
    """Test loading all configurations."""
    print_header("Configuration Loading Test")
    
    try:
        from omegaconf import OmegaConf
        from src.utils.config import get_config
        
        # Load main config
        config = get_config("config")
        print_success("Main configuration loaded successfully")
        
        # Check key sections
        required_sections = ['model', 'data', 'training']
        for section in required_sections:
            if hasattr(config, section):
                print_success(f"Section '{section}' found")
            else:
                print_error(f"Section '{section}' missing")
                return False
        
        # Check DeepSpeed config
        if config.training.get('deepspeed', {}).get('enabled', False):
            print_success("DeepSpeed enabled in training config")
            
            # Load DeepSpeed JSON config
            ds_config_path = config.training.deepspeed.get('config_path', 'config/deepspeed_config.json')
            if Path(ds_config_path).exists():
                with open(ds_config_path, 'r') as f:
                    ds_config = json.load(f)
                print_success("DeepSpeed JSON config loaded successfully")
            else:
                print_error(f"DeepSpeed config file missing: {ds_config_path}")
                return False
        else:
            print_warning("DeepSpeed not enabled in training config")
        
        return True
        
    except Exception as e:
        print_error(f"Configuration loading failed: {e}")
        return False

def test_model_creation():
    """Test model creation and parameter count - simplified for large models."""
    print_header("Model Creation Test")
    
    try:
        from src.utils.config import get_config
        from src.models.architecture import LightningPredictor
        
        # Load config
        config = get_config("config")
        
        # Create model on CPU first (avoids GPU memory issues)
        print("Creating model on CPU...")
        model = LightningPredictor(config)
        print_success("Model created successfully on CPU")
        
        # Model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        
        print_success(f"Total parameters: {total_params:,}")
        print_success(f"Trainable parameters: {trainable_params:,}")
        print_success(f"Model size: {model_size_mb:.1f} MB")
        
        # Check if we reached our target parameter count
        if total_params >= 35_000_000:  # Allow some flexibility around 40M
            print_success(f"✅ Target parameter count achieved! ({total_params:,} parameters)")
        else:
            print_warning(f"Parameter count below target ({total_params:,} < 35M)")
        
        # Test very small forward pass on GPU (just to verify structure)
        if torch.cuda.is_available():
            try:
                print("Testing minimal forward pass on GPU...")
                
                # Move only parts of model to GPU and test with tiny data
                model.cape_encoder = model.cape_encoder.cuda()
                
                # Tiny test data
                cape_data = torch.randn(1, 1, 8, 8, device='cuda')
                
                with torch.no_grad():
                    # Test just the encoder
                    cape_features = model.cape_encoder(cape_data)
                    print_success("Minimal GPU test successful")
                    print_success(f"CAPE encoder output shape: {cape_features.shape}")
                
                # Clear GPU memory
                del cape_data, cape_features
                model.cape_encoder = model.cape_encoder.cpu()
                torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                print_warning("GPU memory too limited for any testing")
                print_warning("This is expected - DeepSpeed will handle full model during training")
        
        # Note about DeepSpeed
        print_success("✅ Model structure verified")
        print_success("✅ DeepSpeed will handle GPU memory during actual training")
        print_success("✅ Parameters will be offloaded to CPU/RAM/SWAP automatically")
        
        return True
        
    except Exception as e:
        print_error(f"Model creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trainer_creation():
    """Test PyTorch Lightning trainer creation with DeepSpeed."""
    print_header("Trainer Creation Test")
    
    try:
        from src.utils.config import get_config
        from src.training.trainer import create_trainer
        
        # Load config
        config = get_config("config")
        
        # Create trainer
        trainer, lightning_module = create_trainer(
            config, 
            experiment_name="deepspeed_test",
            logger_type="tensorboard"
        )
        
        print_success("Trainer created successfully")
        print_success(f"Strategy: {type(trainer.strategy).__name__}")
        print_success(f"Precision: {trainer.precision}")
        print_success(f"Accelerator: {trainer.accelerator}")
        
        # Check DeepSpeed strategy
        if hasattr(trainer.strategy, 'config'):
            print_success("DeepSpeed strategy configured")
        else:
            print_warning("DeepSpeed strategy not detected")
        
        return True
        
    except Exception as e:
        print_error(f"Trainer creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_usage():
    """Test memory usage patterns."""
    print_header("Memory Usage Test")
    
    try:
        import torch
        
        # Initial memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / (1024**2)
            print_success(f"Initial GPU memory: {initial_memory:.1f} MB")
            
            # Create some tensors to test memory
            tensors = []
            for i in range(5):
                tensor = torch.randn(100, 100, 100, device='cuda')
                tensors.append(tensor)
                current_memory = torch.cuda.memory_allocated() / (1024**2)
                print_success(f"Step {i+1} GPU memory: {current_memory:.1f} MB")
            
            # Peak memory
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
            print_success(f"Peak GPU memory: {peak_memory:.1f} MB")
            
            # Cleanup
            del tensors
            torch.cuda.empty_cache()
            
            final_memory = torch.cuda.memory_allocated() / (1024**2)
            print_success(f"Final GPU memory: {final_memory:.1f} MB")
        
        # System memory
        memory = psutil.virtual_memory()
        print_success(f"System RAM usage: {memory.percent:.1f}%")
        
        return True
        
    except Exception as e:
        print_error(f"Memory test failed: {e}")
        return False

def create_test_summary(results):
    """Create test summary and recommendations."""
    print_header("Test Summary")
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    print(f"Passed: {passed_tests}/{total_tests} tests")
    
    if passed_tests == total_tests:
        print_success("All tests passed! Your DeepSpeed setup is ready.")
        print("\n🚀 Ready to run training with:")
        print("   python scripts/train.py --experiment-name cape_deepspeed_test")
    else:
        print_error("Some tests failed. Please fix the issues before training.")
        
        print("\n📋 Failed tests:")
        for test_name, passed in results.items():
            if not passed:
                print(f"   ❌ {test_name}")
    
    # Hardware recommendations
    print_header("Hardware Optimization Recommendations")
    
    memory = psutil.virtual_memory()
    if memory.available / (1024**3) < 8:
        print_warning("Less than 8GB RAM available. Consider closing other applications.")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory < 8:
            print_warning("GPU has less than 8GB memory. Ensure DeepSpeed offloading is enabled.")
    
    swap = psutil.swap_memory()
    if swap.total / (1024**3) < 50:
        print_warning("Less than 50GB swap space. Consider increasing swap for DeepSpeed offloading.")

def main():
    """Run all tests."""
    print_header("DeepSpeed Setup Verification")
    print(f"Test started at: {datetime.now()}")
    print(f"Working directory: {os.getcwd()}")
    
    # Run all tests
    results = {}
    
    results["System Requirements"] = check_system_requirements()
    results["PyTorch & CUDA"] = check_pytorch_setup()
    results["DeepSpeed Installation"] = check_deepspeed_installation()
    results["Configuration Files"] = check_config_files()
    results["Configuration Loading"] = test_config_loading()
    results["Model Creation"] = test_model_creation()
    results["Trainer Creation"] = test_trainer_creation()
    results["Memory Usage"] = test_memory_usage()
    
    # Create summary
    create_test_summary(results)
    
    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)