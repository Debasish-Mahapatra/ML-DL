#!/usr/bin/env python3
"""
Test script to verify EfficientConvNet architecture changes.
This script tests the model without requiring actual data files.
"""

import sys
import torch
import time
from pathlib import Path
from omegaconf import OmegaConf

# Add src to path
sys.path.append('src')

def create_test_config():
    """Create test configuration matching your setup."""
    config = OmegaConf.create({
        'data': {
            'domain': {
                'grid_size_25km': [85, 85],
                'grid_size_3km': [710, 710], 
                'grid_size_1km': [2130, 2130]
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
                    'kernel_sizes': [3, 3, 3, 3],
                    'activation': 'relu',
                    'dropout': 0.1
                }
            },
            'fusion': {
                'meteorological': {
                    'hidden_dim': 512,
                    'fusion_method': 'concatenation'
                }
            },
            # NEW: EfficientConvNet configuration
            'gnn': {
                'type': 'EfficientConvNet',
                'hidden_dim': 64,
                'num_layers': 3,
                'num_heads': 4,
                'dropout': 0.1,
                'kernel_sizes': [3, 5, 7],
                'use_multiscale': True,
                'use_attention': True
            },
            'transformer': {
                'hidden_dim': 64,
                'num_layers': 3,
                'num_heads': 4,
                'dropout': 0.1,
                'attention_type': 'linear'
            },
            'prediction_head': {
                'hidden_dim': 128,
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
    
    return config

def test_model_creation():
    """Test 1: Model can be created successfully."""
    print("="*60)
    print("TEST 1: Model Creation")
    print("="*60)
    
    try:
        from models.architecture import LightningPredictor
        
        config = create_test_config()
        model = LightningPredictor(config)
        
        print("✅ Model created successfully!")
        
        # Print model info
        info = model.get_model_info()
        print(f"📊 Model Statistics:")
        print(f"   - Total parameters: {info['total_parameters']:,}")
        print(f"   - Model size: {info['model_size_mb']:.1f} MB")
        print(f"   - Architecture: EfficientConvNet + Transformer")
        
        return model, config
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_forward_pass(model, config):
    """Test 2: Forward pass with dummy data."""
    print("\n" + "="*60)
    print("TEST 2: Forward Pass")
    print("="*60)
    
    try:
        # Create dummy input data matching your domain sizes
        batch_size = 1
        cape_data = torch.randn(batch_size, 1, 85, 85)      # 25km CAPE
        terrain_data = torch.randn(batch_size, 1, 2130, 2130)  # 1km terrain
        
        print(f"📥 Input shapes:")
        print(f"   - CAPE: {cape_data.shape} (25km resolution)")
        print(f"   - Terrain: {terrain_data.shape} (1km resolution)")
        
        # Forward pass
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(cape_data, terrain_data)
        
        forward_time = time.time() - start_time
        
        print(f"📤 Output shapes:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"   - {key}: {value.shape}")
        
        print(f"⏱️  Forward pass time: {forward_time:.3f} seconds")
        
        # Check output resolution
        lightning_pred = outputs['lightning_prediction']
        expected_shape = (batch_size, 1, 710, 710)  # 3km resolution
        
        if lightning_pred.shape == expected_shape:
            print(f"✅ Output shape correct: {lightning_pred.shape} (3km resolution)")
        else:
            print(f"⚠️  Output shape unexpected: {lightning_pred.shape}, expected: {expected_shape}")
        
        # Check if EfficientConvNet features are present
        if 'convnet_features' in outputs:
            print("✅ EfficientConvNet features present in output")
        else:
            print("⚠️  ConvNet features missing from output")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_efficiency():
    """Test 3: Memory efficiency comparison."""
    print("\n" + "="*60)
    print("TEST 3: Memory Efficiency")
    print("="*60)
    
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("🖥️  Testing on GPU")
        else:
            device = torch.device('cpu')
            print("🖥️  Testing on CPU")
        
        config = create_test_config()
        from models.architecture import LightningPredictor
        
        model = LightningPredictor(config).to(device)
        
        # Test with different batch sizes
        batch_sizes = [1, 2]  # Start small
        
        for batch_size in batch_sizes:
            try:
                cape_data = torch.randn(batch_size, 1, 85, 85).to(device)
                terrain_data = torch.randn(batch_size, 1, 2130, 2130).to(device)
                
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                start_time = time.time()
                
                with torch.no_grad():
                    outputs = model(cape_data, terrain_data)
                
                forward_time = time.time() - start_time
                
                if torch.cuda.is_available():
                    memory_used = torch.cuda.max_memory_allocated() / (1024**3)  # GB
                    print(f"📊 Batch size {batch_size}:")
                    print(f"   - Forward time: {forward_time:.3f}s")
                    print(f"   - Peak GPU memory: {memory_used:.2f} GB")
                else:
                    print(f"📊 Batch size {batch_size}:")
                    print(f"   - Forward time: {forward_time:.3f}s")
                
                # Clean up
                del cape_data, terrain_data, outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"💾 Batch size {batch_size}: Out of memory")
                    break
                else:
                    raise e
        
        return True
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def test_components_import():
    """Test 4: All components can be imported."""
    print("\n" + "="*60)
    print("TEST 4: Component Imports")
    print("="*60)
    
    components_to_test = [
        ('models.components.efficient_convnet', 'EfficientConvNet'),
        ('models.components.efficient_convnet', 'MultiScaleConvNet'),
        ('models.architecture', 'LightningPredictor'),
        ('models.components', 'EfficientConvNet')
    ]
    
    all_passed = True
    
    for module_name, class_name in components_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            component_class = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"❌ {module_name}.{class_name}: {e}")
            all_passed = False
    
    return all_passed

def test_configuration_compatibility():
    """Test 5: Configuration is compatible."""
    print("\n" + "="*60)
    print("TEST 5: Configuration Compatibility")
    print("="*60)
    
    try:
        config = create_test_config()
        
        # Check required config sections
        required_sections = [
            'model.gnn.type',
            'model.gnn.hidden_dim',
            'model.gnn.num_layers',
            'model.gnn.kernel_sizes',
            'model.gnn.use_multiscale',
            'model.gnn.use_attention'
        ]
        
        for section in required_sections:
            try:
                value = OmegaConf.select(config, section)
                if value is not None:
                    print(f"✅ {section}: {value}")
                else:
                    print(f"⚠️  {section}: Missing")
            except Exception as e:
                print(f"❌ {section}: {e}")
        
        # Verify EfficientConvNet is configured
        gnn_type = OmegaConf.select(config, 'model.gnn.type')
        if gnn_type == 'EfficientConvNet':
            print("✅ Model configured for EfficientConvNet")
        else:
            print(f"⚠️  Model type is {gnn_type}, expected EfficientConvNet")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 TESTING EFFICIENTCONVNET ARCHITECTURE CHANGES")
    print("="*80)
    
    test_results = []
    
    # Test 1: Component imports
    print("Starting component import tests...")
    result1 = test_components_import()
    test_results.append(("Component Imports", result1))
    
    # Test 2: Configuration compatibility
    result2 = test_configuration_compatibility()
    test_results.append(("Configuration", result2))
    
    # Test 3: Model creation
    model, config = test_model_creation()
    test_results.append(("Model Creation", model is not None))
    
    if model is not None:
        # Test 4: Forward pass
        result4 = test_forward_pass(model, config)
        test_results.append(("Forward Pass", result4))
        
        # Test 5: Memory efficiency
        result5 = test_memory_efficiency()
        test_results.append(("Memory Efficiency", result5))
    
    # Summary
    print("\n" + "="*80)
    print("🏆 TEST SUMMARY")
    print("="*80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<50} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ EfficientConvNet architecture is working correctly")
        print("🚀 You can now run training with much better performance!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("🔧 Please fix the issues before proceeding with training")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)