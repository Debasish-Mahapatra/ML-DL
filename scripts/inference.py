"""
Enhanced inference script for lightning prediction with physics-informed analysis.
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
import numpy as np
import xarray as xr
from omegaconf import OmegaConf

from src.utils.config import get_config
from src.training.trainer import LightningTrainer
from src.data.preprocessing import DataPreprocessor
from src.utils.io_utils import NetCDFHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_inference_data(cape_file: str, terrain_file: str, preprocessor: DataPreprocessor) -> dict:
    """Load and preprocess data for inference."""
    
    # Load CAPE data
    cape_ds = NetCDFHandler.load_netcdf(cape_file, variables=['cape'])
    if cape_ds is None:
        raise ValueError(f"Failed to load CAPE data from {cape_file}")
    
    cape_tensor = preprocessor.preprocess_cape(cape_ds['cape'])
    
    # Load terrain data
    terrain_ds = NetCDFHandler.load_netcdf(terrain_file, variables=['elevation'])
    if terrain_ds is None:
        raise ValueError(f"Failed to load terrain data from {terrain_file}")
    
    # Target size for terrain (3km resolution)
    target_size = (cape_tensor.shape[-2] * 8, cape_tensor.shape[-1] * 8)
    terrain_tensor = preprocessor.preprocess_terrain(terrain_ds['elevation'], target_size)
    
    # Store raw CAPE values for physics analysis
    raw_cape = cape_ds['cape'].values if hasattr(cape_ds['cape'], 'values') else cape_ds['cape']
    
    # Close datasets
    cape_ds.close()
    terrain_ds.close()
    
    return {
        'cape': cape_tensor.unsqueeze(0),  # Add batch dimension
        'terrain': terrain_tensor.unsqueeze(0),  # Add batch dimension
        'raw_cape': raw_cape  # Keep original CAPE values for analysis
    }

def analyze_cape_physics(cape_data: np.ndarray, config: dict = None) -> dict:
    """
    Analyze CAPE data for physics regime distribution and expected lightning patterns.
    """
    
    # Default CAPE thresholds (enhanced physics)
    if config and hasattr(config, 'training') and hasattr(config.training, 'cape_physics'):
        thresholds = config.training.cape_physics.thresholds
    else:
        thresholds = {
            'no_lightning': 1000.0,
            'moderate': 2500.0,
            'high': 4000.0,
            'saturation': 5000.0
        }
    
    # Flatten CAPE data for analysis
    cape_flat = cape_data.flatten()
    cape_flat = cape_flat[~np.isnan(cape_flat)]  # Remove NaN values
    
    # Calculate regime statistics
    no_lightning_mask = cape_flat < thresholds['no_lightning']
    moderate_mask = (cape_flat >= thresholds['no_lightning']) & (cape_flat < thresholds['moderate'])
    high_mask = (cape_flat >= thresholds['moderate']) & (cape_flat < thresholds['high'])
    very_high_mask = cape_flat >= thresholds['high']
    
    regime_stats = {
        'overall': {
            'min': float(cape_flat.min()),
            'max': float(cape_flat.max()),
            'mean': float(cape_flat.mean()),
            'std': float(cape_flat.std()),
            'median': float(np.median(cape_flat)),
            'total_pixels': len(cape_flat)
        },
        'regimes': {
            'no_lightning': {
                'count': int(no_lightning_mask.sum()),
                'percentage': float(no_lightning_mask.mean() * 100),
                'expected_lightning_rate': 0.05
            },
            'moderate': {
                'count': int(moderate_mask.sum()),
                'percentage': float(moderate_mask.mean() * 100),
                'expected_lightning_rate': 0.20
            },
            'high': {
                'count': int(high_mask.sum()),
                'percentage': float(high_mask.mean() * 100),
                'expected_lightning_rate': 0.50
            },
            'very_high': {
                'count': int(very_high_mask.sum()),
                'percentage': float(very_high_mask.mean() * 100),
                'expected_lightning_rate': 0.70
            }
        },
        'thresholds': thresholds
    }
    
    return regime_stats

def run_inference_with_physics(model: LightningTrainer, 
                               input_data: dict,
                               device: torch.device) -> dict:
    """Run inference with enhanced physics analysis."""
    
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        # Move data to device
        cape_data = input_data['cape'].to(device)
        terrain_data = input_data['terrain'].to(device)
        
        # Run inference
        outputs = model(cape_data, terrain_data)
        
        # Convert logits to probabilities if needed
        if 'lightning_prediction' in outputs:
            lightning_logits = outputs['lightning_prediction']
            lightning_probs = torch.sigmoid(lightning_logits)
            outputs['lightning_probability'] = lightning_probs
        
        # Move results back to CPU
        results = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                results[key] = value.cpu()
            else:
                results[key] = value
    
    return results

def analyze_prediction_physics(predictions: torch.Tensor, 
                              cape_data: np.ndarray,
                              regime_stats: dict) -> dict:
    """
    Analyze how predictions align with physics expectations across CAPE regimes.
    """
    
    # Flatten data for analysis
    pred_flat = predictions.flatten().numpy()
    cape_flat = cape_data.flatten()
    
    # Remove NaN/invalid values
    valid_mask = ~(np.isnan(cape_flat) | np.isnan(pred_flat))
    pred_flat = pred_flat[valid_mask]
    cape_flat = cape_flat[valid_mask]
    
    thresholds = regime_stats['thresholds']
    
    # Analyze each regime
    regime_analysis = {}
    
    regimes = [
        ('no_lightning', cape_flat < thresholds['no_lightning']),
        ('moderate', (cape_flat >= thresholds['no_lightning']) & (cape_flat < thresholds['moderate'])),
        ('high', (cape_flat >= thresholds['moderate']) & (cape_flat < thresholds['high'])),
        ('very_high', cape_flat >= thresholds['high'])
    ]
    
    for regime_name, mask in regimes:
        if mask.sum() == 0:
            continue
        
        regime_preds = pred_flat[mask]
        regime_cape = cape_flat[mask]
        expected_rate = regime_stats['regimes'][regime_name]['expected_lightning_rate']
        
        # Calculate actual prediction statistics
        actual_rate = (regime_preds > 0.5).mean()
        mean_prob = regime_preds.mean()
        
        regime_analysis[regime_name] = {
            'pixel_count': int(mask.sum()),
            'cape_range': {
                'min': float(regime_cape.min()),
                'max': float(regime_cape.max()),
                'mean': float(regime_cape.mean())
            },
            'predictions': {
                'mean_probability': float(mean_prob),
                'lightning_rate_predicted': float(actual_rate),
                'lightning_rate_expected': float(expected_rate),
                'physics_alignment': float(1.0 - abs(actual_rate - expected_rate))
            }
        }
    
    # Overall physics metrics
    cape_pred_correlation = np.corrcoef(cape_flat, pred_flat)[0, 1]
    
    physics_analysis = {
        'regime_analysis': regime_analysis,
        'overall_metrics': {
            'cape_prediction_correlation': float(cape_pred_correlation) if not np.isnan(cape_pred_correlation) else 0.0,
            'mean_prediction': float(pred_flat.mean()),
            'prediction_std': float(pred_flat.std()),
            'high_confidence_pixels': int((pred_flat > 0.7).sum()),
            'low_confidence_pixels': int((pred_flat < 0.3).sum())
        }
    }
    
    return physics_analysis

def save_enhanced_predictions(predictions: torch.Tensor,
                            cape_data: np.ndarray,
                            regime_stats: dict,
                            physics_analysis: dict,
                            output_file: str,
                            reference_file: str = None):
    """Save predictions with enhanced metadata and physics analysis."""
    
    # Remove batch dimension
    pred_np = predictions.squeeze(0).squeeze(0).numpy()
    
    # Create coordinates
    height, width = pred_np.shape
    lat = np.linspace(17.0, 22.0, height)  # Approximate Odisha bounds
    lon = np.linspace(81.0, 87.0, width)
    
    # If reference file provided, use its coordinates
    if reference_file:
        try:
            ref_ds = xr.open_dataset(reference_file)
            if 'lat' in ref_ds.coords and 'lon' in ref_ds.coords:
                # Interpolate to prediction grid size
                ref_lat = ref_ds.lat.values
                ref_lon = ref_ds.lon.values
                
                lat = np.linspace(ref_lat.min(), ref_lat.max(), height)
                lon = np.linspace(ref_lon.min(), ref_lon.max(), width)
            
            ref_ds.close()
        except Exception as e:
            logger.warning(f"Could not use reference coordinates: {e}")
    
    # Resize CAPE data to match predictions for saving
    if cape_data.shape != pred_np.shape:
        from scipy.ndimage import zoom
        if len(cape_data.shape) == 3:  # (time, lat, lon)
            cape_data = cape_data[0]  # Take first time step
        
        # Calculate zoom factors
        zoom_factors = (pred_np.shape[0] / cape_data.shape[0], 
                       pred_np.shape[1] / cape_data.shape[1])
        cape_resized = zoom(cape_data, zoom_factors, order=1)
    else:
        cape_resized = cape_data
    
    # Create confidence masks based on physics
    confidence_mask = np.ones_like(pred_np)
    
    # Higher confidence in areas with appropriate CAPE values
    for regime, analysis in physics_analysis['regime_analysis'].items():
        if regime == 'no_lightning':
            # Lower confidence where we predict lightning in low CAPE areas
            low_cape_mask = cape_resized < regime_stats['thresholds']['no_lightning']
            confidence_mask[low_cape_mask & (pred_np > 0.5)] *= 0.5
        elif regime == 'very_high':
            # Higher confidence in very high CAPE areas
            high_cape_mask = cape_resized >= regime_stats['thresholds']['high']
            confidence_mask[high_cape_mask] *= 1.2
    
    confidence_mask = np.clip(confidence_mask, 0.0, 1.0)
    
    # Create dataset with enhanced variables
    ds = xr.Dataset({
        'lightning_probability': (['lat', 'lon'], pred_np),
        'cape_input': (['lat', 'lon'], cape_resized),
        'physics_confidence': (['lat', 'lon'], confidence_mask),
        'lightning_binary': (['lat', 'lon'], (pred_np > 0.5).astype(int))
    }, coords={
        'lat': lat,
        'lon': lon
    })
    
    # Add comprehensive metadata
    ds.lightning_probability.attrs = {
        'long_name': 'Lightning occurrence probability',
        'units': 'probability',
        'description': 'Predicted probability of lightning occurrence from enhanced physics model',
        'valid_range': [0.0, 1.0]
    }
    
    ds.cape_input.attrs = {
        'long_name': 'Convective Available Potential Energy',
        'units': 'J/kg',
        'description': 'CAPE values used as model input'
    }
    
    ds.physics_confidence.attrs = {
        'long_name': 'Physics-based confidence score',
        'units': 'dimensionless',
        'description': 'Confidence in prediction based on CAPE physics regime'
    }
    
    ds.lightning_binary.attrs = {
        'long_name': 'Binary lightning prediction',
        'units': 'dimensionless',
        'description': 'Binary lightning occurrence (threshold=0.5)'
    }
    
    # Global attributes with physics analysis
    ds.attrs = {
        'title': 'Enhanced Lightning Prediction Results with Physics Analysis',
        'creation_time': datetime.now().isoformat(),
        'model': 'LightningPredictor_Enhanced_Physics',
        'resolution': '3km',
        'cape_prediction_correlation': physics_analysis['overall_metrics']['cape_prediction_correlation'],
        'mean_prediction_probability': physics_analysis['overall_metrics']['mean_prediction'],
        'high_confidence_pixels': physics_analysis['overall_metrics']['high_confidence_pixels'],
        'physics_regimes_analyzed': list(physics_analysis['regime_analysis'].keys()),
        'cape_thresholds': str(regime_stats['thresholds']),
        'cape_statistics': str(regime_stats['overall'])
    }
    
    # Save to file
    ds.to_netcdf(output_file)
    ds.close()
    
    logger.info(f"Saved enhanced predictions with physics analysis to {output_file}")

def save_physics_analysis_report(regime_stats: dict, 
                                physics_analysis: dict,
                                output_dir: str):
    """Save detailed physics analysis report."""
    
    # Create JSON report
    report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'cape_regime_statistics': regime_stats,
        'physics_alignment_analysis': physics_analysis
    }
    
    json_file = f"{output_dir}/physics_analysis_report.json"
    with open(json_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Create markdown summary
    markdown_content = f"""# Lightning Inference - Physics Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## CAPE Data Statistics
- **Total Pixels**: {regime_stats['overall']['total_pixels']:,}
- **CAPE Range**: {regime_stats['overall']['min']:.1f} - {regime_stats['overall']['max']:.1f} J/kg
- **Mean CAPE**: {regime_stats['overall']['mean']:.1f} J/kg
- **Standard Deviation**: {regime_stats['overall']['std']:.1f} J/kg

## CAPE Regime Distribution
"""
    
    for regime, stats in regime_stats['regimes'].items():
        regime_name = regime.replace('_', ' ').title()
        markdown_content += f"""
### {regime_name} Regime
- **Pixel Count**: {stats['count']:,} ({stats['percentage']:.2f}%)
- **Expected Lightning Rate**: {stats['expected_lightning_rate']:.2f}
"""
        
        if regime in physics_analysis['regime_analysis']:
            analysis = physics_analysis['regime_analysis'][regime]
            markdown_content += f"""- **Predicted Lightning Rate**: {analysis['predictions']['lightning_rate_predicted']:.4f}
- **Mean Prediction Probability**: {analysis['predictions']['mean_probability']:.4f}
- **Physics Alignment Score**: {analysis['predictions']['physics_alignment']:.4f}
"""
    
    markdown_content += f"""
## Overall Physics Performance
- **CAPE-Prediction Correlation**: {physics_analysis['overall_metrics']['cape_prediction_correlation']:.4f}
- **Mean Prediction Probability**: {physics_analysis['overall_metrics']['mean_prediction']:.4f}
- **High Confidence Pixels** (>0.7): {physics_analysis['overall_metrics']['high_confidence_pixels']:,}
- **Low Confidence Pixels** (<0.3): {physics_analysis['overall_metrics']['low_confidence_pixels']:,}

## Physics Assessment
{'✅ **EXCELLENT** - Strong physics alignment' if physics_analysis['overall_metrics']['cape_prediction_correlation'] > 0.6 else '✓ **GOOD** - Moderate physics alignment' if physics_analysis['overall_metrics']['cape_prediction_correlation'] > 0.3 else '⚠️ **NEEDS IMPROVEMENT** - Weak physics alignment'}

**Correlation Score**: {physics_analysis['overall_metrics']['cape_prediction_correlation']:.4f}
- Above 0.6: Excellent physics learning
- 0.3-0.6: Good physics learning  
- Below 0.3: Physics constraints need strengthening

Generated with enhanced physics-aware inference framework.
"""
    
    markdown_file = f"{output_dir}/physics_analysis_summary.md"
    with open(markdown_file, 'w') as f:
        f.write(markdown_content)
    
    logger.info(f"Saved physics analysis reports to {output_dir}")

def main():
    """Main inference function with enhanced physics analysis."""
    
    parser = argparse.ArgumentParser(description="Enhanced Lightning Prediction Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--cape-file", type=str, required=True,
                       help="Path to CAPE NetCDF file")
    parser.add_argument("--terrain-file", type=str, required=True,
                       help="Path to terrain NetCDF file")
    parser.add_argument("--output-file", type=str, required=True,
                       help="Output NetCDF file for predictions")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file")
    parser.add_argument("--reference-file", type=str, default=None,
                       help="Reference file for coordinates")
    parser.add_argument("--save-features", action="store_true",
                       help="Save intermediate features")
    parser.add_argument("--physics-analysis", action="store_true", default=True,
                       help="Perform physics analysis (enabled by default)")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to run inference on")
    
    args = parser.parse_args()
    
    # Validate input files
    for file_path in [args.checkpoint, args.cape_file, args.terrain_file]:
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            return 1
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load model
    try:
        logger.info("Loading enhanced physics model...")
        model = LightningTrainer.load_from_checkpoint(args.checkpoint, map_location=device)
        
        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded: {total_params:,} parameters")
        
        # Check if model has enhanced physics
        if hasattr(model.config, 'training') and hasattr(model.config.training, 'cape_physics'):
            logger.info("✅ Enhanced physics model detected")
        else:
            logger.warning("⚠️ Model may not have enhanced physics constraints")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Initialize preprocessor
    try:
        # Use model's preprocessor stats if available
        if hasattr(model, 'preprocessor'):
            preprocessor = model.preprocessor
        else:
            # Create default preprocessor
            preprocessor = DataPreprocessor()
        
        logger.info("Preprocessor initialized")
    except Exception as e:
        logger.error(f"Failed to initialize preprocessor: {e}")
        return 1
    
    # Load and preprocess input data
    try:
        logger.info("Loading input data...")
        input_data = load_inference_data(args.cape_file, args.terrain_file, preprocessor)
        
        logger.info(f"Input data loaded:")
        logger.info(f"  CAPE shape: {input_data['cape'].shape}")
        logger.info(f"  Terrain shape: {input_data['terrain'].shape}")
        
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        return 1
    
    # Analyze CAPE physics
    if args.physics_analysis:
        try:
            logger.info("Analyzing CAPE physics regimes...")
            regime_stats = analyze_cape_physics(input_data['raw_cape'], model.config)
            
            logger.info("CAPE Regime Distribution:")
            for regime, stats in regime_stats['regimes'].items():
                logger.info(f"  {regime.replace('_', ' ').title()}: {stats['count']:,} pixels ({stats['percentage']:.1f}%)")
            
        except Exception as e:
            logger.error(f"CAPE physics analysis failed: {e}")
            regime_stats = None
    
    # Run inference
    try:
        logger.info("Running enhanced inference...")
        results = run_inference_with_physics(model, input_data, device)
        
        predictions = results['lightning_probability']
        logger.info(f"Inference completed. Prediction shape: {predictions.shape}")
        
        # Log prediction statistics
        pred_np = predictions.numpy()
        logger.info(f"Prediction statistics:")
        logger.info(f"  Min: {pred_np.min():.6f}")
        logger.info(f"  Max: {pred_np.max():.6f}")
        logger.info(f"  Mean: {pred_np.mean():.6f}")
        logger.info(f"  Lightning probability > 0.5: {(pred_np > 0.5).sum():,} pixels ({(pred_np > 0.5).mean()*100:.2f}%)")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return 1
    
    # Physics analysis of predictions
    physics_analysis = None
    if args.physics_analysis and regime_stats:
        try:
            logger.info("Analyzing prediction physics alignment...")
            physics_analysis = analyze_prediction_physics(predictions, input_data['raw_cape'], regime_stats)
            
            # Log key physics metrics
            overall = physics_analysis['overall_metrics']
            logger.info(f"Physics Analysis Results:")
            logger.info(f"  CAPE-Prediction Correlation: {overall['cape_prediction_correlation']:.4f}")
            logger.info(f"  High Confidence Pixels: {overall['high_confidence_pixels']:,}")
            
            for regime, analysis in physics_analysis['regime_analysis'].items():
                alignment = analysis['predictions']['physics_alignment']
                logger.info(f"  {regime.replace('_', ' ').title()} Physics Alignment: {alignment:.4f}")
            
        except Exception as e:
            logger.error(f"Physics analysis of predictions failed: {e}")
            physics_analysis = None
    
    # Save predictions
    try:
        logger.info("Saving enhanced predictions...")
        
        if physics_analysis and regime_stats:
            save_enhanced_predictions(
                predictions, input_data['raw_cape'], regime_stats, 
                physics_analysis, args.output_file, args.reference_file
            )
        else:
            # Fallback to basic saving
            save_basic_predictions(predictions, args.output_file, args.reference_file)
        
    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")
        return 1
    
    # Save physics analysis reports
    if args.physics_analysis and regime_stats and physics_analysis:
        try:
            output_dir = Path(args.output_file).parent
            save_physics_analysis_report(regime_stats, physics_analysis, output_dir)
        except Exception as e:
            logger.warning(f"Failed to save physics analysis report: {e}")
    
    # Save intermediate features if requested
    if args.save_features:
        try:
            features_file = args.output_file.replace('.nc', '_features.pt')
            torch.save({
                'cape_features': results.get('cape_features'),
                'terrain_features': results.get('terrain_features'),
                'fused_features': results.get('fused_features'),
                'model_outputs': results
            }, features_file)
            
            logger.info(f"Saved intermediate features to {features_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save features: {e}")
    
    logger.info("=== INFERENCE SUMMARY ===")
    logger.info(f"✅ Enhanced inference completed successfully")
    logger.info(f"📁 Results saved to: {args.output_file}")
    
    if physics_analysis:
        correlation = physics_analysis['overall_metrics']['cape_prediction_correlation']
        logger.info(f"🔬 Physics Performance: {correlation:.4f} correlation")
        if correlation > 0.6:
            logger.info("🎯 EXCELLENT physics alignment!")
        elif correlation > 0.3:
            logger.info("✅ GOOD physics alignment")
        else:
            logger.info("⚠️ Physics alignment could be improved")
    
    return 0

def save_basic_predictions(predictions: torch.Tensor, output_file: str, reference_file: str = None):
    """Fallback function for basic prediction saving."""
    
    # Remove batch dimension
    pred_np = predictions.squeeze(0).squeeze(0).numpy()
    
    # Create coordinates
    height, width = pred_np.shape
    lat = np.linspace(17.0, 22.0, height)
    lon = np.linspace(81.0, 87.0, width)
    
    # Create basic dataset
    ds = xr.Dataset({
        'lightning_probability': (['lat', 'lon'], pred_np)
    }, coords={
        'lat': lat,
        'lon': lon
    })
    
    ds.lightning_probability.attrs = {
        'long_name': 'Lightning occurrence probability',
        'units': 'probability',
        'description': 'Predicted probability of lightning occurrence'
    }
    
    ds.attrs = {
        'title': 'Lightning Prediction Results',
        'creation_time': datetime.now().isoformat(),
        'model': 'LightningPredictor'
    }
    
    ds.to_netcdf(output_file)
    ds.close()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)