"""
Enhanced inference script for lightning prediction with physics-informed analysis.
VERSION: Updated to be compatible with both single-stage (tensor output) and 
         two-stage (dictionary output) models.
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

def get_lightning_prediction(model_output):
    """
    Compatibility helper to get lightning prediction from both old and new models.
    
    Args:
        model_output: The raw output from the model's forward pass.
    
    Returns:
        The tensor containing lightning prediction logits.
    """
    if isinstance(model_output, dict):
        # New two-stage models return a dictionary
        if 'lightning_prediction' not in model_output:
            raise KeyError("Model output dictionary is missing 'lightning_prediction' key.")
        return model_output['lightning_prediction']
    elif torch.is_tensor(model_output):
        # Old models return a single tensor
        return model_output
    else:
        raise TypeError(f"Unsupported model output type: {type(model_output)}")


def load_inference_data(cape_file: str, terrain_file: str, preprocessor: DataPreprocessor, config: OmegaConf) -> dict:
    """Load and preprocess data for inference."""
    cape_ds = NetCDFHandler.load_netcdf(cape_file, variables=['cape'])
    if cape_ds is None: raise ValueError(f"Failed to load CAPE data from {cape_file}")
    
    cape_tensor = preprocessor.preprocess_cape(cape_ds['cape'])
    
    terrain_ds = NetCDFHandler.load_netcdf(terrain_file, variables=['elevation'])
    if terrain_ds is None: raise ValueError(f"Failed to load terrain data from {terrain_file}")
    
    # Use target shape from config for consistency
    target_size = tuple(config.data.domain.grid_size_3km)
    terrain_tensor = preprocessor.preprocess_terrain(terrain_ds['elevation'], target_size)
    
    raw_cape = cape_ds['cape'].values if hasattr(cape_ds['cape'], 'values') else cape_ds['cape']
    
    cape_ds.close()
    terrain_ds.close()
    
    return {
        'cape': cape_tensor.unsqueeze(0),
        'terrain': terrain_tensor.unsqueeze(0),
        'raw_cape': raw_cape
    }

def analyze_cape_physics(cape_data: np.ndarray, config: OmegaConf = None) -> dict:
    """Analyze CAPE data for physics regime distribution."""
    thresholds = {'no_lightning': 1000.0, 'moderate': 2500.0, 'high': 4000.0}
    if config and 'cape_physics' in config.training:
        thresholds = config.training.cape_physics.thresholds

    cape_flat = cape_data.flatten()
    cape_flat = cape_flat[~np.isnan(cape_flat)]
    
    masks = {
        'no_lightning': cape_flat < thresholds['no_lightning'],
        'moderate': (cape_flat >= thresholds['no_lightning']) & (cape_flat < thresholds['moderate']),
        'high': (cape_flat >= thresholds['moderate']) & (cape_flat < thresholds['high']),
        'very_high': cape_flat >= thresholds['high']
    }
    
    regime_stats = {
        'overall': {'min': float(cape_flat.min()), 'max': float(cape_flat.max()), 'mean': float(cape_flat.mean()), 'std': float(cape_flat.std()), 'median': float(np.median(cape_flat)), 'total_pixels': len(cape_flat)},
        'regimes': {name: {'count': int(mask.sum()), 'percentage': float(mask.mean() * 100)} for name, mask in masks.items()},
        'thresholds': thresholds
    }
    return regime_stats

def run_inference_with_physics(model: LightningTrainer, input_data: dict, device: torch.device) -> dict:
    """Run inference and process outputs."""
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        cape_data = input_data['cape'].to(device)
        terrain_data = input_data['terrain'].to(device)
        
        raw_outputs = model(cape_data, terrain_data)
        
        # --- UPDATED: Use compatibility helper ---
        lightning_logits = get_lightning_prediction(raw_outputs)
        lightning_probs = torch.sigmoid(lightning_logits)
        
        # Return only the necessary final prediction for analysis and saving
        return {'lightning_probability': lightning_probs.cpu()}

def analyze_prediction_physics(predictions: torch.Tensor, cape_data: np.ndarray, regime_stats: dict) -> dict:
    """Analyze how predictions align with physics expectations."""
    pred_flat = predictions.flatten().numpy()
    cape_flat = cape_data.flatten()
    
    valid_mask = ~(np.isnan(cape_flat) | np.isnan(pred_flat))
    pred_flat, cape_flat = pred_flat[valid_mask], cape_flat[valid_mask]
    
    thresholds = regime_stats['thresholds']
    regime_analysis = {}
    
    masks = {
        'no_lightning': cape_flat < thresholds['no_lightning'],
        'moderate': (cape_flat >= thresholds['no_lightning']) & (cape_flat < thresholds['moderate']),
        'high': (cape_flat >= thresholds['moderate']) & (cape_flat < thresholds['high']),
        'very_high': cape_flat >= thresholds['high']
    }

    for name, mask in masks.items():
        if not mask.any(): continue
        regime_preds = pred_flat[mask]
        regime_analysis[name] = {
            'pixel_count': int(mask.sum()),
            'predictions': {
                'mean_probability': float(regime_preds.mean()),
                'lightning_rate_predicted': float((regime_preds > 0.5).mean())
            }
        }
        
    cape_pred_corr = np.corrcoef(cape_flat, pred_flat)[0, 1]
    
    return {
        'regime_analysis': regime_analysis,
        'overall_metrics': {
            'cape_prediction_correlation': float(cape_pred_corr) if not np.isnan(cape_pred_corr) else 0.0,
            'mean_prediction': float(pred_flat.mean()),
            'high_confidence_pixels': int((pred_flat > 0.7).sum())
        }
    }

def save_enhanced_predictions(predictions: torch.Tensor, output_file: str):
    """Save predictions to a NetCDF file."""
    pred_np = predictions.squeeze(0).squeeze(0).numpy()
    height, width = pred_np.shape
    
    ds = xr.Dataset(
        {'lightning_probability': (['lat', 'lon'], pred_np)},
        coords={'lat': np.arange(height), 'lon': np.arange(width)}
    )
    ds.lightning_probability.attrs = {'long_name': 'Lightning occurrence probability', 'units': 'probability'}
    ds.attrs = {'title': 'Lightning Prediction Results', 'creation_time': datetime.now().isoformat()}
    
    ds.to_netcdf(output_file)
    ds.close()
    logger.info(f"Saved predictions to {output_file}")

def save_physics_analysis_report(regime_stats: dict, physics_analysis: dict, output_dir: str):
    """Save detailed physics analysis report."""
    report = {'cape_regime_statistics': regime_stats, 'physics_alignment_analysis': physics_analysis}
    
    with open(f"{output_dir}/physics_analysis_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    md = f"# Lightning Inference - Physics Analysis Report\n\n"
    md += f"## CAPE-Prediction Correlation: {physics_analysis['overall_metrics']['cape_prediction_correlation']:.4f}\n\n"
    for regime, stats in regime_stats['regimes'].items():
        md += f"### {regime.replace('_', ' ').title()} Regime\n"
        md += f"- **Pixel Count**: {stats['count']:,} ({stats['percentage']:.2f}%)\n"
        if regime in physics_analysis['regime_analysis']:
            analysis = physics_analysis['regime_analysis'][regime]
            md += f"- **Mean Prediction Probability**: {analysis['predictions']['mean_probability']:.4f}\n"
    
    with open(f"{output_dir}/physics_analysis_summary.md", 'w') as f:
        f.write(md)

    logger.info(f"Saved physics analysis reports to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Lightning Prediction Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--cape-file", type=str, required=True, help="Path to CAPE NetCDF file")
    parser.add_argument("--terrain-file", type=str, required=True, help="Path to terrain NetCDF file")
    parser.add_argument("--output-file", type=str, required=True, help="Output NetCDF file for predictions")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (auto-detected if not provided)")
    parser.add_argument("--no-physics", action="store_true", help="Skip physics analysis")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        model = LightningTrainer.load_from_checkpoint(args.checkpoint, map_location=device)
        preprocessor = DataPreprocessor()
        input_data = load_inference_data(args.cape_file, args.terrain_file, preprocessor, model.config)
    except Exception as e:
        logger.error(f"Failed to load model or data: {e}")
        return 1
    
    try:
        results = run_inference_with_physics(model, input_data, device)
        predictions = results['lightning_probability']
        logger.info(f"Inference completed. Prediction shape: {predictions.shape}")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return 1
        
    save_enhanced_predictions(predictions, args.output_file)
    
    if not args.no_physics:
        try:
            regime_stats = analyze_cape_physics(input_data['raw_cape'], model.config)
            physics_analysis = analyze_prediction_physics(predictions, input_data['raw_cape'], regime_stats)
            output_dir = Path(args.output_file).parent
            save_physics_analysis_report(regime_stats, physics_analysis, output_dir)
        except Exception as e:
            logger.warning(f"Physics analysis failed: {e}")

    logger.info("Inference completed successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
