"""
Inference script for lightning prediction on new data.
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
    
    # Close datasets
    cape_ds.close()
    terrain_ds.close()
    
    return {
        'cape': cape_tensor.unsqueeze(0),  # Add batch dimension
        'terrain': terrain_tensor.unsqueeze(0)  # Add batch dimension
    }

def run_inference(model: LightningTrainer, 
                 input_data: dict,
                 device: torch.device) -> dict:
    """Run inference on input data."""
    
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        # Move data to device
        cape_data = input_data['cape'].to(device)
        terrain_data = input_data['terrain'].to(device)
        
        # Run inference
        outputs = model(cape_data, terrain_data)
        
        # Move results back to CPU
        results = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                results[key] = value.cpu()
            else:
                results[key] = value
    
    return results

def save_predictions(predictions: torch.Tensor,
                    output_file: str,
                    reference_file: str = None):
    """Save predictions to NetCDF file."""
    
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
    
    # Create dataset
    ds = xr.Dataset({
        'lightning_probability': (['lat', 'lon'], pred_np)
    }, coords={
        'lat': lat,
        'lon': lon
    })
    
    # Add metadata
    ds.lightning_probability.attrs = {
        'long_name': 'Lightning occurrence probability',
        'units': 'probability',
        'description': 'Predicted probability of lightning occurrence'
    }
    
    ds.attrs = {
        'title': 'Lightning Prediction Results',
        'creation_time': datetime.now().isoformat(),
        'model': 'LightningPredictor',
        'resolution': '3km'
    }
    
    # Save to file
    ds.to_netcdf(output_file)
    ds.close()
    
    logger.info(f"Saved predictions to {output_file}")

def main():
    """Main inference function."""
    
    parser = argparse.ArgumentParser(description="Lightning Prediction Inference")
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
        logger.info("Loading model...")
        model = LightningTrainer.load_from_checkpoint(args.checkpoint, map_location=device)
        model_info = model.model.get_model_info()
        logger.info(f"Model loaded: {model_info['total_parameters']:,} parameters")
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
    
    # Run inference
    try:
        logger.info("Running inference...")
        results = run_inference(model, input_data, device)
        
        predictions = results['lightning_prediction']
        logger.info(f"Inference completed. Prediction shape: {predictions.shape}")
        
        # Log prediction statistics
        pred_np = predictions.numpy()
        logger.info(f"Prediction statistics:")
        logger.info(f"  Min: {pred_np.min():.6f}")
        logger.info(f"  Max: {pred_np.max():.6f}")
        logger.info(f"  Mean: {pred_np.mean():.6f}")
        logger.info(f"  Lightning probability > 0.5: {(pred_np > 0.5).sum()} pixels")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return 1
    
    # Save predictions
    try:
        logger.info("Saving predictions...")
        save_predictions(predictions, args.output_file, args.reference_file)
        
    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")
        return 1
    
    # Save intermediate features if requested
    if args.save_features:
        try:
            features_file = args.output_file.replace('.nc', '_features.pt')
            torch.save({
                'cape_features': results.get('cape_features'),
                'terrain_features': results.get('terrain_features'),
                'fused_features': results.get('fused_features'),
                'gnn_features': results.get('gnn_features'),
                'transformer_features': results.get('transformer_features')
            }, features_file)
            
            logger.info(f"Saved intermediate features to {features_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save features: {e}")
    
    logger.info(f"Inference completed successfully. Results saved to {args.output_file}")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)