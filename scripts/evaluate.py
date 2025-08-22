"""
Enhanced evaluation script for lightning prediction model with physics analysis.
VERSION: Streamlined to remove plotting functions and add compatibility for 
         both single-stage and two-stage model outputs.
"""

import os
import sys
import argparse
from pathlib import Path
import logging
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.utils.config import get_config
from src.data.data_loader import LightningDataModule
from src.training.trainer import LightningTrainer
from src.training.metrics import LightningMetrics, evaluate_model

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
    else:
        # Old models return a single tensor
        return model_output

def load_model_from_checkpoint(checkpoint_path: str, config_path: str = None) -> LightningTrainer:
    """Load trained model from checkpoint."""
    if config_path:
        config = OmegaConf.load(config_path)
    else:
        checkpoint_dir = Path(checkpoint_path).parent
        possible_config_paths = [
            checkpoint_dir / "config.yaml",
            checkpoint_dir.parent / "config.yaml",
            "config"
        ]
        config = None
        for path in possible_config_paths:
            try:
                if path.is_dir():
                    config = get_config(str(path))
                else:
                    config = OmegaConf.load(path)
                logger.info(f"Loaded config from {path}")
                break
            except Exception:
                continue
        if config is None:
            raise FileNotFoundError("Could not find a valid configuration file.")
    
    model = LightningTrainer.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    logger.info(f"Loaded model from {checkpoint_path}")
    return model

def evaluate_cape_physics_performance(model: LightningTrainer,
                                    datamodule: LightningDataModule,
                                    split: str = "test") -> dict:
    """Evaluate model performance across different CAPE regimes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    dataloader = getattr(datamodule, f"{split}_dataloader")()
    
    cape_thresholds = model.config.training.get('cape_physics', {}).get('thresholds', {
        'no_lightning': 1000.0, 'moderate': 2500.0, 'high': 4000.0
    })
    
    regime_data = {
        'no_lightning': {'predictions': [], 'targets': [], 'cape_values': []},
        'moderate': {'predictions': [], 'targets': [], 'cape_values': []},
        'high': {'predictions': [], 'targets': [], 'cape_values': []},
        'very_high': {'predictions': [], 'targets': [], 'cape_values': []}
    }
    
    all_predictions, all_targets, all_cape = [], [], []
    
    logger.info(f"Analyzing CAPE physics performance on {split} set...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i % 10 == 0:
                logger.info(f"Processing batch {i}/{len(dataloader)}")
            
            cape_data = batch['cape'].to(device)
            terrain_data = batch['terrain'].to(device)
            lightning_targets = batch['lightning']
            
            raw_outputs = model(cape_data, terrain_data)
            lightning_logits = get_lightning_prediction(raw_outputs)
            predictions = torch.sigmoid(lightning_logits).cpu()
            
            cape_flat = cape_data.cpu().flatten().numpy()
            pred_flat = predictions.flatten().numpy()
            target_flat = lightning_targets.flatten().numpy()
            
            all_predictions.extend(pred_flat)
            all_targets.extend(target_flat)
            all_cape.extend(cape_flat)
            
            for j in range(len(cape_flat)):
                cape_val = cape_flat[j]
                if cape_val < cape_thresholds['no_lightning']: regime = 'no_lightning'
                elif cape_val < cape_thresholds['moderate']: regime = 'moderate'
                elif cape_val < cape_thresholds['high']: regime = 'high'
                else: regime = 'very_high'
                
                regime_data[regime]['predictions'].append(pred_flat[j])
                regime_data[regime]['targets'].append(target_flat[j])
                regime_data[regime]['cape_values'].append(cape_val)

    all_cape_np = np.array(all_cape)
    physics_analysis = {
        'overall_cape_stats': {
            'min': float(all_cape_np.min()), 'max': float(all_cape_np.max()),
            'mean': float(all_cape_np.mean()), 'std': float(all_cape_np.std()),
            'median': float(np.median(all_cape_np))
        },
        'regime_analysis': {}
    }
    
    for regime, data in regime_data.items():
        if not data['predictions']: continue
        
        preds, targets = np.array(data['predictions']), np.array(data['targets'])
        binary_preds = (preds > 0.5).astype(int)
        
        regime_metrics = {
            'sample_count': len(preds),
            'percentage_of_total': len(preds) / len(all_predictions) * 100,
            'lightning_rates': {'true_rate': float(targets.mean()), 'predicted_rate': float(binary_preds.mean())},
            'prediction_stats': {'mean_probability': float(preds.mean()), 'std_probability': float(preds.std())}
        }
        
        if len(np.unique(targets)) > 1:
            try:
                regime_metrics['classification_metrics'] = {
                    'f1_score': float(f1_score(targets, binary_preds, zero_division=0)),
                    'roc_auc': float(roc_auc_score(targets, preds))
                }
            except Exception as e:
                logger.warning(f"Could not compute metrics for regime {regime}: {e}")
        
        physics_analysis['regime_analysis'][regime] = regime_metrics
        
    return physics_analysis

def evaluate_on_dataset(model: LightningTrainer, 
                        datamodule: LightningDataModule,
                        split: str = "test") -> dict:
    """Evaluate model on specified dataset split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    dataloader = getattr(datamodule, f"{split}_dataloader")()
    
    metrics = LightningMetrics(
        threshold=0.5, spatial_tolerance=1,
        compute_spatial_metrics=True, compute_probabilistic_metrics=True
    )
    
    logger.info(f"Evaluating on {split} set...")
    # This now uses the imported evaluate_model helper, which is assumed to handle the loop.
    # The key is that the model's predict_step is what gets called, and it's already updated.
    computed_metrics = evaluate_model(model, dataloader, metrics, device)
    
    logger.info(f"Evaluation on {split} set completed")
    return computed_metrics

def generate_predictions(model: LightningTrainer,
                        datamodule: LightningDataModule,
                        output_dir: str,
                        num_samples: int = 10):
    """Generate and save prediction samples for manual analysis."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    test_loader = datamodule.test_dataloader()
    
    predictions_list, targets_list, cape_list = [], [], []
    sample_count = 0
    
    logger.info(f"Generating {num_samples} prediction samples...")
    with torch.no_grad():
        for batch in test_loader:
            if sample_count >= num_samples: break
            
            cape_data = batch['cape'].to(device)
            terrain_data = batch['terrain'].to(device)
            
            raw_outputs = model(cape_data, terrain_data)
            lightning_logits = get_lightning_prediction(raw_outputs)
            predictions = torch.sigmoid(lightning_logits).cpu()
            
            batch_size = predictions.shape[0]
            for i in range(min(batch_size, num_samples - sample_count)):
                predictions_list.append(predictions[i])
                targets_list.append(batch['lightning'][i])
                cape_list.append(batch['cape'][i])
                sample_count += 1
    
    prediction_data = {
        'predictions': torch.stack(predictions_list),
        'targets': torch.stack(targets_list),
        'cape': torch.stack(cape_list)
    }
    
    torch.save(prediction_data, f"{output_dir}/prediction_samples.pt")
    logger.info(f"Saved {sample_count} prediction samples to {output_dir}")
    return prediction_data

def create_enhanced_evaluation_report(metrics: dict, 
                                     physics_analysis: dict,
                                     model_info: dict,
                                     output_dir: str):
    """Create comprehensive JSON and Markdown evaluation reports."""
    report = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "model_info": model_info,
        "standard_metrics": metrics,
        "physics_analysis": physics_analysis
    }
    
    with open(f"{output_dir}/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    md = f"# Lightning Prediction Model - Evaluation Report\n\n"
    md += f"## Model Information\n- **Total Parameters**: {model_info.get('total_parameters', 'N/A'):,}\n"
    md += f"- **Model Size**: {model_info.get('model_size_mb', 'N/A'):.1f} MB\n\n"
    md += f"## Standard Performance Metrics\n"
    md += f"- **F1 Score**: {metrics.get('f1_score', 0):.4f}\n"
    md += f"- **ROC AUC**: {metrics.get('roc_auc', 0):.4f}\n"
    md += f"- **Critical Success Index**: {metrics.get('critical_success_index', 0):.4f}\n\n"
    md += f"## CAPE Physics Analysis\n"
    
    for regime, analysis in physics_analysis.get('regime_analysis', {}).items():
        md += f"#### {regime.replace('_', ' ').title()} Regime\n"
        md += f"- **Sample Count**: {analysis['sample_count']:,} ({analysis['percentage_of_total']:.2f}% of total)\n"
        md += f"- **True Lightning Rate**: {analysis['lightning_rates']['true_rate']:.4f}\n"
        md += f"- **Predicted Lightning Rate**: {analysis['lightning_rates']['predicted_rate']:.4f}\n"
        md += f"- **Mean Prediction Probability**: {analysis['prediction_stats']['mean_probability']:.4f}\n\n"
        
    with open(f"{output_dir}/evaluation_report.md", "w") as f:
        f.write(md)
    
    logger.info(f"Created evaluation reports in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Lightning Prediction Model Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (auto-detected if not provided)")
    parser.add_argument("--output-dir", type=str, default=f"outputs/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}", help="Output directory for results")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of prediction samples to generate")
    parser.add_argument("--skip-physics", action="store_true", help="Skip detailed physics analysis for faster evaluation")
    
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        model = load_model_from_checkpoint(args.checkpoint, args.config)
        model_info = model.get_model_info()
        datamodule = LightningDataModule(model.config)
        datamodule.setup()
    except Exception as e:
        logger.error(f"Failed to load model or data: {e}")
        return 1
    
    metrics = evaluate_on_dataset(model, datamodule, args.split)
    
    physics_analysis = {}
    if not args.skip_physics:
        physics_analysis = evaluate_cape_physics_performance(model, datamodule, args.split)
    
    if args.num_samples > 0:
        generate_predictions(model, datamodule, args.output_dir, args.num_samples)
    
    create_enhanced_evaluation_report(metrics, physics_analysis, model_info, args.output_dir)
    
    logger.info("=== EVALUATION SUMMARY ===")
    logger.info(f"{args.split.upper()} Results:")
    logger.info(f"  F1 Score: {metrics.get('f1_score', 0):.4f}")
    logger.info(f"  ROC AUC: {metrics.get('roc_auc', 0):.4f}")
    logger.info(f"Detailed results saved to: {args.output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
