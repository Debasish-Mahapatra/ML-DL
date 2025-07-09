"""
Evaluation script for lightning prediction model.
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
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf

from src.utils.config import get_config
from src.data.data_loader import LightningDataModule
from src.training.trainer import LightningTrainer
from src.training.metrics import evaluate_model, LightningMetrics
from src.utils.visualization import create_prediction_plots, create_metric_plots

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model_from_checkpoint(checkpoint_path: str, config_path: str = None) -> LightningTrainer:
    """Load trained model from checkpoint."""
    
    # Load config
    if config_path:
        config = OmegaConf.load(config_path)
    else:
        # Try to find config in the same directory as checkpoint
        checkpoint_dir = Path(checkpoint_path).parent
        possible_config_paths = [
            checkpoint_dir / "config.yaml",
            checkpoint_dir.parent / "config.yaml",
            "config"
        ]
        
        config = None
        for config_path in possible_config_paths:
            try:
                if Path(config_path).is_dir():
                    config = get_config(str(config_path))
                else:
                    config = OmegaConf.load(config_path)
                logger.info(f"Loaded config from {config_path}")
                break
            except Exception:
                continue
        
        if config is None:
            raise FileNotFoundError("Could not find configuration file")
    
    # Load model
    model = LightningTrainer.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    return model

def evaluate_on_dataset(model: LightningTrainer, 
                        datamodule: LightningDataModule,
                        split: str = "test") -> dict:
    """Evaluate model on specified dataset split."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Get appropriate dataloader
    if split == "train":
        dataloader = datamodule.train_dataloader()
    elif split == "val":
        dataloader = datamodule.val_dataloader()
    elif split == "test":
        dataloader = datamodule.test_dataloader()
    else:
        raise ValueError(f"Unknown split: {split}")
    
    # Initialize metrics
    metrics = LightningMetrics(
        threshold=0.5,
        spatial_tolerance=1,
        compute_spatial_metrics=True,
        compute_probabilistic_metrics=True
    )
    
    # Evaluate
    logger.info(f"Evaluating on {split} set...")
    computed_metrics = evaluate_model(model, dataloader, metrics, device)
    
    logger.info(f"Evaluation on {split} set completed")
    return computed_metrics

def generate_predictions(model: LightningTrainer,
                        datamodule: LightningDataModule,
                        output_dir: str,
                        num_samples: int = 10):
    """Generate and save prediction samples."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Get test dataloader
    test_loader = datamodule.test_dataloader()
    
    predictions_list = []
    targets_list = []
    cape_list = []
    
    sample_count = 0
    
    logger.info(f"Generating {num_samples} prediction samples...")
    
    with torch.no_grad():
        for batch in test_loader:
            if sample_count >= num_samples:
                break
            
            # Move to device
            cape_data = batch['cape'].to(device)
            terrain_data = batch['terrain'].to(device)
            lightning_targets = batch['lightning']
            
            # Generate predictions
            outputs = model(cape_data, terrain_data)
            predictions = outputs['lightning_prediction'].cpu()
            
            # Store samples
            batch_size = predictions.shape[0]
            for i in range(min(batch_size, num_samples - sample_count)):
                predictions_list.append(predictions[i])
                targets_list.append(lightning_targets[i])
                cape_list.append(batch['cape'][i])
                sample_count += 1
    
    # Save predictions
    prediction_data = {
        'predictions': torch.stack(predictions_list),
        'targets': torch.stack(targets_list),
        'cape': torch.stack(cape_list)
    }
    
    torch.save(prediction_data, f"{output_dir}/prediction_samples.pt")
    logger.info(f"Saved {sample_count} prediction samples to {output_dir}")
    
    return prediction_data

def create_evaluation_report(metrics: dict, 
                           model_info: dict,
                           output_dir: str):
    """Create comprehensive evaluation report."""
    
    report = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "model_info": model_info,
        "metrics": metrics
    }
    
    # Save JSON report
    with open(f"{output_dir}/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    # Create markdown report
    markdown_content = f"""# Lightning Prediction Model - Evaluation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information
- Total Parameters: {model_info.get('total_parameters', 'N/A'):,}
- Model Size: {model_info.get('model_size_mb', 'N/A'):.1f} MB
- CAPE-only Mode: {model_info.get('cape_only_mode', 'N/A')}
- Domain Adaptation: {model_info.get('domain_adaptation_enabled', 'N/A')}

## Performance Metrics

### Classification Metrics
- **Accuracy**: {metrics.get('accuracy', 0):.4f}
- **Precision**: {metrics.get('precision', 0):.4f}
- **Recall**: {metrics.get('recall', 0):.4f}
- **F1 Score**: {metrics.get('f1_score', 0):.4f}
- **Balanced Accuracy**: {metrics.get('balanced_accuracy', 0):.4f}

### Lightning-Specific Metrics
- **Lightning Detection Rate**: {metrics.get('lightning_detection_rate', 0):.4f}
- **False Alarm Ratio**: {metrics.get('false_alarm_ratio', 0):.4f}
- **Critical Success Index**: {metrics.get('critical_success_index', 0):.4f}
- **Heidke Skill Score**: {metrics.get('heidke_skill_score', 0):.4f}

### Probabilistic Metrics
- **ROC AUC**: {metrics.get('roc_auc', 0):.4f}
- **Average Precision**: {metrics.get('average_precision', 0):.4f}
- **Brier Score**: {metrics.get('brier_score', 0):.4f}

### Spatial Metrics
- **Spatial Accuracy**: {metrics.get('spatial_accuracy', 0):.4f}
- **Spatial F1**: {metrics.get('spatial_f1', 0):.4f}

### Confusion Matrix
- True Positives: {metrics.get('true_positives', 0)}
- False Positives: {metrics.get('false_positives', 0)}
- True Negatives: {metrics.get('true_negatives', 0)}
- False Negatives: {metrics.get('false_negatives', 0)}
"""
    
    with open(f"{output_dir}/evaluation_report.md", "w") as f:
        f.write(markdown_content)
    
    logger.info(f"Created evaluation report in {output_dir}")

def plot_metrics(metrics: dict, output_dir: str):
    """Create visualization plots for metrics."""
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Classification metrics
    classification_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    classification_values = [metrics.get(m, 0) for m in classification_metrics]
    
    axes[0, 0].bar(classification_metrics, classification_values, color='skyblue')
    axes[0, 0].set_title('Classification Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(classification_values):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # Lightning-specific metrics
    lightning_metrics = ['lightning_detection_rate', 'false_alarm_ratio', 'critical_success_index']
    lightning_values = [metrics.get(m, 0) for m in lightning_metrics]
    
    axes[0, 1].bar(lightning_metrics, lightning_values, color='lightcoral')
    axes[0, 1].set_title('Lightning-Specific Metrics')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(lightning_values):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    # Confusion Matrix
    tp = metrics.get('true_positives', 0)
    fp = metrics.get('false_positives', 0)
    tn = metrics.get('true_negatives', 0)
    fn = metrics.get('false_negatives', 0)
    
    conf_matrix = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Lightning', 'Lightning'],
                yticklabels=['No Lightning', 'Lightning'],
                ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')
    
    # Probabilistic metrics
    prob_metrics = ['roc_auc', 'average_precision', 'brier_score']
    prob_values = [metrics.get(m, 0) for m in prob_metrics]
    
    axes[1, 1].bar(prob_metrics, prob_values, color='lightgreen')
    axes[1, 1].set_title('Probabilistic Metrics')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_ylim(0, 1)
    for i, v in enumerate(prob_values):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_overview.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Created metrics visualization in {output_dir}")

def main():
    """Main evaluation function."""
    
    parser = argparse.ArgumentParser(description="Evaluate Lightning Prediction Model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file (auto-detected if not provided)")
    parser.add_argument("--data-config", type=str, default=None,
                       help="Override data configuration directory")
    parser.add_argument("--output-dir", type=str, 
                       default=f"outputs/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help="Output directory for results")
    parser.add_argument("--splits", nargs="+", default=["test"],
                       choices=["train", "val", "test"],
                       help="Dataset splits to evaluate")
    parser.add_argument("--num-samples", type=int, default=10,
                       help="Number of prediction samples to generate")
    parser.add_argument("--generate-plots", action="store_true",
                       help="Generate visualization plots")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Override batch size for evaluation")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    try:
        model = load_model_from_checkpoint(args.checkpoint, args.config)
        model_info = model.model.get_model_info()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Setup data module
    try:
        if args.data_config:
            config = get_config(args.data_config)
        else:
            config = model.config
