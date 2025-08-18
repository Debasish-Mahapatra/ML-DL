"""
Enhanced evaluation script for lightning prediction model with physics analysis.
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

def evaluate_cape_physics_performance(model: LightningTrainer,
                                    datamodule: LightningDataModule,
                                    split: str = "test") -> dict:
    """
    Evaluate model performance across different CAPE regimes.
    This analyzes how well the enhanced physics constraints are working.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Get dataloader
    if split == "train":
        dataloader = datamodule.train_dataloader()
    elif split == "val":
        dataloader = datamodule.val_dataloader()
    elif split == "test":
        dataloader = datamodule.test_dataloader()
    else:
        raise ValueError(f"Unknown split: {split}")
    
    # CAPE thresholds (from enhanced physics)
    cape_thresholds = {
        'no_lightning': 1000.0,
        'moderate': 2500.0,
        'high': 4000.0,
        'saturation': 5000.0
    }
    
    # Storage for CAPE regime analysis
    regime_data = {
        'no_lightning': {'predictions': [], 'targets': [], 'cape_values': []},
        'moderate': {'predictions': [], 'targets': [], 'cape_values': []},
        'high': {'predictions': [], 'targets': [], 'cape_values': []},
        'very_high': {'predictions': [], 'targets': [], 'cape_values': []}
    }
    
    # Overall statistics
    all_predictions = []
    all_targets = []
    all_cape = []
    
    logger.info(f"Analyzing CAPE physics performance on {split} set...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:
                logger.info(f"Processing batch {batch_idx}/{len(dataloader)}")
            
            # Move to device
            cape_data = batch['cape'].to(device)
            terrain_data = batch['terrain'].to(device)
            lightning_targets = batch['lightning']
            
            # Generate predictions
            outputs = model(cape_data, terrain_data)
            predictions = torch.sigmoid(outputs['lightning_prediction']).cpu()  # Convert logits to probabilities
            
            # Flatten for analysis
            cape_flat = cape_data.cpu().flatten()
            pred_flat = predictions.flatten()
            target_flat = lightning_targets.flatten()
            
            # Store overall data
            all_predictions.extend(pred_flat.numpy())
            all_targets.extend(target_flat.numpy())
            all_cape.extend(cape_flat.numpy())
            
            # Categorize by CAPE regime
            for i in range(len(cape_flat)):
                cape_val = cape_flat[i].item()
                pred_val = pred_flat[i].item()
                target_val = target_flat[i].item()
                
                if cape_val < cape_thresholds['no_lightning']:
                    regime = 'no_lightning'
                elif cape_val < cape_thresholds['moderate']:
                    regime = 'moderate'
                elif cape_val < cape_thresholds['high']:
                    regime = 'high'
                else:
                    regime = 'very_high'
                
                regime_data[regime]['predictions'].append(pred_val)
                regime_data[regime]['targets'].append(target_val)
                regime_data[regime]['cape_values'].append(cape_val)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_cape = np.array(all_cape)
    
    # Analyze each regime
    physics_analysis = {
        'overall_cape_stats': {
            'min': float(all_cape.min()),
            'max': float(all_cape.max()),
            'mean': float(all_cape.mean()),
            'std': float(all_cape.std()),
            'median': float(np.median(all_cape))
        },
        'regime_analysis': {}
    }
    
    for regime, data in regime_data.items():
        if len(data['predictions']) == 0:
            continue
        
        preds = np.array(data['predictions'])
        targets = np.array(data['targets'])
        cape_vals = np.array(data['cape_values'])
        
        # Calculate metrics for this regime
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Binary predictions (threshold 0.5)
        binary_preds = (preds > 0.5).astype(int)
        
        # Calculate lightning occurrence rates
        true_lightning_rate = targets.mean()
        predicted_lightning_rate = binary_preds.mean()
        
        # Physics consistency metrics
        regime_metrics = {
            'sample_count': len(preds),
            'percentage_of_total': len(preds) / len(all_predictions) * 100,
            'cape_range': {
                'min': float(cape_vals.min()),
                'max': float(cape_vals.max()),
                'mean': float(cape_vals.mean()),
                'std': float(cape_vals.std())
            },
            'lightning_rates': {
                'true_rate': float(true_lightning_rate),
                'predicted_rate': float(predicted_lightning_rate),
                'rate_difference': float(predicted_lightning_rate - true_lightning_rate)
            },
            'prediction_stats': {
                'mean_probability': float(preds.mean()),
                'std_probability': float(preds.std()),
                'min_probability': float(preds.min()),
                'max_probability': float(preds.max())
            }
        }
        
        # Standard metrics (if we have both classes)
        if len(np.unique(targets)) > 1:
            try:
                regime_metrics['classification_metrics'] = {
                    'accuracy': float(accuracy_score(targets, binary_preds)),
                    'precision': float(precision_score(targets, binary_preds, zero_division=0)),
                    'recall': float(recall_score(targets, binary_preds, zero_division=0)),
                    'f1_score': float(f1_score(targets, binary_preds, zero_division=0)),
                    'roc_auc': float(roc_auc_score(targets, preds)) if len(np.unique(targets)) > 1 else 0.0
                }
            except Exception as e:
                logger.warning(f"Could not compute metrics for regime {regime}: {e}")
                regime_metrics['classification_metrics'] = {}
        
        physics_analysis['regime_analysis'][regime] = regime_metrics
    
    # Physics consistency analysis
    physics_analysis['physics_consistency'] = analyze_physics_consistency(
        all_cape, all_predictions, all_targets, cape_thresholds
    )
    
    return physics_analysis

def analyze_physics_consistency(cape_values, predictions, targets, thresholds):
    """Analyze how well predictions follow physics expectations."""
    
    consistency_analysis = {}
    
    # Expected vs actual lightning rates by CAPE regime
    regimes = [
        ('no_lightning', cape_values < thresholds['no_lightning']),
        ('moderate', (cape_values >= thresholds['no_lightning']) & (cape_values < thresholds['moderate'])),
        ('high', (cape_values >= thresholds['moderate']) & (cape_values < thresholds['high'])),
        ('very_high', cape_values >= thresholds['high'])
    ]
    
    expected_rates = {
        'no_lightning': 0.05,    # Very low lightning expected
        'moderate': 0.20,        # Moderate lightning expected
        'high': 0.50,           # High lightning expected
        'very_high': 0.70       # Very high lightning expected
    }
    
    for regime_name, mask in regimes:
        if mask.sum() == 0:
            continue
        
        regime_preds = predictions[mask]
        regime_targets = targets[mask]
        
        actual_rate = (regime_preds > 0.5).mean()
        expected_rate = expected_rates[regime_name]
        true_rate = regime_targets.mean()
        
        consistency_analysis[f'{regime_name}_consistency'] = {
            'expected_rate': expected_rate,
            'actual_predicted_rate': float(actual_rate),
            'true_lightning_rate': float(true_rate),
            'physics_alignment': float(1.0 - abs(actual_rate - expected_rate)),
            'prediction_accuracy': float(1.0 - abs(actual_rate - true_rate)) if true_rate > 0 else 0.0
        }
    
    # CAPE-prediction correlation
    cape_pred_correlation = np.corrcoef(cape_values, predictions)[0, 1]
    cape_target_correlation = np.corrcoef(cape_values, targets)[0, 1]
    
    consistency_analysis['correlations'] = {
        'cape_prediction_correlation': float(cape_pred_correlation) if not np.isnan(cape_pred_correlation) else 0.0,
        'cape_target_correlation': float(cape_target_correlation) if not np.isnan(cape_target_correlation) else 0.0,
        'correlation_improvement': float(cape_pred_correlation - cape_target_correlation) if not np.isnan(cape_pred_correlation) and not np.isnan(cape_target_correlation) else 0.0
    }
    
    return consistency_analysis

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
            predictions = torch.sigmoid(outputs['lightning_prediction']).cpu()  # Convert to probabilities
            
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

def create_enhanced_evaluation_report(metrics: dict, 
                                     physics_analysis: dict,
                                     model_info: dict,
                                     output_dir: str):
    """Create comprehensive evaluation report with physics analysis."""
    
    # Create comprehensive report
    report = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "model_info": model_info,
        "standard_metrics": metrics,
        "physics_analysis": physics_analysis
    }
    
    # Save JSON report
    with open(f"{output_dir}/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    # Create enhanced markdown report
    markdown_content = f"""# Lightning Prediction Model - Enhanced Evaluation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information
- Total Parameters: {model_info.get('total_parameters', 'N/A'):,}
- Model Size: {model_info.get('model_size_mb', 'N/A'):.1f} MB
- CAPE-only Mode: {model_info.get('cape_only_mode', 'N/A')}
- Domain Adaptation: {model_info.get('domain_adaptation_enabled', 'N/A')}

## Standard Performance Metrics

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

## CAPE Physics Analysis

### Overall CAPE Statistics
- **CAPE Range**: {physics_analysis['overall_cape_stats']['min']:.1f} - {physics_analysis['overall_cape_stats']['max']:.1f} J/kg
- **Mean CAPE**: {physics_analysis['overall_cape_stats']['mean']:.1f} J/kg
- **CAPE Standard Deviation**: {physics_analysis['overall_cape_stats']['std']:.1f} J/kg

### CAPE Regime Performance
"""
    
    # Add regime analysis
    for regime, analysis in physics_analysis['regime_analysis'].items():
        regime_name = regime.replace('_', ' ').title()
        markdown_content += f"""
#### {regime_name} Regime
- **Sample Count**: {analysis['sample_count']:,} ({analysis['percentage_of_total']:.2f}% of total)
- **CAPE Range**: {analysis['cape_range']['min']:.1f} - {analysis['cape_range']['max']:.1f} J/kg
- **True Lightning Rate**: {analysis['lightning_rates']['true_rate']:.4f}
- **Predicted Lightning Rate**: {analysis['lightning_rates']['predicted_rate']:.4f}
- **Mean Prediction Probability**: {analysis['prediction_stats']['mean_probability']:.4f}
"""
        
        if 'classification_metrics' in analysis and analysis['classification_metrics']:
            cm = analysis['classification_metrics']
            markdown_content += f"""- **F1 Score**: {cm.get('f1_score', 0):.4f}
- **ROC AUC**: {cm.get('roc_auc', 0):.4f}
"""
    
    # Add physics consistency analysis
    if 'physics_consistency' in physics_analysis:
        pc = physics_analysis['physics_consistency']
        markdown_content += f"""
### Physics Consistency Analysis

#### CAPE-Lightning Correlations
- **CAPE-Prediction Correlation**: {pc['correlations']['cape_prediction_correlation']:.4f}
- **CAPE-Target Correlation**: {pc['correlations']['cape_target_correlation']:.4f}
- **Correlation Improvement**: {pc['correlations']['correlation_improvement']:.4f}

#### Physics Regime Alignment
"""
        for key, value in pc.items():
            if key.endswith('_consistency'):
                regime = key.replace('_consistency', '').replace('_', ' ').title()
                markdown_content += f"""
##### {regime} Regime
- **Expected Rate**: {value['expected_rate']:.2f}
- **Actual Predicted Rate**: {value['actual_predicted_rate']:.4f}
- **Physics Alignment Score**: {value['physics_alignment']:.4f}
"""
    
    markdown_content += f"""
### Confusion Matrix
- True Positives: {metrics.get('true_positives', 0)}
- False Positives: {metrics.get('false_positives', 0)}
- True Negatives: {metrics.get('true_negatives', 0)}
- False Negatives: {metrics.get('false_negatives', 0)}

## Summary

This evaluation demonstrates the performance of the enhanced lightning prediction model with realistic CAPE physics constraints. The physics analysis shows how well the model learned the relationship between CAPE values and lightning occurrence across different atmospheric regimes.

**Key Insights:**
- The model shows {'good' if metrics.get('f1_score', 0) > 0.3 else 'moderate' if metrics.get('f1_score', 0) > 0.1 else 'poor'} overall performance with F1 score of {metrics.get('f1_score', 0):.4f}
- CAPE-prediction correlation of {physics_analysis['physics_consistency']['correlations']['cape_prediction_correlation']:.4f} indicates {'strong' if physics_analysis['physics_consistency']['correlations']['cape_prediction_correlation'] > 0.5 else 'moderate' if physics_analysis['physics_consistency']['correlations']['cape_prediction_correlation'] > 0.3 else 'weak'} physics learning
- Physics constraints {'effectively' if physics_analysis['physics_consistency']['correlations']['cape_prediction_correlation'] > physics_analysis['physics_consistency']['correlations']['cape_target_correlation'] else 'need adjustment to'} guide the model predictions

Generated with enhanced physics-aware evaluation framework.
"""
    
    with open(f"{output_dir}/evaluation_report.md", "w") as f:
        f.write(markdown_content)
    
    logger.info(f"Created enhanced evaluation report in {output_dir}")

def plot_enhanced_metrics(metrics: dict, physics_analysis: dict, output_dir: str):
    """Create enhanced visualization plots including physics analysis."""
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    
    # Create multi-panel figure
    fig = plt.figure(figsize=(20, 15))
    
    # Standard metrics (top row)
    ax1 = plt.subplot(3, 3, 1)
    classification_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    classification_values = [metrics.get(m, 0) for m in classification_metrics]
    bars1 = ax1.bar(classification_metrics, classification_values, color='skyblue')
    ax1.set_title('Classification Metrics', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    for bar, val in zip(bars1, classification_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom')
    
    ax2 = plt.subplot(3, 3, 2)
    lightning_metrics = ['lightning_detection_rate', 'false_alarm_ratio', 'critical_success_index']
    lightning_values = [metrics.get(m, 0) for m in lightning_metrics]
    bars2 = ax2.bar([m.replace('_', '\n') for m in lightning_metrics], lightning_values, color='lightcoral')
    ax2.set_title('Lightning-Specific Metrics', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1)
    for bar, val in zip(bars2, lightning_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom')
    
    # Confusion Matrix
    ax3 = plt.subplot(3, 3, 3)
    tp = metrics.get('true_positives', 0)
    fp = metrics.get('false_positives', 0)
    tn = metrics.get('true_negatives', 0)
    fn = metrics.get('false_negatives', 0)
    conf_matrix = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Lightning', 'Lightning'],
                yticklabels=['No Lightning', 'Lightning'], ax=ax3)
    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # CAPE regime analysis (middle row)
    ax4 = plt.subplot(3, 3, 4)
    regime_names = list(physics_analysis['regime_analysis'].keys())
    regime_counts = [physics_analysis['regime_analysis'][r]['sample_count'] for r in regime_names]
    ax4.pie(regime_counts, labels=[r.replace('_', ' ').title() for r in regime_names], autopct='%1.1f%%')
    ax4.set_title('CAPE Regime Distribution', fontsize=14, fontweight='bold')
    
    ax5 = plt.subplot(3, 3, 5)
    lightning_rates_true = [physics_analysis['regime_analysis'][r]['lightning_rates']['true_rate'] for r in regime_names]
    lightning_rates_pred = [physics_analysis['regime_analysis'][r]['lightning_rates']['predicted_rate'] for r in regime_names]
    x = np.arange(len(regime_names))
    width = 0.35
    ax5.bar(x - width/2, lightning_rates_true, width, label='True Rate', color='orange', alpha=0.7)
    ax5.bar(x + width/2, lightning_rates_pred, width, label='Predicted Rate', color='blue', alpha=0.7)
    ax5.set_xlabel('CAPE Regime')
    ax5.set_ylabel('Lightning Rate')
    ax5.set_title('Lightning Rates by CAPE Regime', fontsize=14, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([r.replace('_', '\n') for r in regime_names])
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Physics consistency (bottom row)
    ax6 = plt.subplot(3, 3, 6)
    if 'physics_consistency' in physics_analysis:
        consistency_regimes = []
        physics_alignment = []
        for key, value in physics_analysis['physics_consistency'].items():
            if key.endswith('_consistency'):
                regime = key.replace('_consistency', '').replace('_', ' ').title()
                consistency_regimes.append(regime)
                physics_alignment.append(value['physics_alignment'])
        
        if consistency_regimes:
            bars6 = ax6.bar(consistency_regimes, physics_alignment, color='green', alpha=0.7)
            ax6.set_title('Physics Alignment by Regime', fontsize=14, fontweight='bold')
            ax6.set_ylabel('Alignment Score')
            ax6.set_ylim(0, 1)
            for bar, val in zip(bars6, physics_alignment):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{val:.3f}', ha='center', va='bottom')
    
    # CAPE statistics
    ax7 = plt.subplot(3, 3, 7)
    cape_stats = physics_analysis['overall_cape_stats']
    stats_names = ['Min', 'Mean', 'Median', 'Max']
    stats_values = [cape_stats['min'], cape_stats['mean'], cape_stats['median'], cape_stats['max']]
    ax7.bar(stats_names, stats_values, color='purple', alpha=0.7)
    ax7.set_title('CAPE Statistics (J/kg)', fontsize=14, fontweight='bold')
    ax7.set_ylabel('CAPE (J/kg)')
    
    # Correlations
    ax8 = plt.subplot(3, 3, 8)
    if 'correlations' in physics_analysis['physics_consistency']:
        corr_data = physics_analysis['physics_consistency']['correlations']
        corr_names = ['CAPE-Prediction', 'CAPE-Target']
        corr_values = [corr_data['cape_prediction_correlation'], corr_data['cape_target_correlation']]
        bars8 = ax8.bar(corr_names, corr_values, color=['red', 'blue'], alpha=0.7)
        ax8.set_title('CAPE Correlations', fontsize=14, fontweight='bold')
        ax8.set_ylabel('Correlation Coefficient')
        ax8.set_ylim(-1, 1)
        ax8.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        for bar, val in zip(bars8, corr_values):
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.05, 
                    f'{val:.3f}', ha='center', va='bottom' if val >= 0 else 'top')
    
    # Regime performance comparison
    ax9 = plt.subplot(3, 3, 9)
    if regime_names:
        f1_scores = []
        for regime in regime_names:
            regime_data = physics_analysis['regime_analysis'][regime]
            if 'classification_metrics' in regime_data and 'f1_score' in regime_data['classification_metrics']:
                f1_scores.append(regime_data['classification_metrics']['f1_score'])
            else:
                f1_scores.append(0.0)
        
        bars9 = ax9.bar([r.replace('_', '\n') for r in regime_names], f1_scores, color='teal', alpha=0.7)
        ax9.set_title('F1 Score by CAPE Regime', fontsize=14, fontweight='bold')
        ax9.set_ylabel('F1 Score')
        ax9.set_ylim(0, 1)
        for bar, val in zip(bars9, f1_scores):
            ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/enhanced_metrics_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create CAPE distribution plot
    create_cape_distribution_plot(physics_analysis, output_dir)
    
    logger.info(f"Created enhanced metrics visualization in {output_dir}")

def create_cape_distribution_plot(physics_analysis: dict, output_dir: str):
    """Create detailed CAPE distribution analysis plot."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # CAPE range by regime
    regime_data = physics_analysis['regime_analysis']
    regimes = list(regime_data.keys())
    
    # Sample counts
    counts = [regime_data[r]['sample_count'] for r in regimes]
    percentages = [regime_data[r]['percentage_of_total'] for r in regimes]
    
    colors = ['lightcoral', 'orange', 'lightgreen', 'skyblue']
    bars1 = ax1.bar([r.replace('_', ' ').title() for r in regimes], counts, color=colors[:len(regimes)])
    ax1.set_title('Sample Distribution by CAPE Regime', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Samples')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add percentage labels
    for bar, pct in zip(bars1, percentages):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Mean CAPE by regime
    mean_capes = [regime_data[r]['cape_range']['mean'] for r in regimes]
    std_capes = [regime_data[r]['cape_range']['std'] for r in regimes]
    
    bars2 = ax2.bar([r.replace('_', ' ').title() for r in regimes], mean_capes, 
                   yerr=std_capes, color=colors[:len(regimes)], alpha=0.7, capsize=5)
    ax2.set_title('Mean CAPE by Regime (±1 std)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('CAPE (J/kg)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Lightning rates comparison
    true_rates = [regime_data[r]['lightning_rates']['true_rate'] for r in regimes]
    pred_rates = [regime_data[r]['lightning_rates']['predicted_rate'] for r in regimes]
    
    x = np.arange(len(regimes))
    width = 0.35
    ax3.bar(x - width/2, true_rates, width, label='Observed', color='red', alpha=0.7)
    ax3.bar(x + width/2, pred_rates, width, label='Predicted', color='blue', alpha=0.7)
    ax3.set_xlabel('CAPE Regime')
    ax3.set_ylabel('Lightning Occurrence Rate')
    ax3.set_title('Lightning Rates: Observed vs Predicted', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([r.replace('_', '\n') for r in regimes])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Prediction probability distributions
    mean_probs = [regime_data[r]['prediction_stats']['mean_probability'] for r in regimes]
    std_probs = [regime_data[r]['prediction_stats']['std_probability'] for r in regimes]
    
    bars4 = ax4.bar([r.replace('_', ' ').title() for r in regimes], mean_probs, 
                   yerr=std_probs, color=colors[:len(regimes)], alpha=0.7, capsize=5)
    ax4.set_title('Mean Prediction Probability by Regime (±1 std)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Prediction Probability')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cape_distribution_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main evaluation function with enhanced physics analysis."""
    
    parser = argparse.ArgumentParser(description="Enhanced Lightning Prediction Model Evaluation")
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
    parser.add_argument("--analyze-physics", action="store_true", default=True,
                       help="Perform detailed physics analysis (enabled by default)")
    parser.add_argument("--skip-physics", action="store_true",
                       help="Skip physics analysis (for faster evaluation)")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    try:
        model = load_model_from_checkpoint(args.checkpoint, args.config)
        model_info = {
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024,
            'cape_only_mode': getattr(model.model, 'cape_only_mode', True),
            'domain_adaptation_enabled': getattr(model.model, 'use_domain_adaptation', False)
        }
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1
    
    # Setup data module
    try:
        if args.data_config:
            config = get_config(args.data_config)
        else:
            config = model.config
        
        if args.batch_size:
            config.data.batch_size = args.batch_size
        
        datamodule = LightningDataModule(config)
        datamodule.setup()
    except Exception as e:
        logger.error(f"Failed to setup data module: {e}")
        return 1
    
    # Evaluate on specified splits
    all_metrics = {}
    all_physics_analysis = {}
    
    for split in args.splits:
        logger.info(f"Evaluating on {split} split...")
        
        # Standard evaluation
        try:
            metrics = evaluate_on_dataset(model, datamodule, split)
            all_metrics[split] = metrics
            logger.info(f"Standard evaluation completed for {split}")
        except Exception as e:
            logger.error(f"Standard evaluation failed for {split}: {e}")
            continue
        
        # Physics analysis
        if args.analyze_physics and not args.skip_physics:
            try:
                physics_analysis = evaluate_cape_physics_performance(model, datamodule, split)
                all_physics_analysis[split] = physics_analysis
                logger.info(f"Physics analysis completed for {split}")
            except Exception as e:
                logger.error(f"Physics analysis failed for {split}: {e}")
                all_physics_analysis[split] = {}
    
    # Generate prediction samples
    if args.num_samples > 0:
        try:
            prediction_data = generate_predictions(model, datamodule, args.output_dir, args.num_samples)
            logger.info("Prediction samples generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate prediction samples: {e}")
    
    # Create reports and visualizations
    for split in args.splits:
        if split not in all_metrics:
            continue
        
        split_output_dir = f"{args.output_dir}/{split}"
        Path(split_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create enhanced report
        if split in all_physics_analysis and all_physics_analysis[split]:
            create_enhanced_evaluation_report(
                all_metrics[split], 
                all_physics_analysis[split], 
                model_info, 
                split_output_dir
            )
        else:
            # Fallback to standard report if physics analysis failed
            logger.warning(f"Creating standard report for {split} (physics analysis not available)")
            create_standard_evaluation_report(all_metrics[split], model_info, split_output_dir)
        
        # Generate plots
        if args.generate_plots:
            try:
                if split in all_physics_analysis and all_physics_analysis[split]:
                    plot_enhanced_metrics(all_metrics[split], all_physics_analysis[split], split_output_dir)
                else:
                    plot_standard_metrics(all_metrics[split], split_output_dir)
            except Exception as e:
                logger.error(f"Failed to generate plots for {split}: {e}")
    
    # Summary
    logger.info("=== EVALUATION SUMMARY ===")
    for split in args.splits:
        if split in all_metrics:
            metrics = all_metrics[split]
            logger.info(f"{split.upper()} Results:")
            logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
            logger.info(f"  F1 Score: {metrics.get('f1_score', 0):.4f}")
            logger.info(f"  Lightning Detection Rate: {metrics.get('lightning_detection_rate', 0):.4f}")
            logger.info(f"  False Alarm Ratio: {metrics.get('false_alarm_ratio', 0):.4f}")
            
            if split in all_physics_analysis and 'physics_consistency' in all_physics_analysis[split]:
                pc = all_physics_analysis[split]['physics_consistency']['correlations']
                logger.info(f"  CAPE-Prediction Correlation: {pc['cape_prediction_correlation']:.4f}")
    
    logger.info(f"Detailed results saved to: {args.output_dir}")
    logger.info("Enhanced evaluation completed successfully!")
    
    return 0

def create_standard_evaluation_report(metrics: dict, model_info: dict, output_dir: str):
    """Create standard evaluation report (fallback when physics analysis fails)."""
    
    report = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "model_info": model_info,
        "metrics": metrics
    }
    
    with open(f"{output_dir}/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    # Simple markdown report
    markdown_content = f"""# Lightning Prediction Model - Evaluation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information
- Total Parameters: {model_info.get('total_parameters', 'N/A'):,}
- Model Size: {model_info.get('model_size_mb', 'N/A'):.1f} MB

## Performance Metrics
- **Accuracy**: {metrics.get('accuracy', 0):.4f}
- **F1 Score**: {metrics.get('f1_score', 0):.4f}
- **Precision**: {metrics.get('precision', 0):.4f}
- **Recall**: {metrics.get('recall', 0):.4f}
- **ROC AUC**: {metrics.get('roc_auc', 0):.4f}
"""
    
    with open(f"{output_dir}/evaluation_report.md", "w") as f:
        f.write(markdown_content)

def plot_standard_metrics(metrics: dict, output_dir: str):
    """Create standard metrics plots (fallback)."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Classification metrics
    classification_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    classification_values = [metrics.get(m, 0) for m in classification_metrics]
    
    axes[0, 0].bar(classification_metrics, classification_values, color='skyblue')
    axes[0, 0].set_title('Classification Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_ylim(0, 1)
    
    # Confusion Matrix
    tp = metrics.get('true_positives', 0)
    fp = metrics.get('false_positives', 0)
    tn = metrics.get('true_negatives', 0)
    fn = metrics.get('false_negatives', 0)
    
    conf_matrix = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Lightning', 'Lightning'],
                yticklabels=['No Lightning', 'Lightning'],
                ax=axes[0, 1])
    axes[0, 1].set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_overview.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)