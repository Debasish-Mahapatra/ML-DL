"""
Enhanced visualization utilities for lightning prediction with physics analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import Dict, List, Optional, Tuple, Union
import torch
from pathlib import Path

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Custom colormaps for physics visualization
def create_physics_colormap():
    """Create custom colormap for physics-informed visualizations."""
    
    # CAPE regime colors
    cape_colors = ['#2C3E50', '#3498DB', '#F39C12', '#E74C3C', '#8E44AD']  # Dark blue to purple
    cape_cmap = LinearSegmentedColormap.from_list('cape_physics', cape_colors, N=256)
    
    # Lightning probability colors
    lightning_colors = ['#FFFFFF', '#FFF3CD', '#FFE066', '#FF9999', '#FF4444', '#CC0000']
    lightning_cmap = LinearSegmentedColormap.from_list('lightning_prob', lightning_colors, N=256)
    
    # Physics alignment colors
    physics_colors = ['#FF4444', '#FFB366', '#FFFF66', '#B3FF66', '#66FF66']  # Red to green
    physics_cmap = LinearSegmentedColormap.from_list('physics_alignment', physics_colors, N=256)
    
    return {
        'cape': cape_cmap,
        'lightning': lightning_cmap,
        'physics': physics_cmap
    }

PHYSICS_CMAPS = create_physics_colormap()

def plot_cape_physics_analysis(cape_data: np.ndarray, 
                              predictions: np.ndarray,
                              targets: Optional[np.ndarray] = None,
                              thresholds: Optional[Dict] = None,
                              output_file: str = None,
                              title: str = "CAPE Physics Analysis") -> plt.Figure:
    """
    Create comprehensive CAPE physics analysis visualization.
    
    Args:
        cape_data: CAPE values (2D array)
        predictions: Model predictions (2D array)
        targets: Ground truth lightning (2D array, optional)
        thresholds: CAPE physics thresholds
        output_file: Path to save figure
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    
    # Default thresholds
    if thresholds is None:
        thresholds = {
            'no_lightning': 1000.0,
            'moderate': 2500.0,
            'high': 4000.0,
            'saturation': 5000.0
        }
    
    # Create figure with subplots
    if targets is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
    
    # Plot 1: CAPE data with physics regimes
    cape_plot = axes[0].imshow(cape_data, cmap=PHYSICS_CMAPS['cape'], vmin=0, vmax=6000)
    axes[0].set_title('CAPE with Physics Regimes', fontsize=14, fontweight='bold')
    
    # Add regime boundaries
    cape_masked = np.ma.masked_where(cape_data < thresholds['no_lightning'], cape_data)
    axes[0].contour(cape_masked, levels=[thresholds['no_lightning']], colors='white', linewidths=2, alpha=0.8)
    axes[0].contour(cape_data, levels=[thresholds['moderate']], colors='yellow', linewidths=2, alpha=0.8)
    axes[0].contour(cape_data, levels=[thresholds['high']], colors='orange', linewidths=2, alpha=0.8)
    
    # Add colorbar
    cbar1 = plt.colorbar(cape_plot, ax=axes[0], shrink=0.8)
    cbar1.set_label('CAPE (J/kg)', fontsize=12)
    
    # Add regime labels
    axes[0].text(0.02, 0.98, 'No Lightning\n(<1000)', transform=axes[0].transAxes, 
                va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10, fontweight='bold')
    axes[0].text(0.02, 0.02, f'Very High\n(>{thresholds["high"]:.0f})', transform=axes[0].transAxes, 
                va='bottom', ha='left', bbox=dict(boxstyle='round', facecolor='purple', alpha=0.8, color='white'),
                fontsize=10, fontweight='bold', color='white')
    
    # Plot 2: Lightning predictions
    pred_plot = axes[1].imshow(predictions, cmap=PHYSICS_CMAPS['lightning'], vmin=0, vmax=1)
    axes[1].set_title('Lightning Predictions', fontsize=14, fontweight='bold')
    cbar2 = plt.colorbar(pred_plot, ax=axes[1], shrink=0.8)
    cbar2.set_label('Lightning Probability', fontsize=12)
    
    # Plot 3: CAPE-Prediction overlay
    # Create physics-informed visualization
    cape_norm = np.clip(cape_data / 5000.0, 0, 1)
    physics_score = np.zeros_like(cape_data)
    
    # Calculate physics alignment score
    no_lightning_mask = cape_data < thresholds['no_lightning']
    moderate_mask = (cape_data >= thresholds['no_lightning']) & (cape_data < thresholds['moderate'])
    high_mask = (cape_data >= thresholds['moderate']) & (cape_data < thresholds['high'])
    very_high_mask = cape_data >= thresholds['high']
    
    # Expected vs actual rates
    physics_score[no_lightning_mask] = 1.0 - predictions[no_lightning_mask]  # Should be low
    physics_score[moderate_mask] = 1.0 - np.abs(predictions[moderate_mask] - 0.25)  # Should be ~0.25
    physics_score[high_mask] = 1.0 - np.abs(predictions[high_mask] - 0.5)  # Should be ~0.5
    physics_score[very_high_mask] = predictions[very_high_mask]  # Should be high
    
    physics_plot = axes[2].imshow(physics_score, cmap=PHYSICS_CMAPS['physics'], vmin=0, vmax=1)
    axes[2].set_title('Physics Alignment Score', fontsize=14, fontweight='bold')
    cbar3 = plt.colorbar(physics_plot, ax=axes[2], shrink=0.8)
    cbar3.set_label('Physics Alignment', fontsize=12)
    
    # Plot 4: CAPE vs Prediction scatter
    # Flatten arrays for scatter plot
    cape_flat = cape_data.flatten()
    pred_flat = predictions.flatten()
    
    # Sample for visibility if too many points
    if len(cape_flat) > 10000:
        indices = np.random.choice(len(cape_flat), 10000, replace=False)
        cape_flat = cape_flat[indices]
        pred_flat = pred_flat[indices]
    
    scatter = axes[3].scatter(cape_flat, pred_flat, c=cape_flat, cmap=PHYSICS_CMAPS['cape'], 
                             alpha=0.6, s=1)
    axes[3].set_xlabel('CAPE (J/kg)', fontsize=12)
    axes[3].set_ylabel('Lightning Probability', fontsize=12)
    axes[3].set_title('CAPE vs Prediction Correlation', fontsize=14, fontweight='bold')
    
    # Add correlation coefficient
    correlation = np.corrcoef(cape_flat, pred_flat)[0, 1]
    axes[3].text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=axes[3].transAxes,
                va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=12, fontweight='bold')
    
    # Add physics expectation line
    cape_range = np.linspace(0, cape_flat.max(), 100)
    expected_prob = np.zeros_like(cape_range)
    expected_prob[cape_range < thresholds['no_lightning']] = 0.05
    expected_prob[(cape_range >= thresholds['no_lightning']) & (cape_range < thresholds['moderate'])] = 0.25
    expected_prob[(cape_range >= thresholds['moderate']) & (cape_range < thresholds['high'])] = 0.5
    expected_prob[cape_range >= thresholds['high']] = 0.75
    
    axes[3].plot(cape_range, expected_prob, 'r--', linewidth=3, alpha=0.8, label='Physics Expectation')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # Plot 5: Ground truth comparison (if available)
    if targets is not None:
        target_plot = axes[4].imshow(targets, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[4].set_title('Ground Truth Lightning', fontsize=14, fontweight='bold')
        cbar5 = plt.colorbar(target_plot, ax=axes[4], shrink=0.8)
        cbar5.set_label('Lightning Occurrence', fontsize=12)
        
        # Plot 6: Prediction vs Truth scatter
        target_flat = targets.flatten()
        if len(target_flat) > 10000:
            target_flat = target_flat[indices]
        
        axes[5].scatter(target_flat, pred_flat, c=cape_flat, cmap=PHYSICS_CMAPS['cape'],
                       alpha=0.6, s=1)
        axes[5].set_xlabel('Ground Truth', fontsize=12)
        axes[5].set_ylabel('Predictions', fontsize=12)
        axes[5].set_title('Prediction vs Truth', fontsize=14, fontweight='bold')
        axes[5].plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')
        axes[5].legend()
        axes[5].grid(True, alpha=0.3)
    
    # Remove axes ticks for image plots
    for i in range(min(4, len(axes))):
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved physics analysis plot to {output_file}")
    
    return fig

def plot_regime_performance_analysis(regime_analysis: Dict,
                                   output_file: str = None,
                                   title: str = "CAPE Regime Performance Analysis") -> plt.Figure:
    """
    Create detailed regime performance analysis plots.
    
    Args:
        regime_analysis: Dictionary with regime analysis data
        output_file: Path to save figure
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    regimes = list(regime_analysis.keys())
    regime_names = [r.replace('_', ' ').title() for r in regimes]
    
    # Colors for each regime
    colors = ['#E74C3C', '#F39C12', '#2ECC71', '#9B59B6'][:len(regimes)]
    
    # Plot 1: Sample distribution
    counts = [regime_analysis[r]['pixel_count'] for r in regimes]
    bars1 = axes[0].bar(regime_names, counts, color=colors)
    axes[0].set_title('Sample Distribution by Regime', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Pixels')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add percentage labels
    total_samples = sum(counts)
    for bar, count in zip(bars1, counts):
        percentage = count / total_samples * 100
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Lightning rates comparison
    expected_rates = []
    predicted_rates = []
    
    for regime in regimes:
        data = regime_analysis[regime]
        expected_rates.append(data['predictions']['lightning_rate_expected'])
        predicted_rates.append(data['predictions']['lightning_rate_predicted'])
    
    x = np.arange(len(regimes))
    width = 0.35
    
    bars2a = axes[1].bar(x - width/2, expected_rates, width, label='Expected', color='red', alpha=0.7)
    bars2b = axes[1].bar(x + width/2, predicted_rates, width, label='Predicted', color='blue', alpha=0.7)
    
    axes[1].set_xlabel('CAPE Regime')
    axes[1].set_ylabel('Lightning Rate')
    axes[1].set_title('Lightning Rates: Expected vs Predicted', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([r.replace(' ', '\n') for r in regime_names])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars2a, bars2b]:
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Physics alignment scores
    alignment_scores = [regime_analysis[r]['predictions']['physics_alignment'] for r in regimes]
    bars3 = axes[2].bar(regime_names, alignment_scores, color=colors)
    axes[2].set_title('Physics Alignment by Regime', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Alignment Score')
    axes[2].set_ylim(0, 1)
    axes[2].tick_params(axis='x', rotation=45)
    
    # Color bars based on performance
    for bar, score in zip(bars3, alignment_scores):
        if score > 0.8:
            bar.set_color('green')
        elif score > 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
        
        # Add score labels
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: CAPE ranges by regime
    cape_means = [regime_analysis[r]['cape_range']['mean'] for r in regimes]
    cape_mins = [regime_analysis[r]['cape_range']['min'] for r in regimes]
    cape_maxs = [regime_analysis[r]['cape_range']['max'] for r in regimes]
    
    # Create error bars showing range
    cape_errors = [[m - min_val for m, min_val in zip(cape_means, cape_mins)],
                   [max_val - m for m, max_val in zip(cape_means, cape_maxs)]]
    
    bars4 = axes[3].bar(regime_names, cape_means, yerr=cape_errors, color=colors,
                       alpha=0.7, capsize=5, error_kw={'linewidth': 2})
    axes[3].set_title('CAPE Ranges by Regime', fontsize=14, fontweight='bold')
    axes[3].set_ylabel('CAPE (J/kg)')
    axes[3].tick_params(axis='x', rotation=45)
    
    # Plot 5: Prediction probability distributions
    mean_probs = [regime_analysis[r]['predictions']['mean_probability'] for r in regimes]
    bars5 = axes[4].bar(regime_names, mean_probs, color=colors)
    axes[4].set_title('Mean Prediction Probability by Regime', fontsize=14, fontweight='bold')
    axes[4].set_ylabel('Mean Probability')
    axes[4].set_ylim(0, 1)
    axes[4].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, prob in zip(bars5, mean_probs):
        axes[4].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 6: Summary metrics heatmap
    metrics_data = []
    metric_names = ['Lightning Rate', 'Physics Alignment', 'Mean Probability']
    
    for regime in regimes:
        data = regime_analysis[regime]
        row = [
            data['predictions']['lightning_rate_predicted'],
            data['predictions']['physics_alignment'],
            data['predictions']['mean_probability']
        ]
        metrics_data.append(row)
    
    metrics_array = np.array(metrics_data)
    
    im = axes[5].imshow(metrics_array.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[5].set_title('Regime Performance Heatmap', fontsize=14, fontweight='bold')
    axes[5].set_xticks(range(len(regimes)))
    axes[5].set_xticklabels([r.replace(' ', '\n') for r in regime_names], rotation=45)
    axes[5].set_yticks(range(len(metric_names)))
    axes[5].set_yticklabels(metric_names)
    
    # Add text annotations
    for i in range(len(regimes)):
        for j in range(len(metric_names)):
            text = axes[5].text(i, j, f'{metrics_array[i, j]:.3f}',
                              ha='center', va='center', fontweight='bold',
                              color='white' if metrics_array[i, j] < 0.5 else 'black')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[5], shrink=0.8)
    cbar.set_label('Score', fontsize=12)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved regime analysis plot to {output_file}")
    
    return fig

def plot_training_physics_metrics(training_logs: Dict,
                                 output_file: str = None,
                                 title: str = "Training Physics Metrics") -> plt.Figure:
    """
    Plot physics-related metrics during training.
    
    Args:
        training_logs: Dictionary with training metrics over time
        output_file: Path to save figure
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    epochs = training_logs.get('epoch', [])
    
    # Plot 1: Loss components
    main_loss = training_logs.get('train_loss', [])
    physics_loss = training_logs.get('train_physics_loss', [])
    charge_loss = training_logs.get('train_charge_separation_loss', [])
    
    axes[0].plot(epochs, main_loss, label='Main Loss', linewidth=2, color='blue')
    axes[0].plot(epochs, physics_loss, label='Physics Loss', linewidth=2, color='red')
    axes[0].plot(epochs, charge_loss, label='Charge Separation', linewidth=2, color='orange')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Components', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Plot 2: F1 Score progression
    train_f1 = training_logs.get('train_f1_score', [])
    val_f1 = training_logs.get('val_f1_score', [])
    
    axes[1].plot(epochs, train_f1, label='Train F1', linewidth=2, color='green')
    axes[1].plot(epochs, val_f1, label='Validation F1', linewidth=2, color='purple')
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Model Performance Over Time', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Physics weights (if adaptive)
    if 'charge_separation_weight' in training_logs:
        charge_weight = training_logs['charge_separation_weight']
        physics_weight = training_logs.get('physics_weight', [])
        
        axes[2].plot(epochs, charge_weight, label='Charge Separation Weight', linewidth=2, color='red')
        if physics_weight:
            axes[2].plot(epochs, physics_weight, label='Overall Physics Weight', linewidth=2, color='orange')
        
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Weight')
        axes[2].set_title('Adaptive Physics Weights', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Learning rate
    learning_rate = training_logs.get('lr', [])
    if learning_rate:
        axes[3].plot(epochs, learning_rate, linewidth=2, color='black')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Learning Rate')
        axes[3].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        axes[3].set_yscale('log')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved training metrics plot to {output_file}")
    
    return fig

def create_prediction_comparison_plot(cape_data: np.ndarray,
                                    predictions: np.ndarray,
                                    targets: np.ndarray,
                                    output_file: str = None,
                                    title: str = "Prediction Comparison") -> plt.Figure:
    """
    Create side-by-side comparison of predictions vs targets.
    
    Args:
        cape_data: CAPE input data
        predictions: Model predictions
        targets: Ground truth
        output_file: Path to save figure
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: CAPE input
    cape_plot = axes[0, 0].imshow(cape_data, cmap=PHYSICS_CMAPS['cape'], vmin=0, vmax=5000)
    axes[0, 0].set_title('CAPE Input (J/kg)', fontsize=14, fontweight='bold')
    plt.colorbar(cape_plot, ax=axes[0, 0], shrink=0.8)
    
    # Plot 2: Predictions
    pred_plot = axes[0, 1].imshow(predictions, cmap=PHYSICS_CMAPS['lightning'], vmin=0, vmax=1)
    axes[0, 1].set_title('Model Predictions', fontsize=14, fontweight='bold')
    plt.colorbar(pred_plot, ax=axes[0, 1], shrink=0.8)
    
    # Plot 3: Ground truth
    target_plot = axes[1, 0].imshow(targets, cmap='RdYlBu_r', vmin=0, vmax=1)
    axes[1, 0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    plt.colorbar(target_plot, ax=axes[1, 0], shrink=0.8)
    
    # Plot 4: Difference (error map)
    difference = predictions - targets
    diff_plot = axes[1, 1].imshow(difference, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 1].set_title('Prediction Error (Pred - Truth)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(diff_plot, ax=axes[1, 1], shrink=0.8)
    cbar.set_label('Error')
    
    # Remove ticks
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved prediction comparison to {output_file}")
    
    return fig

# Convenience functions for backward compatibility
def create_prediction_plots(predictions: np.ndarray, 
                          targets: np.ndarray,
                          cape_data: Optional[np.ndarray] = None,
                          output_dir: str = None) -> List[plt.Figure]:
    """Create standard prediction plots with optional physics analysis."""
    
    figures = []
    
    # Basic comparison plot
    fig1 = create_prediction_comparison_plot(
        cape_data if cape_data is not None else np.zeros_like(predictions),
        predictions, targets,
        output_file=f"{output_dir}/prediction_comparison.png" if output_dir else None
    )
    figures.append(fig1)
    
    # Physics analysis if CAPE data available
    if cape_data is not None:
        fig2 = plot_cape_physics_analysis(
            cape_data, predictions, targets,
            output_file=f"{output_dir}/cape_physics_analysis.png" if output_dir else None
        )
        figures.append(fig2)
    
    return figures

def create_metric_plots(metrics: Dict, output_dir: str = None) -> List[plt.Figure]:
    """Create standard metric plots."""
    
    figures = []
    
    # Basic metrics plot (placeholder - implement based on metrics structure)
    fig = plt.figure(figsize=(12, 8))
    # Implementation would depend on metrics dictionary structure
    plt.title("Model Performance Metrics")
    
    if output_dir:
        plt.savefig(f"{output_dir}/metrics_summary.png", dpi=300, bbox_inches='tight')
    
    figures.append(fig)
    return figures