"""
Evaluation metrics for lightning prediction models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import warnings

# DEBUG UTILITIES IMPORTS - NEW ADDITION
from ..utils.debug_utils import debug_print, is_debug_enabled

class LightningMetrics(nn.Module):
    """
    Comprehensive metrics for lightning prediction evaluation.
    Handles the unique challenges of lightning prediction including class imbalance.
    """
    
    def __init__(self,
                 threshold: float = 0.5,
                 spatial_tolerance: int = 1,
                 temporal_tolerance: int = 0,
                 compute_spatial_metrics: bool = True,
                 compute_probabilistic_metrics: bool = True):
        """
        Initialize lightning metrics.
        
        Args:
            threshold: Decision threshold for binary classification
            spatial_tolerance: Spatial tolerance in grid points for neighborhood verification
            temporal_tolerance: Temporal tolerance in time steps (future use)
            compute_spatial_metrics: Whether to compute spatial verification metrics
            compute_probabilistic_metrics: Whether to compute probabilistic metrics
        """
        super().__init__()
        
        self.threshold = threshold
        self.spatial_tolerance = spatial_tolerance
        self.temporal_tolerance = temporal_tolerance
        self.compute_spatial_metrics = compute_spatial_metrics
        self.compute_probabilistic_metrics = compute_probabilistic_metrics
        
        # Initialize metric storage
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.predictions_list = []
        self.targets_list = []
        self.probabilities_list = []
        
    def update(self, 
               predictions: torch.Tensor, 
               targets: torch.Tensor,
               probabilities: Optional[torch.Tensor] = None):
        """
        Update metrics with new batch.
        
        Args:
            predictions: Binary predictions (B, H, W) or (B, 1, H, W)
            targets: Ground truth targets (B, H, W)
            probabilities: Prediction probabilities (B, H, W) or (B, 1, H, W)
        """
        # DEBUG: Log original shapes - MODIFIED
        if is_debug_enabled("metrics"):
            debug_print(f"METRICS UPDATE:", "metrics")
            debug_print(f"predictions original shape: {predictions.shape}", "metrics")
            debug_print(f"targets original shape: {targets.shape}", "metrics")
            if probabilities is not None:
                debug_print(f"probabilities original shape: {probabilities.shape}", "metrics")
        
        # Handle different input shapes - ENSURE CONSISTENCY
        if predictions.dim() == 4 and predictions.shape[1] == 1:
            predictions = predictions.squeeze(1)
            if is_debug_enabled("metrics"):
                debug_print(f"predictions after squeeze: {predictions.shape}", "metrics")
        
        # FIX: Also squeeze targets if they have a channel dimension
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
            if is_debug_enabled("metrics"):
                debug_print(f"targets after squeeze: {targets.shape}", "metrics")
        
        if probabilities is not None and probabilities.dim() == 4 and probabilities.shape[1] == 1:
            probabilities = probabilities.squeeze(1)
            if is_debug_enabled("metrics"):
                debug_print(f"probabilities after squeeze: {probabilities.shape}", "metrics")
        
        # FIX: Ensure predictions and targets have exactly the same shape
        if predictions.shape != targets.shape:
            if is_debug_enabled("metrics"):
                debug_print(f"SHAPE MISMATCH DETECTED!", "metrics")
                debug_print(f"predictions shape: {predictions.shape}", "metrics")
                debug_print(f"targets shape: {targets.shape}", "metrics")
            
            # Try to fix common mismatches
            if predictions.dim() == 3 and targets.dim() == 2:
                # predictions: (B, H, W), targets: (H, W) - add batch dimension to targets
                targets = targets.unsqueeze(0)
                if is_debug_enabled("metrics"):
                    debug_print(f"Fixed targets shape: {targets.shape}", "metrics")
            elif predictions.dim() == 2 and targets.dim() == 3:
                # predictions: (H, W), targets: (B, H, W) - add batch dimension to predictions
                predictions = predictions.unsqueeze(0)
                if is_debug_enabled("metrics"):
                    debug_print(f"Fixed predictions shape: {predictions.shape}", "metrics")
                
            # After attempted fixes, check again
            if predictions.shape != targets.shape:
                # Last resort: interpolate to match
                if is_debug_enabled("metrics"):
                    debug_print(f"Attempting interpolation fix...", "metrics")
                    
                if predictions.numel() != targets.numel():
                    # Reshape to match the smaller tensor
                    min_h = min(predictions.shape[-2], targets.shape[-2])
                    min_w = min(predictions.shape[-1], targets.shape[-1])
                    
                    predictions = torch.nn.functional.interpolate(
                        predictions.unsqueeze(1) if predictions.dim() == 3 else predictions,
                        size=(min_h, min_w), mode='bilinear', align_corners=False
                    ).squeeze(1) if predictions.dim() == 4 else predictions
                    
                    targets = torch.nn.functional.interpolate(
                        targets.unsqueeze(1) if targets.dim() == 3 else targets,
                        size=(min_h, min_w), mode='nearest'
                    ).squeeze(1) if targets.dim() == 4 else targets
                    
                    if is_debug_enabled("metrics"):
                        debug_print(f"After interpolation - predictions: {predictions.shape}, targets: {targets.shape}", "metrics")
        
        # DEBUG: Log shapes after processing - MODIFIED
        if is_debug_enabled("metrics"):
            debug_print(f"predictions final shape: {predictions.shape}", "metrics")
            debug_print(f"targets final shape: {targets.shape}", "metrics")
        
        # Convert to numpy for sklearn metrics
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Final shape check
        if pred_np.shape != target_np.shape:
            if is_debug_enabled("metrics"):
                debug_print(f"CRITICAL ERROR: Still mismatched after fixes!", "metrics")
                debug_print(f"pred_np: {pred_np.shape}, target_np: {target_np.shape}", "metrics")
            return  # Skip this batch to prevent crash
        
        # DEBUG: Log numpy array shapes and flattened lengths - MODIFIED
        if is_debug_enabled("metrics"):
            debug_print(f"pred_np shape: {pred_np.shape}", "metrics")
            debug_print(f"target_np shape: {target_np.shape}", "metrics")
            debug_print(f"pred_np.flatten() length: {pred_np.flatten().shape[0]}", "metrics")
            debug_print(f"target_np.flatten() length: {target_np.flatten().shape[0]}", "metrics")
        
        # Store for batch computation
        self.predictions_list.append(pred_np)
        self.targets_list.append(target_np)
        
        # DEBUG: Log accumulated totals - MODIFIED
        if is_debug_enabled("metrics"):
            total_pred_samples = sum(p.flatten().shape[0] for p in self.predictions_list)
            total_target_samples = sum(t.flatten().shape[0] for t in self.targets_list)
            debug_print(f"Total accumulated pred samples: {total_pred_samples}", "metrics")
            debug_print(f"Total accumulated target samples: {total_target_samples}", "metrics")
        
        if probabilities is not None:
            # Apply same fixes to probabilities
            if probabilities.shape != targets.shape:
                probabilities = torch.nn.functional.interpolate(
                    probabilities.unsqueeze(1) if probabilities.dim() == 3 else probabilities,
                    size=targets.shape[-2:], mode='bilinear', align_corners=False
                ).squeeze(1) if probabilities.dim() == 4 else probabilities
                
            prob_np = probabilities.detach().cpu().numpy()
            if is_debug_enabled("metrics"):
                debug_print(f"prob_np shape: {prob_np.shape}", "metrics")
            self.probabilities_list.append(prob_np)
            
            if is_debug_enabled("metrics"):
                total_prob_samples = sum(p.flatten().shape[0] for p in self.probabilities_list)
                debug_print(f"Total accumulated prob samples: {total_prob_samples}", "metrics")
        
        if is_debug_enabled("metrics"):
            debug_print("DEBUG METRICS UPDATE END", "metrics")
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated predictions and targets.
        
        Returns:
            Dictionary of computed metrics
        """
        if not self.predictions_list:
            return {}
        
        # Concatenate all batches
        all_predictions = np.concatenate([p.flatten() for p in self.predictions_list])
        all_targets = np.concatenate([t.flatten() for t in self.targets_list])
        
        if self.probabilities_list:
            all_probabilities = np.concatenate([p.flatten() for p in self.probabilities_list])
        else:
            all_probabilities = None
        
        metrics = {}
        
        # Basic classification metrics
        metrics.update(self._compute_basic_metrics(all_predictions, all_targets))
        
        # Probabilistic metrics
        if self.compute_probabilistic_metrics and all_probabilities is not None:
            metrics.update(self._compute_probabilistic_metrics(all_probabilities, all_targets))
        
        # Spatial verification metrics
        if self.compute_spatial_metrics:
            metrics.update(self._compute_spatial_metrics())
        
        # Lightning-specific metrics
        metrics.update(self._compute_lightning_specific_metrics(all_predictions, all_targets))
        
        return metrics
    
    def _compute_basic_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute basic classification metrics."""
        
        # FIX: Ensure arrays have same length to prevent broadcast errors
        min_length = min(len(predictions), len(targets))
        if len(predictions) != len(targets):
            predictions = predictions[:min_length]
            targets = targets[:min_length]
        
        # Convert probabilities to binary if needed
        if predictions.max() <= 1.0 and predictions.min() >= 0.0:
            binary_preds = (predictions > self.threshold).astype(int)
        else:
            binary_preds = predictions.astype(int)
        
        binary_targets = targets.astype(int)
        
        # Handle edge cases
        if len(np.unique(binary_targets)) == 1:
            warnings.warn("Only one class present in targets")
            if binary_targets[0] == 0:  # All negative
                return {
                    'accuracy': 1.0 if np.all(binary_preds == 0) else (binary_preds == 0).mean(),
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0
                }
            else:  # All positive
                return {
                    'accuracy': 1.0 if np.all(binary_preds == 1) else (binary_preds == 1).mean(),
                    'precision': 1.0 if np.all(binary_preds == 1) else (binary_preds == 1).mean(),
                    'recall': 1.0,
                    'f1_score': 1.0 if np.all(binary_preds == 1) else 2 * (binary_preds == 1).mean() / (1 + (binary_preds == 1).mean())
                }
        
        return {
            'accuracy': accuracy_score(binary_targets, binary_preds),
            'precision': precision_score(binary_targets, binary_preds, zero_division=0),
            'recall': recall_score(binary_targets, binary_preds, zero_division=0),
            'f1_score': f1_score(binary_targets, binary_preds, zero_division=0)
        }
    
    def _compute_probabilistic_metrics(self, probabilities: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute probabilistic metrics."""
        
        binary_targets = targets.astype(int)
        
        # Handle edge cases
        if len(np.unique(binary_targets)) == 1:
            warnings.warn("Only one class present in targets for probabilistic metrics")
            return {
                'roc_auc': 0.5,
                'average_precision': binary_targets[0],
                'brier_score': np.mean((probabilities - binary_targets) ** 2)
            }
        
        try:
            roc_auc = roc_auc_score(binary_targets, probabilities)
        except ValueError:
            roc_auc = 0.5
        
        try:
            avg_precision = average_precision_score(binary_targets, probabilities)
        except ValueError:
            avg_precision = np.mean(binary_targets)
        
        # Brier Score
        brier_score = np.mean((probabilities - binary_targets) ** 2)
        
        # Reliability (calibration) - simplified
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        reliability = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = binary_targets[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                reliability += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return {
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'brier_score': brier_score,
            'reliability': reliability
        }
    
    def _compute_spatial_metrics(self) -> Dict[str, float]:
        """Compute spatial verification metrics."""
        # Placeholder for spatial metrics
        # Would need to compute neighborhood-based verification
        return {
            'spatial_accuracy': 0.0,
            'spatial_precision': 0.0,
            'spatial_recall': 0.0
        }
    
    def _create_neighborhood_mask(self, binary_array: np.ndarray, tolerance: int) -> np.ndarray:
        """Create neighborhood mask for spatial verification."""
        from scipy import ndimage
        
        # Create structuring element (disk/square)
        struct_size = 2 * tolerance + 1
        struct_element = np.ones((struct_size, struct_size))
        
        # Dilate the binary array
        neighborhood_mask = ndimage.binary_dilation(binary_array, structure=struct_element)
        
        return neighborhood_mask.astype(int)
    
    def _compute_lightning_specific_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute lightning-specific verification metrics."""
        
        # FIX: Ensure arrays have same length to prevent broadcast errors
        min_length = min(len(predictions), len(targets))
        if len(predictions) != len(targets):
            predictions = predictions[:min_length]
            targets = targets[:min_length]
        
        # Convert probabilities to binary if needed
        if predictions.max() <= 1.0 and predictions.min() >= 0.0:
            binary_preds = (predictions > self.threshold).astype(int)
        else:
            binary_preds = predictions.astype(int)
        
        binary_targets = targets.astype(int)
        
        metrics = {}
        
        # Lightning Detection Rate (same as recall)
        lightning_events = np.sum(binary_targets)
        detected_lightning = np.sum(binary_preds * binary_targets)
        detection_rate = detected_lightning / lightning_events if lightning_events > 0 else 0.0
        
        # False Alarm Ratio
        total_alarms = np.sum(binary_preds)
        false_alarms = np.sum(binary_preds * (1 - binary_targets))
        false_alarm_ratio = false_alarms / total_alarms if total_alarms > 0 else 0.0
        
        # Critical Success Index (Threat Score)
        hits = detected_lightning
        misses = lightning_events - detected_lightning
        false_alarms = false_alarms
        csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0.0
        
        # Frequency Bias
        frequency_bias = total_alarms / lightning_events if lightning_events > 0 else float('inf')
        
        # Heidke Skill Score
        po = np.mean(binary_preds == binary_targets)  # Observed accuracy
        n = len(binary_targets)
        n1 = np.sum(binary_targets)
        n0 = n - n1
        m1 = np.sum(binary_preds)
        m0 = n - m1
        pe = (n1 * m1 + n0 * m0) / (n * n)  # Expected accuracy by chance
        hss = (po - pe) / (1 - pe) if pe != 1 else 0.0
        
        metrics.update({
            'lightning_detection_rate': detection_rate,
            'false_alarm_ratio': false_alarm_ratio,
            'critical_success_index': csi,
            'frequency_bias': frequency_bias,
            'heidke_skill_score': hss
        })
        
        return metrics


class MetricTracker:
    """
    Tracks metrics across training epochs and provides aggregation.
    """
    
    def __init__(self, metrics_to_track: Optional[List[str]] = None):
        """
        Initialize metric tracker.
        
        Args:
            metrics_to_track: List of metric names to track. If None, tracks all metrics.
        """
        self.metrics_to_track = metrics_to_track
        self.epoch_metrics = []
        self.best_metrics = {}
        self.best_epoch = {}
    
    def update(self, epoch: int, metrics: Dict[str, float]):
        """
        Update tracker with new epoch metrics.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of computed metrics
        """
        # Filter metrics if specified
        if self.metrics_to_track:
            filtered_metrics = {k: v for k, v in metrics.items() if k in self.metrics_to_track}
        else:
            filtered_metrics = metrics.copy()
        
        # Add epoch info
        filtered_metrics['epoch'] = epoch
        self.epoch_metrics.append(filtered_metrics)
        
        # Update best metrics
        for metric_name, metric_value in filtered_metrics.items():
            if metric_name == 'epoch':
                continue
            
            # For loss-like metrics (lower is better)
            if 'loss' in metric_name.lower() or 'error' in metric_name.lower() or metric_name == 'brier_score':
                is_better = (metric_name not in self.best_metrics or 
                           metric_value < self.best_metrics[metric_name])
            else:  # For score-like metrics (higher is better)
                is_better = (metric_name not in self.best_metrics or 
                           metric_value > self.best_metrics[metric_name])
            
            if is_better:
                self.best_metrics[metric_name] = metric_value
                self.best_epoch[metric_name] = epoch
    
    def get_best_metrics(self) -> Dict[str, Tuple[float, int]]:
        """
        Get best metrics and their epochs.
        
        Returns:
            Dictionary mapping metric names to (best_value, best_epoch) tuples
        """
        return {name: (value, self.best_epoch[name]) 
                for name, value in self.best_metrics.items()}
    
    def get_metric_history(self, metric_name: str) -> List[float]:
        """Get history of a specific metric."""
        return [epoch_data.get(metric_name, None) for epoch_data in self.epoch_metrics]
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """Get metrics from the latest epoch."""
        return self.epoch_metrics[-1] if self.epoch_metrics else {}


def compute_class_weights(targets: torch.Tensor) -> torch.Tensor:
    """
    Compute class weights for imbalanced lightning data.
    
    Args:
        targets: Ground truth targets
        
    Returns:
        Class weights tensor
    """
    targets_flat = targets.flatten()
    unique_classes, counts = torch.unique(targets_flat, return_counts=True)
    
    # Compute inverse frequency weights
    total_samples = len(targets_flat)
    weights = total_samples / (len(unique_classes) * counts.float())
    
    # Create weight tensor
    class_weights = torch.zeros(2, device=targets.device)
    for cls, weight in zip(unique_classes, weights):
        class_weights[int(cls)] = weight
    
    return class_weights


def evaluate_model(model: nn.Module, 
                  dataloader: torch.utils.data.DataLoader,
                  metrics: LightningMetrics,
                  device: torch.device) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Lightning prediction model
        dataloader: Data loader
        metrics: Lightning metrics instance
        device: Device to run evaluation on
        
    Returns:
        Dictionary of computed metrics
    """
    model.eval()
    metrics.reset()
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            cape_data = batch['cape'].to(device)
            terrain_data = batch['terrain'].to(device)
            lightning_targets = batch['lightning'].to(device)
            
            # Forward pass
            outputs = model(cape_data, terrain_data)
            predictions = outputs['lightning_prediction']
            
            # Update metrics
            metrics.update(predictions, lightning_targets, predictions)
    
    # Compute final metrics
    computed_metrics = metrics.compute()
    
    return computed_metrics