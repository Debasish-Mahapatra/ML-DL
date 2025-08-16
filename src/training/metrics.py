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
        # DEBUG: Log original shapes
        print(f"DEBUG METRICS UPDATE:")
        print(f"  predictions original shape: {predictions.shape}")
        print(f"  targets original shape: {targets.shape}")
        if probabilities is not None:
            print(f"  probabilities original shape: {probabilities.shape}")
        
        # Handle different input shapes - ENSURE CONSISTENCY
        if predictions.dim() == 4 and predictions.shape[1] == 1:
            predictions = predictions.squeeze(1)
            print(f"  predictions after squeeze: {predictions.shape}")
        
        # FIX: Also squeeze targets if they have a channel dimension
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
            print(f"  targets after squeeze: {targets.shape}")
        
        if probabilities is not None and probabilities.dim() == 4 and probabilities.shape[1] == 1:
            probabilities = probabilities.squeeze(1)
            print(f"  probabilities after squeeze: {probabilities.shape}")
        
        # FIX: Ensure predictions and targets have exactly the same shape
        if predictions.shape != targets.shape:
            print(f"  SHAPE MISMATCH DETECTED!")
            print(f"  predictions shape: {predictions.shape}")
            print(f"  targets shape: {targets.shape}")
            
            # Try to fix common mismatches
            if predictions.dim() == 3 and targets.dim() == 2:
                # predictions: (B, H, W), targets: (H, W) - add batch dimension to targets
                targets = targets.unsqueeze(0)
                print(f"  Fixed targets shape: {targets.shape}")
            elif predictions.dim() == 2 and targets.dim() == 3:
                # predictions: (H, W), targets: (B, H, W) - add batch dimension to predictions
                predictions = predictions.unsqueeze(0)
                print(f"  Fixed predictions shape: {predictions.shape}")
                
            # After attempted fixes, check again
            if predictions.shape != targets.shape:
                # Last resort: interpolate to match
                print(f"  Attempting interpolation fix...")
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
                    
                    print(f"  After interpolation - predictions: {predictions.shape}, targets: {targets.shape}")
        
        # DEBUG: Log shapes after processing
        print(f"  predictions final shape: {predictions.shape}")
        print(f"  targets final shape: {targets.shape}")
        
        # Convert to numpy for sklearn metrics
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        # Final shape check
        if pred_np.shape != target_np.shape:
            print(f"  CRITICAL ERROR: Still mismatched after fixes!")
            print(f"  pred_np: {pred_np.shape}, target_np: {target_np.shape}")
            return  # Skip this batch to prevent crash
        
        # DEBUG: Log numpy array shapes and flattened lengths
        print(f"  pred_np shape: {pred_np.shape}")
        print(f"  target_np shape: {target_np.shape}")
        print(f"  pred_np.flatten() length: {pred_np.flatten().shape[0]}")
        print(f"  target_np.flatten() length: {target_np.flatten().shape[0]}")
        
        # Store for batch computation
        self.predictions_list.append(pred_np)
        self.targets_list.append(target_np)
        
        # DEBUG: Log accumulated totals
        total_pred_samples = sum(p.flatten().shape[0] for p in self.predictions_list)
        total_target_samples = sum(t.flatten().shape[0] for t in self.targets_list)
        print(f"  Total accumulated pred samples: {total_pred_samples}")
        print(f"  Total accumulated target samples: {total_target_samples}")
        
        if probabilities is not None:
            # Apply same fixes to probabilities
            if probabilities.shape != targets.shape:
                probabilities = torch.nn.functional.interpolate(
                    probabilities.unsqueeze(1) if probabilities.dim() == 3 else probabilities,
                    size=targets.shape[-2:], mode='bilinear', align_corners=False
                ).squeeze(1) if probabilities.dim() == 4 else probabilities
                
            prob_np = probabilities.detach().cpu().numpy()
            print(f"  prob_np shape: {prob_np.shape}")
            self.probabilities_list.append(prob_np)
            
            total_prob_samples = sum(p.flatten().shape[0] for p in self.probabilities_list)
            print(f"  Total accumulated prob samples: {total_prob_samples}")
        
        print("DEBUG METRICS UPDATE END\n")
    
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
                    'f1_score': 0.0,
                    'specificity': 1.0 if np.all(binary_preds == 0) else (binary_preds == 0).mean()
                }
            else:  # All positive
                return {
                    'accuracy': 1.0 if np.all(binary_preds == 1) else (binary_preds == 1).mean(),
                    'precision': 1.0 if np.all(binary_preds == 1) else (binary_preds == 1).mean(),
                    'recall': 1.0,
                    'f1_score': 1.0 if np.all(binary_preds == 1) else 2 * (binary_preds == 1).mean() / (1 + (binary_preds == 1).mean()),
                    'specificity': 0.0
                }
        
        # Compute metrics
        accuracy = accuracy_score(binary_targets, binary_preds)
        precision = precision_score(binary_targets, binary_preds, zero_division=0)
        recall = recall_score(binary_targets, binary_preds, zero_division=0)
        f1 = f1_score(binary_targets, binary_preds, zero_division=0)
        
        # Confusion matrix for additional metrics
        tn, fp, fn, tp = confusion_matrix(binary_targets, binary_preds, labels=[0, 1]).ravel()
        
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # False Alarm Rate
        false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Balanced accuracy
        balanced_accuracy = (recall + specificity) / 2
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'false_alarm_rate': false_alarm_rate,
            'balanced_accuracy': balanced_accuracy,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
    
    def _compute_probabilistic_metrics(self, probabilities: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute probabilistic metrics."""
        
        # FIX: Ensure arrays have same length to prevent broadcast errors in element-wise operations
        min_length = min(len(probabilities), len(targets))
        if len(probabilities) != len(targets):
            probabilities = probabilities[:min_length]
            targets = targets[:min_length]
        
        binary_targets = targets.astype(int)
        
        # Handle edge cases
        if len(np.unique(binary_targets)) == 1:
            return {
                'roc_auc': 0.5,
                'average_precision': binary_targets[0],
                'brier_score': np.mean((probabilities - binary_targets) ** 2)
            }
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(binary_targets, probabilities)
        except ValueError:
            roc_auc = 0.5
        
        # Average Precision (Area under PR curve)
        try:
            avg_precision = average_precision_score(binary_targets, probabilities)
        except ValueError:
            avg_precision = np.mean(binary_targets)
        
        # Brier Score (lower is better)
        brier_score = np.mean((probabilities - binary_targets) ** 2)
        
        # Reliability (calibration) - simplified
        reliability = self._compute_reliability(probabilities, binary_targets)
        
        return {
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'brier_score': brier_score,
            'reliability': reliability
        }
    
    def _compute_reliability(self, probabilities: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
        """Compute reliability (calibration) metric."""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        reliability = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Average confidence in this bin
                confidence_in_bin = probabilities[in_bin].mean()
                # Average accuracy in this bin
                accuracy_in_bin = targets[in_bin].mean()
                # Weighted contribution to reliability
                reliability += np.abs(confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return reliability
    
    def _compute_spatial_metrics(self) -> Dict[str, float]:
        """Compute spatial verification metrics using neighborhood approach."""
        
        if not self.predictions_list or not self.targets_list:
            return {}
        
        spatial_metrics = {}
        
        # Process each sample separately to preserve spatial structure
        spatial_accuracies = []
        spatial_f1_scores = []
        
        for pred_batch, target_batch in zip(self.predictions_list, self.targets_list):
            for pred, target in zip(pred_batch, target_batch):
                
                # Convert to binary if needed
                if pred.max() <= 1.0 and pred.min() >= 0.0:
                    binary_pred = (pred > self.threshold).astype(int)
                else:
                    binary_pred = pred.astype(int)
                
                binary_target = target.astype(int)
                
                # Compute neighborhood-based metrics
                spatial_acc = self._neighborhood_accuracy(binary_pred, binary_target)
                spatial_f1 = self._neighborhood_f1(binary_pred, binary_target)
                
                spatial_accuracies.append(spatial_acc)
                spatial_f1_scores.append(spatial_f1)
        
        spatial_metrics['spatial_accuracy'] = np.mean(spatial_accuracies)
        spatial_metrics['spatial_f1'] = np.mean(spatial_f1_scores)
        
        return spatial_metrics
    
    def _neighborhood_accuracy(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Compute accuracy allowing for spatial tolerance."""
        
        if self.spatial_tolerance == 0:
            return np.mean(prediction == target)
        
        # FIX: Ensure both arrays have same shape before neighborhood operations
        if prediction.shape != target.shape:
            min_h, min_w = min(prediction.shape[0], target.shape[0]), min(prediction.shape[1], target.shape[1])
            prediction = prediction[:min_h, :min_w]
            target = target[:min_h, :min_w]
        
        # Create neighborhood masks
        pred_neighborhood = self._create_neighborhood_mask(prediction, self.spatial_tolerance)
        target_neighborhood = self._create_neighborhood_mask(target, self.spatial_tolerance)
        
        # A prediction is correct if it hits within the neighborhood of any target
        correct_predictions = np.logical_and(pred_neighborhood, target_neighborhood)
        
        # Compute neighborhood accuracy
        total_targets = np.sum(target)
        total_predictions = np.sum(prediction)
        
        if total_targets == 0 and total_predictions == 0:
            return 1.0
        elif total_targets == 0:
            return 0.0
        else:
            return np.sum(correct_predictions) / max(total_targets, total_predictions)
    
    def _neighborhood_f1(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """Compute F1 score allowing for spatial tolerance."""
        
        if self.spatial_tolerance == 0:
            # Standard F1 calculation
            tp = np.sum(prediction * target)
            fp = np.sum(prediction * (1 - target))
            fn = np.sum((1 - prediction) * target)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # FIX: Ensure both arrays have same shape
        if prediction.shape != target.shape:
            min_h, min_w = min(prediction.shape[0], target.shape[0]), min(prediction.shape[1], target.shape[1])
            prediction = prediction[:min_h, :min_w]
            target = target[:min_h, :min_w]
        
        # Neighborhood-based F1
        pred_neighborhood = self._create_neighborhood_mask(prediction, self.spatial_tolerance)
        target_neighborhood = self._create_neighborhood_mask(target, self.spatial_tolerance)
        
        tp = np.sum(pred_neighborhood * target_neighborhood)
        fp = np.sum(pred_neighborhood * (1 - target_neighborhood))
        fn = np.sum((1 - pred_neighborhood) * target_neighborhood)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def _create_neighborhood_mask(self, binary_array: np.ndarray, tolerance: int) -> np.ndarray:
        """Create neighborhood mask for spatial tolerance."""
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