"""
Enhanced debug utilities with physics-specific debugging for lightning prediction.
"""

import logging
import os
import json
import torch
import numpy as np
from typing import Any, Optional, Dict, List
from omegaconf import DictConfig
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class DebugManager:
    """Centralized debug management with configurable switches and physics debugging."""
    
    def __init__(self, config: Optional[DictConfig] = None):
        """Initialize debug manager with configuration."""
        self.config = config
        self._debug_enabled = False
        self._verbose_logging = False
        self._memory_tracking = False
        self._metrics_debug = False
        self._batch_info = False
        self._save_debug_outputs = False
        self._physics_debug = False  # NEW: Physics-specific debugging
        self._cape_analysis = False  # NEW: CAPE data analysis
        
        # Load debug settings from config
        if config and hasattr(config, 'training') and hasattr(config.training, 'debug'):
            debug_config = config.training.debug
            self._debug_enabled = getattr(debug_config, 'enabled', False)
            self._verbose_logging = getattr(debug_config, 'verbose_logging', False)
            self._memory_tracking = getattr(debug_config, 'memory_tracking', False)
            self._metrics_debug = getattr(debug_config, 'metrics_debug', False)
            self._batch_info = getattr(debug_config, 'batch_info', False)
            self._save_debug_outputs = getattr(debug_config, 'save_debug_outputs', False)
            self._physics_debug = getattr(debug_config, 'physics_debug', False)
            self._cape_analysis = getattr(debug_config, 'cape_analysis', False)
        
        # Environment variable override
        if os.getenv('LIGHTNING_DEBUG', '').lower() == 'true':
            self._debug_enabled = True
            self._verbose_logging = True
            self._memory_tracking = True
            self._metrics_debug = True
            self._batch_info = True
            self._physics_debug = True
            self._cape_analysis = True
            logger.info("Full debug mode enabled via environment variable")
        
        # Physics-specific environment variables
        if os.getenv('PHYSICS_DEBUG', '').lower() == 'true':
            self._physics_debug = True
            logger.info("Physics debug mode enabled via environment variable")
        
        if os.getenv('CAPE_ANALYSIS', '').lower() == 'true':
            self._cape_analysis = True
            logger.info("CAPE analysis enabled via environment variable")
    
    @property
    def enabled(self) -> bool:
        """Check if any debug mode is enabled."""
        return self._debug_enabled
    
    @property
    def verbose_logging(self) -> bool:
        """Check if verbose logging is enabled."""
        return self._debug_enabled and self._verbose_logging
    
    @property
    def memory_tracking(self) -> bool:
        """Check if memory tracking is enabled."""
        return self._debug_enabled and self._memory_tracking
    
    @property
    def metrics_debug(self) -> bool:
        """Check if metrics debug is enabled."""
        return self._debug_enabled and self._metrics_debug
    
    @property
    def batch_info(self) -> bool:
        """Check if batch info debug is enabled."""
        return self._debug_enabled and self._batch_info
    
    @property
    def save_debug_outputs(self) -> bool:
        """Check if debug outputs should be saved."""
        return self._debug_enabled and self._save_debug_outputs
    
    @property
    def physics_debug(self) -> bool:
        """Check if physics debug is enabled."""
        return self._debug_enabled and self._physics_debug
    
    @property
    def cape_analysis(self) -> bool:
        """Check if CAPE analysis is enabled."""
        return self._debug_enabled and self._cape_analysis
    
    def debug_print(self, message: str, category: str = "general"):
        """Conditional debug print based on category."""
        if not self._debug_enabled:
            return
            
        category_enabled = {
            "general": self._debug_enabled,
            "memory": self.memory_tracking,
            "metrics": self.metrics_debug,
            "batch": self.batch_info,
            "verbose": self.verbose_logging,
            "physics": self.physics_debug,
            "cape": self.cape_analysis
        }
        
        if category_enabled.get(category, False):
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [DEBUG-{category.upper()}] {message}")
    
    def conditional_trace_memory(self, checkpoint_name: str):
        """Conditional memory tracing."""
        if self.memory_tracking:
            try:
                from .memory_tracker import trace_memory_line
                trace_memory_line()  # Only call if memory tracking enabled
            except ImportError:
                # Memory tracker not available
                pass

class PhysicsDebugger:
    """Specialized debugging for physics constraints and CAPE analysis."""
    
    def __init__(self, debug_manager: DebugManager, experiment_name: str = "physics_debug"):
        self.debug_manager = debug_manager
        self.experiment_name = experiment_name
        self.debug_dir = Path(f"experiments/{experiment_name}/physics_debug")
        
        # Create debug directory if needed
        if self.debug_manager.physics_debug or self.debug_manager.save_debug_outputs:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_cape_batch(self, cape_data: torch.Tensor, batch_idx: int = 0) -> Dict:
        """Analyze CAPE data in a batch for physics debugging."""
        
        if not self.debug_manager.cape_analysis:
            return {}
        
        # Convert to numpy for analysis
        if isinstance(cape_data, torch.Tensor):
            cape_np = cape_data.detach().cpu().numpy()
        else:
            cape_np = cape_data
        
        # Flatten for statistics
        cape_flat = cape_np.flatten()
        cape_flat = cape_flat[~np.isnan(cape_flat)]  # Remove NaN values
        
        # Physics thresholds
        thresholds = {
            'no_lightning': 1000.0,
            'moderate': 2500.0,
            'high': 4000.0,
            'saturation': 5000.0
        }
        
        # Calculate regime distribution
        regime_counts = {
            'no_lightning': np.sum(cape_flat < thresholds['no_lightning']),
            'moderate': np.sum((cape_flat >= thresholds['no_lightning']) & 
                              (cape_flat < thresholds['moderate'])),
            'high': np.sum((cape_flat >= thresholds['moderate']) & 
                          (cape_flat < thresholds['high'])),
            'very_high': np.sum(cape_flat >= thresholds['high'])
        }
        
        total_pixels = len(cape_flat)
        regime_percentages = {k: (v / total_pixels * 100) if total_pixels > 0 else 0 
                             for k, v in regime_counts.items()}
        
        analysis = {
            'batch_idx': batch_idx,
            'cape_stats': {
                'min': float(cape_flat.min()) if len(cape_flat) > 0 else 0,
                'max': float(cape_flat.max()) if len(cape_flat) > 0 else 0,
                'mean': float(cape_flat.mean()) if len(cape_flat) > 0 else 0,
                'std': float(cape_flat.std()) if len(cape_flat) > 0 else 0,
                'total_pixels': total_pixels
            },
            'regime_distribution': regime_counts,
            'regime_percentages': regime_percentages
        }
        
        # Debug print
        self.debug_manager.debug_print(
            f"Batch {batch_idx} CAPE: {analysis['cape_stats']['mean']:.1f}±{analysis['cape_stats']['std']:.1f} J/kg, "
            f"Regimes: No={regime_percentages['no_lightning']:.1f}%, Mod={regime_percentages['moderate']:.1f}%, "
            f"High={regime_percentages['high']:.1f}%, VHigh={regime_percentages['very_high']:.1f}%",
            "cape"
        )
        
        return analysis
    
    def analyze_physics_loss_components(self, loss_dict: Dict, step: int) -> Dict:
        """Analyze physics loss components for debugging."""
        
        if not self.debug_manager.physics_debug:
            return {}
        
        # Extract physics-related losses
        physics_components = {}
        
        for key, value in loss_dict.items():
            if any(physics_term in key.lower() for physics_term in 
                  ['physics', 'charge', 'cape', 'terrain', 'micro']):
                if isinstance(value, torch.Tensor):
                    physics_components[key] = float(value.detach().cpu())
                else:
                    physics_components[key] = float(value)
        
        # Calculate physics ratios
        main_loss = loss_dict.get('main_loss', loss_dict.get('total_loss', 1.0))
        if isinstance(main_loss, torch.Tensor):
            main_loss = float(main_loss.detach().cpu())
        
        physics_ratios = {}
        for key, value in physics_components.items():
            if main_loss > 0:
                physics_ratios[f"{key}_ratio"] = value / main_loss
        
        analysis = {
            'step': step,
            'physics_components': physics_components,
            'physics_ratios': physics_ratios,
            'total_physics_loss': sum(physics_components.values())
        }
        
        # Debug print key physics metrics
        if 'charge_separation' in physics_components:
            self.debug_manager.debug_print(
                f"Step {step}: Charge loss={physics_components['charge_separation']:.6f}, "
                f"Ratio={physics_ratios.get('charge_separation_ratio', 0):.3f}",
                "physics"
            )
        
        return analysis
    
    def analyze_prediction_physics_alignment(self, predictions: torch.Tensor, 
                                           cape_data: torch.Tensor,
                                           targets: Optional[torch.Tensor] = None,
                                           step: int = 0) -> Dict:
        """Analyze how well predictions align with physics expectations."""
        
        if not self.debug_manager.physics_debug:
            return {}
        
        # Convert to numpy
        pred_np = predictions.detach().cpu().numpy().flatten()
        cape_np = cape_data.detach().cpu().numpy().flatten()
        
        # Remove invalid values
        valid_mask = ~(np.isnan(cape_np) | np.isnan(pred_np))
        pred_np = pred_np[valid_mask]
        cape_np = cape_np[valid_mask]
        
        if len(pred_np) == 0:
            return {}
        
        # Apply sigmoid if predictions look like logits
        if pred_np.min() < 0 or pred_np.max() > 1:
            pred_np = 1 / (1 + np.exp(-pred_np))  # Sigmoid
        
        # Physics thresholds
        thresholds = {
            'no_lightning': 1000.0,
            'moderate': 2500.0,
            'high': 4000.0,
            'saturation': 5000.0
        }
        
        # Analyze each regime
        regime_analysis = {}
        regimes = [
            ('no_lightning', cape_np < thresholds['no_lightning']),
            ('moderate', (cape_np >= thresholds['no_lightning']) & 
                        (cape_np < thresholds['moderate'])),
            ('high', (cape_np >= thresholds['moderate']) & 
                     (cape_np < thresholds['high'])),
            ('very_high', cape_np >= thresholds['high'])
        ]
        
        expected_rates = {
            'no_lightning': 0.05,
            'moderate': 0.25,
            'high': 0.50,
            'very_high': 0.75
        }
        
        for regime_name, mask in regimes:
            if mask.sum() == 0:
                continue
            
            regime_preds = pred_np[mask]
            actual_rate = (regime_preds > 0.5).mean()
            expected_rate = expected_rates[regime_name]
            mean_prob = regime_preds.mean()
            
            alignment_score = 1.0 - abs(actual_rate - expected_rate)
            
            regime_analysis[regime_name] = {
                'pixel_count': int(mask.sum()),
                'actual_lightning_rate': float(actual_rate),
                'expected_lightning_rate': float(expected_rate),
                'mean_probability': float(mean_prob),
                'physics_alignment': float(alignment_score)
            }
        
        # Overall correlation
        correlation = np.corrcoef(cape_np, pred_np)[0, 1] if len(cape_np) > 1 else 0.0
        
        analysis = {
            'step': step,
            'cape_prediction_correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'regime_analysis': regime_analysis,
            'overall_prediction_stats': {
                'mean': float(pred_np.mean()),
                'std': float(pred_np.std()),
                'min': float(pred_np.min()),
                'max': float(pred_np.max())
            }
        }
        
        # Debug print key alignment metrics
        self.debug_manager.debug_print(
            f"Step {step}: CAPE-Pred correlation={correlation:.4f}, "
            f"Mean pred={pred_np.mean():.4f}",
            "physics"
        )
        
        # Print regime alignment
        for regime, data in regime_analysis.items():
            if data['pixel_count'] > 0:
                self.debug_manager.debug_print(
                    f"  {regime}: actual={data['actual_lightning_rate']:.3f}, "
                    f"expected={data['expected_lightning_rate']:.3f}, "
                    f"alignment={data['physics_alignment']:.3f}",
                    "physics"
                )
        
        return analysis
    
    def save_debug_checkpoint(self, analysis_data: Dict, checkpoint_name: str):
        """Save debug analysis data to file."""
        
        if not self.debug_manager.save_debug_outputs:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{checkpoint_name}_{timestamp}.json"
        filepath = self.debug_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(analysis_data, f, indent=2, default=str)
            
            self.debug_manager.debug_print(
                f"Saved debug checkpoint: {filepath}",
                "general"
            )
        except Exception as e:
            logger.warning(f"Failed to save debug checkpoint: {e}")

class CAPEPhysicsMonitor:
    """Monitor for tracking CAPE physics performance during training."""
    
    def __init__(self, debug_manager: DebugManager):
        self.debug_manager = debug_manager
        self.physics_debugger = PhysicsDebugger(debug_manager)
        
        # History tracking
        self.correlation_history = []
        self.alignment_history = []
        self.loss_component_history = []
    
    def track_training_step(self, predictions: torch.Tensor,
                           cape_data: torch.Tensor,
                           loss_dict: Dict,
                           targets: Optional[torch.Tensor] = None,
                           step: int = 0):
        """Track physics performance for a training step."""
        
        if not self.debug_manager.physics_debug:
            return
        
        # Analyze physics alignment
        alignment_analysis = self.physics_debugger.analyze_prediction_physics_alignment(
            predictions, cape_data, targets, step
        )
        
        # Analyze loss components
        loss_analysis = self.physics_debugger.analyze_physics_loss_components(
            loss_dict, step
        )
        
        # Store in history
        if alignment_analysis:
            self.correlation_history.append({
                'step': step,
                'correlation': alignment_analysis['cape_prediction_correlation']
            })
            
            # Calculate average alignment across regimes
            regime_alignments = [data['physics_alignment'] 
                               for data in alignment_analysis['regime_analysis'].values()]
            avg_alignment = np.mean(regime_alignments) if regime_alignments else 0.0
            
            self.alignment_history.append({
                'step': step,
                'average_alignment': avg_alignment
            })
        
        if loss_analysis:
            self.loss_component_history.append(loss_analysis)
        
        # Save checkpoint periodically
        if step % 100 == 0 and self.debug_manager.save_debug_outputs:
            checkpoint_data = {
                'step': step,
                'alignment_analysis': alignment_analysis,
                'loss_analysis': loss_analysis,
                'correlation_history': self.correlation_history[-10:],  # Last 10 steps
                'alignment_history': self.alignment_history[-10:]
            }
            self.physics_debugger.save_debug_checkpoint(
                checkpoint_data, f"training_physics_step_{step}"
            )
    
    def get_physics_summary(self) -> Dict:
        """Get summary of physics performance over training."""
        
        if not self.correlation_history:
            return {}
        
        correlations = [item['correlation'] for item in self.correlation_history]
        alignments = [item['average_alignment'] for item in self.alignment_history]
        
        return {
            'latest_correlation': correlations[-1] if correlations else 0.0,
            'best_correlation': max(correlations) if correlations else 0.0,
            'average_correlation': np.mean(correlations) if correlations else 0.0,
            'latest_alignment': alignments[-1] if alignments else 0.0,
            'best_alignment': max(alignments) if alignments else 0.0,
            'average_alignment': np.mean(alignments) if alignments else 0.0,
            'correlation_trend': correlations[-5:] if len(correlations) >= 5 else correlations,
            'total_steps_tracked': len(self.correlation_history)
        }

# Global debug manager instance
_debug_manager = None
_physics_monitor = None

def get_debug_manager(config: Optional[DictConfig] = None) -> DebugManager:
    """Get global debug manager instance."""
    global _debug_manager
    if _debug_manager is None or config is not None:
        _debug_manager = DebugManager(config)
    return _debug_manager

def get_physics_monitor(config: Optional[DictConfig] = None) -> CAPEPhysicsMonitor:
    """Get global physics monitor instance."""
    global _physics_monitor
    debug_manager = get_debug_manager(config)
    
    if _physics_monitor is None:
        _physics_monitor = CAPEPhysicsMonitor(debug_manager)
    
    return _physics_monitor

def debug_print(message: str, category: str = "general"):
    """Convenience function for debug printing."""
    get_debug_manager().debug_print(message, category)

def is_debug_enabled(category: str = "general") -> bool:
    """Check if debug is enabled for a specific category."""
    debug_manager = get_debug_manager()
    
    category_checks = {
        "general": debug_manager.enabled,
        "memory": debug_manager.memory_tracking,
        "metrics": debug_manager.metrics_debug,
        "batch": debug_manager.batch_info,
        "verbose": debug_manager.verbose_logging,
        "physics": debug_manager.physics_debug,
        "cape": debug_manager.cape_analysis
    }
    
    return category_checks.get(category, False)

# Convenience physics debugging functions
def debug_cape_batch(cape_data: torch.Tensor, batch_idx: int = 0) -> Dict:
    """Debug CAPE data in a batch."""
    debug_manager = get_debug_manager()
    physics_debugger = PhysicsDebugger(debug_manager)
    return physics_debugger.analyze_cape_batch(cape_data, batch_idx)

def debug_physics_loss(loss_dict: Dict, step: int) -> Dict:
    """Debug physics loss components."""
    debug_manager = get_debug_manager()
    physics_debugger = PhysicsDebugger(debug_manager)
    return physics_debugger.analyze_physics_loss_components(loss_dict, step)

def debug_prediction_alignment(predictions: torch.Tensor, 
                              cape_data: torch.Tensor,
                              targets: Optional[torch.Tensor] = None,
                              step: int = 0) -> Dict:
    """Debug prediction physics alignment."""
    debug_manager = get_debug_manager()
    physics_debugger = PhysicsDebugger(debug_manager)
    return physics_debugger.analyze_prediction_physics_alignment(
        predictions, cape_data, targets, step
    )

def track_physics_step(predictions: torch.Tensor,
                      cape_data: torch.Tensor,
                      loss_dict: Dict,
                      targets: Optional[torch.Tensor] = None,
                      step: int = 0):
    """Track physics performance for a training step."""
    physics_monitor = get_physics_monitor()
    physics_monitor.track_training_step(
        predictions, cape_data, loss_dict, targets, step
    )