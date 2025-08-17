"""
Debug utilities with configurable debug switches.
"""

import logging
import os
from typing import Any, Optional, Dict
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

class DebugManager:
    """Centralized debug management with configurable switches."""
    
    def __init__(self, config: Optional[DictConfig] = None):
        """Initialize debug manager with configuration."""
        self.config = config
        self._debug_enabled = False
        self._verbose_logging = False
        self._memory_tracking = False
        self._metrics_debug = False
        self._batch_info = False
        self._save_debug_outputs = False
        
        # Load debug settings from config
        if config and hasattr(config, 'training') and hasattr(config.training, 'debug'):
            debug_config = config.training.debug
            self._debug_enabled = getattr(debug_config, 'enabled', False)
            self._verbose_logging = getattr(debug_config, 'verbose_logging', False)
            self._memory_tracking = getattr(debug_config, 'memory_tracking', False)
            self._metrics_debug = getattr(debug_config, 'metrics_debug', False)
            self._batch_info = getattr(debug_config, 'batch_info', False)
            self._save_debug_outputs = getattr(debug_config, 'save_debug_outputs', False)
        
        # Environment variable override
        if os.getenv('LIGHTNING_DEBUG', '').lower() == 'true':
            self._debug_enabled = True
            self._verbose_logging = True
            self._memory_tracking = True
            self._metrics_debug = True
            self._batch_info = True
            logger.info("Debug mode enabled via environment variable")
    
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
    
    def debug_print(self, message: str, category: str = "general"):
        """Conditional debug print based on category."""
        if not self._debug_enabled:
            return
            
        category_enabled = {
            "general": self._debug_enabled,
            "memory": self.memory_tracking,
            "metrics": self.metrics_debug,
            "batch": self.batch_info,
            "verbose": self.verbose_logging
        }
        
        if category_enabled.get(category, False):
            print(f"[DEBUG-{category.upper()}] {message}")
    
    def conditional_trace_memory(self, checkpoint_name: str):
        """Conditional memory tracing."""
        if self.memory_tracking:
            try:
                from .memory_tracker import trace_memory_line
                trace_memory_line()  # Only call if memory tracking enabled
            except ImportError:
                # Memory tracker not available
                pass


# Global debug manager instance
_debug_manager = None

def get_debug_manager(config: Optional[DictConfig] = None) -> DebugManager:
    """Get global debug manager instance."""
    global _debug_manager
    if _debug_manager is None or config is not None:
        _debug_manager = DebugManager(config)
    return _debug_manager

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
        "verbose": debug_manager.verbose_logging
    }
    
    return category_checks.get(category, False)