import os
import sys
import psutil
import torch
import traceback
import time
import logging
from functools import wraps
from typing import Dict, Any, Optional
import threading
import signal
import gc

class MemoryTracker:
    """
    Detailed memory tracker to identify OOM kill locations with line-by-line tracking.
    Now respects debug configuration settings.
    """
    
    def __init__(self, log_interval: float = 5.0, memory_threshold_gb: float = 0.8, debug_enabled: bool = True):
        """
        Initialize memory tracker.
        
        Args:
            log_interval: Seconds between memory logs
            memory_threshold_gb: Memory threshold ratio (0.8 = 80% of available)
            debug_enabled: Whether memory tracking is enabled
        """
        self.log_interval = log_interval
        self.memory_threshold = memory_threshold_gb
        self.debug_enabled = debug_enabled
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread = None
        
        # Only setup logging if debug is enabled
        if self.debug_enabled:
            # Setup logging
            self.logger = logging.getLogger(f'MemoryTracker-{os.getpid()}')
            handler = logging.FileHandler(f'memory_trace_{os.getpid()}.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - LINE:%(lineno)d - %(funcName)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = None
        
        # Track peak usage
        self.peak_memory_gb = 0.0
        self.peak_gpu_memory_gb = 0.0
        
        # Setup signal handlers for SIGTERM/SIGKILL detection only if debug enabled
        if self.debug_enabled:
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGUSR1, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        if not self.debug_enabled or not self.logger:
            return
            
        self.logger.critical(f"SIGNAL RECEIVED: {signum}")
        self.logger.critical(f"TERMINATION_LOCATION: {frame.f_code.co_filename}:{frame.f_lineno}")
        self.logger.critical(f"FUNCTION: {frame.f_code.co_name}")
        self.log_current_memory("TERMINATION_SIGNAL")
        self._dump_stack_trace()
        
    def _dump_stack_trace(self):
        """Dump complete stack trace."""
        if not self.debug_enabled or not self.logger:
            return
            
        self.logger.critical("FULL STACK TRACE AT TERMINATION:")
        for line in traceback.format_stack():
            self.logger.critical(line.strip())
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory information."""
        # System memory
        system_memory = psutil.virtual_memory()
        
        # Process memory
        process_memory = self.process.memory_info()
        
        # GPU memory if available
        gpu_memory_allocated = 0.0
        gpu_memory_reserved = 0.0
        gpu_memory_total = 0.0
        
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return {
            'system_total_gb': system_memory.total / (1024**3),
            'system_available_gb': system_memory.available / (1024**3),
            'system_used_gb': system_memory.used / (1024**3),
            'system_percent': system_memory.percent,
            'process_rss_gb': process_memory.rss / (1024**3),
            'process_vms_gb': process_memory.vms / (1024**3),
            'gpu_allocated_gb': gpu_memory_allocated,
            'gpu_reserved_gb': gpu_memory_reserved,
            'gpu_total_gb': gpu_memory_total,
            'gpu_percent': (gpu_memory_allocated / gpu_memory_total * 100) if gpu_memory_total > 0 else 0
        }
    
    def log_current_memory(self, context: str = ""):
        """Log current memory state with context - ONLY if debug enabled."""
        # EARLY RETURN if debug disabled
        if not self.debug_enabled:
            return
            
        try:
            memory_info = self.get_memory_info()
            
            # Update peaks
            self.peak_memory_gb = max(self.peak_memory_gb, memory_info['process_rss_gb'])
            self.peak_gpu_memory_gb = max(self.peak_gpu_memory_gb, memory_info['gpu_allocated_gb'])
            
            # Get current line info
            frame = sys._getframe(1)
            filename = os.path.basename(frame.f_code.co_filename)
            line_number = frame.f_lineno
            function_name = frame.f_code.co_name
            
            log_msg = (
                f"MEMORY_CHECK [{context}] - "
                f"FILE: {filename}:{line_number} FUNC: {function_name} - "
                f"RAM: {memory_info['process_rss_gb']:.2f}GB "
                f"({memory_info['system_percent']:.1f}% system) - "
                f"GPU: {memory_info['gpu_allocated_gb']:.2f}GB "
                f"({memory_info['gpu_percent']:.1f}% GPU) - "
                f"Peak RAM: {self.peak_memory_gb:.2f}GB - "
                f"Peak GPU: {self.peak_gpu_memory_gb:.2f}GB"
            )
            
            # Only log if logger exists
            if self.logger:
                self.logger.info(log_msg)
            
            # Only print to console if debug enabled
            print(f"[MEMORY] {log_msg}")
            
            # Check if approaching limits
            if memory_info['system_percent'] > self.memory_threshold * 100:
                warning_msg = f"WARNING: Memory usage high ({memory_info['system_percent']:.1f}%)"
                if self.logger:
                    self.logger.warning(warning_msg)
                print(f"[MEMORY WARNING] {warning_msg}")
                
            if memory_info['gpu_percent'] > self.memory_threshold * 100:
                warning_msg = f"WARNING: GPU memory usage high ({memory_info['gpu_percent']:.1f}%)"
                if self.logger:
                    self.logger.warning(warning_msg)
                print(f"[GPU WARNING] {warning_msg}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error logging memory: {e}")
    
    def start_monitoring(self):
        """Start background memory monitoring - ONLY if debug enabled."""
        if not self.debug_enabled:
            return
            
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        if self.logger:
            self.logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop background memory monitoring."""
        if not self.debug_enabled:
            return
            
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        if self.logger:
            self.logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                self.log_current_memory("BACKGROUND_MONITOR")
                time.sleep(self.log_interval)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Monitor loop error: {e}")
                break

def memory_checkpoint(context: str = ""):
    """Decorator to add memory checkpoints to functions - respects debug settings."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if debug is enabled
            try:
                from .debug_utils import is_debug_enabled
                debug_enabled = is_debug_enabled("memory")
            except ImportError:
                # Fallback if debug utils not available
                debug_enabled = os.getenv('LIGHTNING_DEBUG', '').lower() == 'true'
            
            if not debug_enabled:
                # Just run the function without memory tracking
                return func(*args, **kwargs)
            
            # Log entry
            tracker = getattr(wrapper, '_memory_tracker', None)
            if tracker is None:
                tracker = MemoryTracker(debug_enabled=debug_enabled)
                wrapper._memory_tracker = tracker
            
            tracker.log_current_memory(f"ENTER_{func.__name__}_{context}")
            
            try:
                result = func(*args, **kwargs)
                tracker.log_current_memory(f"EXIT_{func.__name__}_{context}")
                return result
            except Exception as e:
                tracker.log_current_memory(f"ERROR_{func.__name__}_{context}")
                if tracker.logger:
                    tracker.logger.error(f"Exception in {func.__name__}: {e}")
                tracker._dump_stack_trace()
                raise
        return wrapper
    return decorator

def trace_memory_line():
    """Manual memory checkpoint - call this at specific lines you want to trace."""
    # Check if debug is enabled
    try:
        from .debug_utils import is_debug_enabled
        debug_enabled = is_debug_enabled("memory")
    except ImportError:
        # Fallback if debug utils not available
        debug_enabled = os.getenv('LIGHTNING_DEBUG', '').lower() == 'true'
    
    if not debug_enabled:
        return  # Do nothing if debug disabled
    
    frame = sys._getframe(1)
    filename = os.path.basename(frame.f_code.co_filename)
    line_number = frame.f_lineno
    function_name = frame.f_code.co_name
    
    # Get or create tracker
    if not hasattr(trace_memory_line, '_tracker'):
        trace_memory_line._tracker = MemoryTracker(debug_enabled=debug_enabled)
    
    trace_memory_line._tracker.log_current_memory(f"MANUAL_TRACE_{filename}_{function_name}")

# Global tracker for easy access
_global_tracker = None

def get_global_tracker() -> MemoryTracker:
    """Get global memory tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        # Check if debug is enabled
        try:
            from .debug_utils import is_debug_enabled
            debug_enabled = is_debug_enabled("memory")
        except ImportError:
            # Fallback if debug utils not available
            debug_enabled = os.getenv('LIGHTNING_DEBUG', '').lower() == 'true'
        
        _global_tracker = MemoryTracker(debug_enabled=debug_enabled)
    return _global_tracker

def start_global_monitoring():
    """Start global memory monitoring - only if debug enabled."""
    tracker = get_global_tracker()
    tracker.start_monitoring()
    return tracker

def stop_global_monitoring():
    """Stop global memory monitoring."""
    global _global_tracker
    if _global_tracker:
        _global_tracker.stop_monitoring()

# Context manager for memory tracking
class MemoryContext:
    """Context manager for memory tracking - respects debug settings."""
    
    def __init__(self, context_name: str):
        self.context_name = context_name
        
        # Check if debug is enabled
        try:
            from .debug_utils import is_debug_enabled
            self.debug_enabled = is_debug_enabled("memory")
        except ImportError:
            # Fallback if debug utils not available
            self.debug_enabled = os.getenv('LIGHTNING_DEBUG', '').lower() == 'true'
        
        if self.debug_enabled:
            self.tracker = get_global_tracker()
        else:
            self.tracker = None
    
    def __enter__(self):
        if self.tracker:
            self.tracker.log_current_memory(f"ENTER_CONTEXT_{self.context_name}")
        return self.tracker
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tracker:
            if exc_type:
                self.tracker.log_current_memory(f"ERROR_CONTEXT_{self.context_name}")
                if self.tracker.logger:
                    self.tracker.logger.error(f"Exception in context {self.context_name}: {exc_val}")
            else:
                self.tracker.log_current_memory(f"EXIT_CONTEXT_{self.context_name}")