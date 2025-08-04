import psutil
import logging
import tracemalloc
from typing import Dict

import torch

class MemoryTracker:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.initial_memory = None
        self.peak_memory = 0
        
        if self.enabled:
            tracemalloc.start()
            
    def get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        memory_stats = {}
        
        # System memory
        process = psutil.Process()
        memory_stats['system_memory_gb'] = process.memory_info().rss / 1024**3
        memory_stats['system_memory_percent'] = process.memory_percent()
        
        # GPU memory if available
        if torch.cuda.is_available():
            memory_stats['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / 1024**3
            memory_stats['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / 1024**3
            memory_stats['gpu_memory_max_allocated_gb'] = torch.cuda.max_memory_allocated() / 1024**3
        
        # Python memory tracking
        if self.enabled and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            memory_stats['python_current_mb'] = current / 1024**2
            memory_stats['python_peak_mb'] = peak / 1024**2
            
            if peak > self.peak_memory:
                self.peak_memory = peak
        
        return memory_stats
    
    def log_memory_usage(self, logger: logging.Logger, context: str = ""):
        """Log current memory usage with context"""
        if not self.enabled:
            return
            
        stats = self.get_current_memory()
        logger.info(f"Memory Usage {context}:")
        logger.info(f"  System: {stats.get('system_memory_gb', 0):.2f}GB ({stats.get('system_memory_percent', 0):.1f}%)")
        
        if torch.cuda.is_available():
            logger.info(f"  GPU Allocated: {stats.get('gpu_memory_allocated_gb', 0):.2f}GB")
            logger.info(f"  GPU Reserved: {stats.get('gpu_memory_reserved_gb', 0):.2f}GB")
            logger.info(f"  GPU Max Allocated: {stats.get('gpu_memory_max_allocated_gb', 0):.2f}GB")
        
        if tracemalloc.is_tracing():
            logger.info(f"  Python Current: {stats.get('python_current_mb', 0):.1f}MB")
            logger.info(f"  Python Peak: {stats.get('python_peak_mb', 0):.1f}MB")
    
    def detect_memory_leak(self, logger: logging.Logger, threshold_mb: float = 100.0) -> bool:
        """Detect potential memory leaks"""
        if not self.enabled or not tracemalloc.is_tracing():
            return False
            
        current, peak = tracemalloc.get_traced_memory()
        
        if self.initial_memory is None:
            self.initial_memory = current
            return False
        
        memory_increase = (current - self.initial_memory) / 1024**2
        
        if memory_increase > threshold_mb:
            logger.warning(f"Potential memory leak detected! Memory increased by {memory_increase:.1f}MB")
            
            # Get top memory consumers
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:5]
            
            logger.warning("Top memory consumers:")
            for stat in top_stats:
                logger.warning(f"  {stat}")
            
            return True
        
        return False
    
    def cleanup(self):
        """Cleanup memory tracker"""
        if self.enabled and tracemalloc.is_tracing():
            tracemalloc.stop()
