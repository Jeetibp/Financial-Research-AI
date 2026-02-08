"""
Memory Management Utilities
Helps monitor and optimize memory usage across the application
"""

import gc
import psutil
import logging
from typing import Optional
from functools import wraps

logger = logging.getLogger(__name__)


class MemoryManager:
    """Memory monitoring and cleanup utilities"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.get_memory_usage()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return 0.0
    
    def get_memory_percent(self) -> float:
        """Get memory usage as percentage of total system memory"""
        try:
            return self.process.memory_percent()
        except Exception as e:
            logger.error(f"Error getting memory percent: {e}")
            return 0.0
    
    def force_garbage_collection(self):
        """Force garbage collection to free memory"""
        try:
            collected = gc.collect()
            logger.info(f"Garbage collection freed {collected} objects")
            return collected
        except Exception as e:
            logger.error(f"Error during garbage collection: {e}")
            return 0
    
    def check_memory_threshold(self, threshold_mb: float = 500) -> bool:
        """
        Check if memory usage exceeds threshold
        
        Args:
            threshold_mb: Memory threshold in MB (default: 500MB)
            
        Returns:
            True if memory usage is above threshold
        """
        try:
            current_memory = self.get_memory_usage()
            if current_memory > threshold_mb:
                logger.warning(
                    f"Memory usage ({current_memory:.2f}MB) exceeds threshold ({threshold_mb}MB)"
                )
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking memory threshold: {e}")
            return False
    
    def log_memory_stats(self):
        """Log current memory statistics"""
        try:
            current = self.get_memory_usage()
            percent = self.get_memory_percent()
            increase = current - self.initial_memory
            
            logger.info(
                f"Memory Stats: Current={current:.2f}MB, "
                f"Percent={percent:.2f}%, "
                f"Increase from start={increase:.2f}MB"
            )
        except Exception as e:
            logger.error(f"Error logging memory stats: {e}")
    
    def cleanup_if_needed(self, threshold_mb: float = 500):
        """Run cleanup if memory threshold exceeded"""
        try:
            if self.check_memory_threshold(threshold_mb):
                logger.info("Running memory cleanup...")
                self.force_garbage_collection()
                
                # Log results
                new_memory = self.get_memory_usage()
                logger.info(f"Memory after cleanup: {new_memory:.2f}MB")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def memory_monitored(threshold_mb: float = 500):
    """
    Decorator to monitor memory usage of a function
    
    Args:
        threshold_mb: Memory threshold to trigger cleanup
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            manager = MemoryManager()
            before = manager.get_memory_usage()
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                after = manager.get_memory_usage()
                increase = after - before
                
                if increase > 50:  # Log if function used >50MB
                    logger.info(
                        f"{func.__name__} memory: +{increase:.2f}MB "
                        f"(before={before:.2f}MB, after={after:.2f}MB)"
                    )
                
                # Cleanup if needed
                manager.cleanup_if_needed(threshold_mb)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            manager = MemoryManager()
            before = manager.get_memory_usage()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                after = manager.get_memory_usage()
                increase = after - before
                
                if increase > 50:
                    logger.info(
                        f"{func.__name__} memory: +{increase:.2f}MB "
                        f"(before={before:.2f}MB, after={after:.2f}MB)"
                    )
                
                manager.cleanup_if_needed(threshold_mb)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global memory manager instance
_global_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance"""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager
