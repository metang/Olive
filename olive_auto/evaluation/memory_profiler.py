"""
Memory profiling utilities for model evaluation.

Tracks peak memory usage during inference using psutil.
"""

import logging
import threading
from typing import Dict, Optional

import psutil

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Track peak memory usage during inference."""

    def __init__(self, process_id: Optional[int] = None):
        """
        Initialize memory profiler.

        Args:
            process_id: Process ID to monitor (default: current process)
        """
        self.process = psutil.Process(process_id)
        self.peak_rss_mb = 0.0  # Resident Set Size (actual physical memory)
        self.peak_vms_mb = 0.0  # Virtual Memory Size
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self, interval_sec: float = 0.1) -> None:
        """
        Start monitoring memory in background thread.

        Args:
            interval_sec: Monitoring interval in seconds
        """
        if self.monitoring:
            logger.warning("Memory monitoring already active")
            return

        self.monitoring = True
        self.peak_rss_mb = 0.0
        self.peak_vms_mb = 0.0

        def monitor():
            try:
                while self.monitoring:
                    try:
                        mem_info = self.process.memory_info()
                        rss_mb = mem_info.rss / (1024 * 1024)
                        vms_mb = mem_info.vms / (1024 * 1024)

                        self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
                        self.peak_vms_mb = max(self.peak_vms_mb, vms_mb)

                    except (psutil.AccessDenied, psutil.NoSuchProcess) as e:
                        logger.warning(f"Failed to read memory: {e}")
                        break

                    # Sleep for the interval
                    import time
                    time.sleep(interval_sec)

            except Exception as e:
                logger.error(f"Memory monitoring thread failed: {e}")

        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self) -> Dict[str, float]:
        """
        Stop monitoring and return peak memory measurements.

        Returns:
            Dictionary with 'peak_rss_mb' and 'peak_vms_mb' keys
        """
        self.monitoring = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

        return {
            "memory_peak_rss_mb": self.peak_rss_mb,
            "memory_peak_vms_mb": self.peak_vms_mb,
        }

    def get_current_memory(self) -> Dict[str, float]:
        """
        Get current memory usage.

        Returns:
            Dictionary with current memory measurements
        """
        try:
            mem_info = self.process.memory_info()
            return {
                "memory_rss_mb": mem_info.rss / (1024 * 1024),
                "memory_vms_mb": mem_info.vms / (1024 * 1024),
            }
        except Exception as e:
            logger.error(f"Failed to get current memory: {e}")
            return {}

    def get_peak_memory(self) -> Dict[str, float]:
        """
        Get peak memory measurements recorded so far.

        Returns:
            Dictionary with peak memory measurements
        """
        return {
            "memory_peak_rss_mb": self.peak_rss_mb,
            "memory_peak_vms_mb": self.peak_vms_mb,
        }


def measure_memory_usage(func):
    """
    Decorator to measure peak memory during function execution.

    Args:
        func: Function to measure

    Returns:
        Decorated function that returns (result, memory_dict)
    """
    def wrapper(*args, **kwargs):
        profiler = MemoryProfiler()
        profiler.start_monitoring()

        try:
            result = func(*args, **kwargs)
        finally:
            memory_dict = profiler.stop_monitoring()

        return result, memory_dict

    return wrapper
