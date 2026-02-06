"""
Progress tracking for model evaluation with periodic updates.

Displays real-time progress with 15-second refresh intervals, showing
completion percentage, elapsed time, and estimated time to completion.
"""

import time
from datetime import timedelta
from typing import Optional


class ProgressTracker:
    """Track and display evaluation progress with 15-second updates."""
    
    def __init__(self, total_models: int, refresh_interval_sec: int = 15):
        """
        Initialize progress tracker.
        
        Args:
            total_models: Total number of models to evaluate
            refresh_interval_sec: Seconds between progress updates (default: 15)
        """
        self.total_models = total_models
        self.completed = 0
        self.refresh_interval = refresh_interval_sec
        
        self.start_time = time.time()
        self.last_refresh_time = self.start_time
        
        self.current_model_name: Optional[str] = None
        self.current_model_start_time: Optional[float] = None
    
    def start_model(self, model_name: str) -> None:
        """
        Mark the start of evaluation for a model.
        
        Args:
            model_name: Name of model being evaluated (e.g., 'cpu_fp32')
        """
        self.current_model_name = model_name
        self.current_model_start_time = time.time()
        print(f"\n[{self.completed + 1}/{self.total_models}] Starting: {model_name}")
    
    def update_metric(self, metric_name: str, status: str) -> None:
        """
        Update status of metric computation.
        
        Args:
            metric_name: Name of metric being computed
            status: Current status ('computing', 'complete', 'failed', etc.)
        """
        indent = "  └─ "
        print(f"{indent}{metric_name}: {status}")
    
    def finish_model(self, model_name: str, elapsed_sec: float, success: bool) -> None:
        """
        Mark completion of model evaluation.
        
        Args:
            model_name: Name of evaluated model
            elapsed_sec: Time taken for evaluation
            success: Whether evaluation succeeded
        """
        self.completed += 1
        status_icon = "✓" if success else "✗"
        print(f"{status_icon} {model_name} completed in {self._format_time(elapsed_sec)}")
        
        # Check if we should refresh progress bar
        current_time = time.time()
        if current_time - self.last_refresh_time >= self.refresh_interval:
            self._render_progress()
            self.last_refresh_time = current_time
    
    def _render_progress(self) -> None:
        """
        Render progress bar with statistics.
        
        Shows format: [████████░░░░░░░░░░] 8/10 (80%) | Elapsed: 5m23s | ETA: 1m22s
        """
        elapsed = time.time() - self.start_time
        progress = self.completed / self.total_models if self.total_models > 0 else 0
        
        # Build progress bar (30 chars wide)
        filled = int(progress * 30)
        bar = "█" * filled + "░" * (30 - filled)
        
        # Calculate ETA
        if progress > 0:
            total_estimated = elapsed / progress
            eta_seconds = total_estimated - elapsed
            eta_str = self._format_time(max(0, eta_seconds))
        else:
            eta_str = "calculating..."
        
        # Print progress line
        print(
            f"\n[{bar}] {self.completed}/{self.total_models} ({progress*100:.0f}%) | "
            f"Elapsed: {self._format_time(elapsed)} | ETA: {eta_str}"
        )
    
    def finish_all(self) -> None:
        """Print final completion message."""
        elapsed = time.time() - self.start_time
        self._render_progress()
        print(f"\n✓ Evaluation complete in {self._format_time(elapsed)}")
        print(f"  ({self.completed}/{self.total_models} models evaluated)")
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """
        Format seconds into human-readable time string.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted string (e.g., "1m23s", "45s", "2h15m")
        """
        td = timedelta(seconds=int(seconds))
        
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h{minutes}m{seconds}s"
        elif minutes > 0:
            return f"{minutes}m{seconds}s"
        else:
            return f"{seconds}s"
