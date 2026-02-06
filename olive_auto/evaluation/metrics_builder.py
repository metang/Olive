"""
Metric configuration builder for model evaluation.

Creates task-specific metric configurations with accuracy as the primary metric,
supporting both primary (lm-eval) and fallback (perplexity) methods.
"""

import logging
from typing import List, Optional

from .config import Metric

logger = logging.getLogger(__name__)


class MetricConfigBuilder:
    """Build metric configurations for different model tasks."""
    
    # Task-specific metric configurations
    # Ordered by priority: accuracy >> latency > memory > size
    TASK_METRICS = {
        "text-generation": [
            Metric(
                name="accuracy",
                metric_type="accuracy",
                sub_types=["perplexity"],
                priority=1,
                enabled=True,
                evaluator_method="lm-eval",
                fallback_method="perplexity",
            ),
            Metric(
                name="latency",
                metric_type="latency",
                sub_types=["p50", "p99"],
                priority=2,
                enabled=True,
            ),
            Metric(
                name="memory",
                metric_type="memory",
                sub_types=["onnx_runtime_peak", "system_peak", "average"],
                priority=3,
                enabled=True,
            ),
            Metric(
                name="size",
                metric_type="size",
                sub_types=["model_bytes"],
                priority=4,
                enabled=True,
            ),
        ],
        "text-classification": [
            Metric(
                name="accuracy",
                metric_type="accuracy",
                sub_types=["accuracy_score", "f1"],
                priority=1,
                enabled=True,
                evaluator_method="sklearn",
                fallback_method=None,
            ),
            Metric(
                name="latency",
                metric_type="latency",
                sub_types=["p50", "p99"],
                priority=2,
                enabled=True,
            ),
            Metric(
                name="memory",
                metric_type="memory",
                sub_types=["onnx_runtime_peak", "system_peak", "average"],
                priority=3,
                enabled=True,
            ),
            Metric(
                name="size",
                metric_type="size",
                sub_types=["model_bytes"],
                priority=4,
                enabled=True,
            ),
        ],
    }
    
    # Default metrics for unknown tasks (accuracy skipped)
    DEFAULT_METRICS = [
        Metric(
            name="latency",
            metric_type="latency",
            sub_types=["p50", "p99"],
            priority=1,
            enabled=True,
        ),
        Metric(
            name="memory",
            metric_type="memory",
            sub_types=["onnx_runtime_peak", "system_peak", "average"],
            priority=2,
            enabled=True,
        ),
        Metric(
            name="size",
            metric_type="size",
            sub_types=["model_bytes"],
            priority=3,
            enabled=True,
        ),
    ]
    
    @classmethod
    def build_for_task(cls, task: str, include_accuracy: bool = True) -> List[Metric]:
        """
        Build metric configuration for a given task.
        
        Accuracy is the highest priority metric, followed by latency, memory, and size.
        Supports fallback methods for accuracy (e.g., lm-eval -> perplexity).
        
        Args:
            task: Model task type (text-generation, text-classification, etc.)
            include_accuracy: Whether to include accuracy metric (skip for unknown tasks)
            
        Returns:
            List of Metric objects ordered by priority
        """
        if task in cls.TASK_METRICS:
            metrics = cls.TASK_METRICS[task]
        else:
            logger.warning(
                f"Unknown task '{task}', using default metrics (no accuracy)"
            )
            metrics = cls.DEFAULT_METRICS
        
        # Filter accuracy metric if requested
        if not include_accuracy:
            metrics = [m for m in metrics if m.metric_type != "accuracy"]
        
        # Sort by priority
        metrics = sorted(metrics, key=lambda m: m.priority)
        
        return metrics
    
    @classmethod
    def get_supported_tasks(cls) -> List[str]:
        """
        Get list of tasks with specialized metric configurations.
        
        Returns:
            List of supported task names
        """
        return list(cls.TASK_METRICS.keys())
    
    @classmethod
    def get_accuracy_metric(cls, task: str) -> Optional[Metric]:
        """
        Get accuracy metric configuration for a task.
        
        Args:
            task: Model task type
            
        Returns:
            Accuracy metric if available, None otherwise
        """
        metrics = cls.TASK_METRICS.get(task)
        if metrics:
            for metric in metrics:
                if metric.metric_type == "accuracy":
                    return metric
        return None
    
    @classmethod
    def has_accuracy_metric(cls, task: str) -> bool:
        """
        Check if a task has an accuracy metric available.
        
        Args:
            task: Model task type
            
        Returns:
            True if accuracy metric is available for this task
        """
        return cls.get_accuracy_metric(task) is not None
