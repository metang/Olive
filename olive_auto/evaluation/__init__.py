"""
Olive-Auto Evaluation Module

Provides evaluation capabilities for olive-auto generated models, including
accuracy assessment, latency measurement, memory profiling, and result reporting.
"""

from .config import (
    AccuracyMethod,
    DatasetConfig,
    EvaluationConfig,
    EvaluationSummary,
    Metric,
    MemoryMetric,
    ModelEvaluationResult,
    ModelMetadata,
    ModelStatus,
    RecommendationSet,
    DATASET_CONFIGS,
    METRIC_PRIORITIES,
)

__all__ = [
    "AccuracyMethod",
    "DatasetConfig",
    "EvaluationConfig",
    "EvaluationSummary",
    "Metric",
    "MemoryMetric",
    "ModelEvaluationResult",
    "ModelMetadata",
    "ModelStatus",
    "RecommendationSet",
    "DATASET_CONFIGS",
    "METRIC_PRIORITIES",
]
