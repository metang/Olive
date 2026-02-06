"""
Configuration and data models for olive-auto evaluation system.

This module defines all dataclasses and enums used throughout the evaluation
pipeline, including model metadata, metrics, results, and evaluation configs.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from pathlib import Path


class ModelStatus(str, Enum):
    """Status of model evaluation."""
    SUCCESS = "success"
    SKIPPED_CORRUPTED = "skipped_corrupted"
    SKIPPED_NO_DATASET = "skipped_no_dataset"
    SKIPPED_EVALUATOR_MISMATCH = "skipped_evaluator_mismatch"
    FAILED = "failed"


class AccuracyMethod(str, Enum):
    """Method used to compute accuracy metric."""
    LM_EVAL = "lm-eval"
    PERPLEXITY = "perplexity"
    SKIPPED = "skipped"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation run."""
    
    test_mode: bool = False
    """If True, use 1% of dataset. If False, use 10%."""
    
    sample_fraction: Optional[float] = None
    """Explicit sample fraction. If None, use test_mode to determine."""
    
    include_memory_profiling: bool = True
    """Include both ONNX Runtime and system-level memory measurements."""
    
    output_dir: Path = field(default_factory=lambda: Path("evaluation_results"))
    """Directory to save evaluation results."""
    
    max_workers: int = 1
    """Number of parallel evaluation workers (currently fixed to 1 for sequential)."""
    
    def get_sample_fraction(self) -> float:
        """Get the actual sample fraction to use."""
        if self.sample_fraction is not None:
            return self.sample_fraction
        return 0.01 if self.test_mode else 0.10


@dataclass
class Metric:
    """Definition of a metric to measure."""
    
    name: str
    """Name of the metric (e.g., 'accuracy', 'latency')."""
    
    metric_type: str
    """Type of metric (e.g., 'accuracy', 'latency', 'memory', 'size')."""
    
    sub_types: List[str] = field(default_factory=list)
    """Sub-metrics (e.g., ['p50', 'p99'] for latency)."""
    
    priority: int = 999
    """Priority order (1=highest, lower=more important)."""
    
    enabled: bool = True
    """Whether this metric should be computed."""
    
    evaluator_method: Optional[str] = None
    """Primary evaluation method (e.g., 'lm-eval')."""
    
    fallback_method: Optional[str] = None
    """Fallback evaluation method if primary fails (e.g., 'perplexity')."""


@dataclass
class MemoryMetric:
    """Memory measurement data."""
    
    onnx_runtime_peak_mb: float
    """Peak memory used by ONNX Runtime inference."""
    
    system_peak_mb: float
    """Peak system memory during inference."""
    
    average_mb: float
    """Average system memory during inference."""


@dataclass
class ModelMetadata:
    """Metadata about a converted model from manifest.json."""
    
    name: str
    """Model identifier (e.g., 'cpu_fp32')."""
    
    target: str
    """Target platform (e.g., 'cpu', 'cuda', 'openvino')."""
    
    precision: str
    """Quantization precision (e.g., 'fp32', 'fp16', 'int4')."""
    
    model_path: Path
    """Absolute path to ONNX model file."""
    
    task: str
    """Model task type (e.g., 'text-generation', 'text-classification')."""
    
    timestamp: str
    """When the model was created (ISO 8601)."""
    
    model_name: Optional[str] = None
    """Original HuggingFace model name (e.g., 'qwen/qwen3-0.6b')."""
    
    config_path: Optional[Path] = None
    """Path to model config JSON if available."""


@dataclass
class ModelEvaluationResult:
    """Result of evaluating a single model."""
    
    model_name: str
    """Model identifier from manifest (e.g., 'cpu_fp32')."""
    
    target: str
    """Target platform."""
    
    precision: str
    """Quantization precision."""
    
    status: ModelStatus
    """Status of evaluation (success, skipped, failed)."""
    
    elapsed_time_sec: float
    """Time taken for evaluation."""
    
    metrics: Optional[Dict[str, float]] = None
    """Computed metrics if status == SUCCESS."""
    
    accuracy_method: Optional[AccuracyMethod] = None
    """Which method was used to compute accuracy (lm-eval vs perplexity)."""
    
    accuracy_confidence: Optional[str] = None
    """Confidence in accuracy result ('high' for lm-eval, 'medium' for fallback)."""
    
    error_message: Optional[str] = None
    """Error message if evaluation failed."""
    
    num_samples_evaluated: int = 0
    """Number of samples used in evaluation."""
    
    def is_success(self) -> bool:
        """Check if evaluation succeeded."""
        return self.status == ModelStatus.SUCCESS
    
    def is_skipped(self) -> bool:
        """Check if evaluation was skipped."""
        return self.status.value.startswith("skipped_")


@dataclass
class RecommendationSet:
    """Smart recommendations based on evaluation results."""
    
    best_overall: Optional[str] = None
    """Best model considering accuracy as primary metric."""
    
    best_accuracy: Optional[str] = None
    """Model with highest accuracy."""
    
    best_latency: Optional[str] = None
    """Model with lowest latency (p50)."""
    
    best_efficiency: Optional[str] = None
    """Model with best accuracy/size trade-off."""
    
    fastest_per_target: Dict[str, str] = field(default_factory=dict)
    """Fastest model for each target (cpu, cuda, etc.)."""
    
    best_per_precision: Dict[str, str] = field(default_factory=dict)
    """Best model for each precision (fp32, fp16, int4)."""


@dataclass
class EvaluationSummary:
    """Summary of all evaluation results."""
    
    timestamp: str
    """When evaluation started (ISO 8601)."""
    
    total_models: int
    """Total models in manifest."""
    
    evaluated_models: int
    """Number of successfully evaluated models."""
    
    skipped_models: int
    """Number of skipped models."""
    
    failed_models: int
    """Number of failed models."""
    
    total_time_sec: float
    """Total evaluation time."""
    
    models: List[ModelEvaluationResult] = field(default_factory=list)
    """Results for each model."""
    
    recommendations: RecommendationSet = field(default_factory=RecommendationSet)
    """Smart recommendations based on results."""
    
    sample_fraction_used: float = 0.10
    """Fraction of dataset used for evaluation."""
    
    pipeline_dir: Optional[Path] = None
    """Path to the olive-auto pipeline directory."""
    
    def get_successful_models(self) -> List[ModelEvaluationResult]:
        """Get only successful evaluation results."""
        return [m for m in self.models if m.is_success()]
    
    def get_skipped_models(self) -> List[ModelEvaluationResult]:
        """Get only skipped models."""
        return [m for m in self.models if m.is_skipped()]
    
    def get_failed_models(self) -> List[ModelEvaluationResult]:
        """Get only failed models."""
        return [m for m in self.models if m.status == ModelStatus.FAILED]


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and sampling."""
    
    task: str
    """Model task type (text-generation, text-classification, etc.)."""
    
    dataset_name: str
    """Name of dataset to load."""
    
    dataset_split: str
    """Which split to use (validation, test, etc.)."""
    
    sample_fraction: float = 0.10
    """Fraction of dataset to sample."""
    
    max_samples: Optional[int] = None
    """Maximum number of samples (overrides fraction if set)."""
    
    min_samples: int = 1
    """Minimum samples required (skip if less)."""
    
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "olive-auto")
    """Cache directory for downloaded datasets."""


# Default dataset configurations by task
DATASET_CONFIGS = {
    "text-generation": DatasetConfig(
        task="text-generation",
        dataset_name="wikitext",
        dataset_split="validation",
        sample_fraction=0.10,
    ),
    "text-classification": DatasetConfig(
        task="text-classification",
        dataset_name="glue",
        dataset_split="validation_matched",
        sample_fraction=0.10,
    ),
}

# Default metric priorities by task
METRIC_PRIORITIES = {
    "text-generation": [
        Metric(
            name="accuracy",
            metric_type="accuracy",
            sub_types=["perplexity"],
            priority=1,
            evaluator_method="lm-eval",
            fallback_method="perplexity",
        ),
        Metric(
            name="latency",
            metric_type="latency",
            sub_types=["p50", "p99"],
            priority=2,
        ),
        Metric(
            name="memory",
            metric_type="memory",
            sub_types=["onnx_runtime_peak", "system_peak", "average"],
            priority=3,
        ),
        Metric(
            name="size",
            metric_type="size",
            sub_types=["model_bytes"],
            priority=4,
        ),
    ],
    "text-classification": [
        Metric(
            name="accuracy",
            metric_type="accuracy",
            sub_types=["accuracy_score", "f1"],
            priority=1,
            evaluator_method="sklearn",
            fallback_method=None,
        ),
        Metric(
            name="latency",
            metric_type="latency",
            sub_types=["p50", "p99"],
            priority=2,
        ),
        Metric(
            name="memory",
            metric_type="memory",
            sub_types=["onnx_runtime_peak", "system_peak", "average"],
            priority=3,
        ),
        Metric(
            name="size",
            metric_type="size",
            sub_types=["model_bytes"],
            priority=4,
        ),
    ],
    "default": [
        Metric(
            name="latency",
            metric_type="latency",
            sub_types=["p50", "p99"],
            priority=1,
        ),
        Metric(
            name="memory",
            metric_type="memory",
            sub_types=["onnx_runtime_peak", "system_peak", "average"],
            priority=2,
        ),
        Metric(
            name="size",
            metric_type="size",
            sub_types=["model_bytes"],
            priority=3,
        ),
    ],
}
