"""
Main evaluation orchestrator for olive-auto pipeline results.

Coordinates evaluation of all models in a pipeline, handling data loading,
metric computation, progress tracking, and result aggregation with smart
recommendations.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import onnx

from .config import (
    AccuracyMethod,
    EvaluationConfig,
    EvaluationSummary,
    Metric,
    ModelEvaluationResult,
    ModelMetadata,
    ModelStatus,
    RecommendationSet,
)
from .dataset_manager import DatasetManager
from .metrics_builder import MetricConfigBuilder
from .progress_tracker import ProgressTracker
from .result_formatter import ResultFormatter

logger = logging.getLogger(__name__)


class ManifestParser:
    """Parse olive-auto manifest.json to extract model metadata."""
    
    @staticmethod
    def parse(manifest_path: Path) -> Dict[str, ModelMetadata]:
        """
        Parse manifest.json and return model metadata.
        
        Args:
            manifest_path: Path to manifest.json
            
        Returns:
            Dictionary mapping model names to ModelMetadata objects
        """
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path) as f:
            manifest_data = json.load(f)
        
        models = {}
        
        # Manifest structure varies, handle common formats
        models_data = manifest_data.get("models", manifest_data.get("optimized_models", {}))
        
        for model_name, model_info in models_data.items():
            try:
                # Extract metadata with fallbacks
                model_path = model_info.get("model_path") or model_info.get("path")
                if not model_path:
                    logger.warning(f"No model path for {model_name}, skipping")
                    continue
                
                # Parse model name to extract target and precision
                # Format: cpu_fp32, cuda_int4, etc.
                parts = model_name.split("_")
                if len(parts) >= 2:
                    target = parts[0]
                    precision = "_".join(parts[1:])
                else:
                    target = "unknown"
                    precision = "unknown"
                
                metadata = ModelMetadata(
                    name=model_name,
                    target=target,
                    precision=precision,
                    model_path=Path(model_path),
                    task=model_info.get("task", "unknown"),
                    timestamp=model_info.get("timestamp", datetime.now().isoformat()),
                    model_name=model_info.get("model_name"),
                    config_path=Path(model_info.get("config_path")) if model_info.get("config_path") else None,
                )
                
                models[model_name] = metadata
                logger.debug(f"Parsed model: {model_name} (target={target}, precision={precision})")
            
            except Exception as e:
                logger.warning(f"Failed to parse model {model_name}: {e}")
                continue
        
        logger.info(f"Parsed {len(models)} models from manifest")
        return models


class PipelineEvaluator:
    """Main evaluation orchestrator."""
    
    def __init__(
        self,
        pipeline_dir: Path,
        test_mode: bool = False,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize pipeline evaluator.
        
        Args:
            pipeline_dir: Path to olive-auto pipeline directory
            test_mode: If True, use 1% of dataset for faster evaluation
            output_dir: Directory for evaluation results (default: pipeline_dir/evaluation_results)
        """
        self.pipeline_dir = Path(pipeline_dir)
        self.test_mode = test_mode
        self.sample_fraction = 0.01 if test_mode else 0.10
        
        # Set output directory
        if output_dir is None:
            output_dir = self.pipeline_dir / "evaluation_results"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.manifest_path = self.pipeline_dir / "manifest.json"
        self.dataset_manager = DatasetManager(test_mode=test_mode)
        self.progress_tracker: Optional[ProgressTracker] = None
        
        # Parse manifest
        try:
            self.models = ManifestParser.parse(self.manifest_path)
        except Exception as e:
            logger.error(f"Failed to parse manifest: {e}")
            self.models = {}
    
    def evaluate_all_models(self) -> EvaluationSummary:
        """
        Evaluate all models in the pipeline.
        
        Returns:
            EvaluationSummary with results for all models
        """
        if not self.models:
            logger.error("No models to evaluate")
            return EvaluationSummary(
                timestamp=datetime.now().isoformat(),
                total_models=0,
                evaluated_models=0,
                skipped_models=0,
                failed_models=0,
                total_time_sec=0,
            )
        
        start_time = time.time()
        
        # Initialize progress tracker
        self.progress_tracker = ProgressTracker(total_models=len(self.models))
        
        results = []
        
        for model_name, metadata in self.models.items():
            self.progress_tracker.start_model(model_name)
            
            model_start = time.time()
            result = self._evaluate_single_model(metadata)
            elapsed = time.time() - model_start
            
            result.elapsed_time_sec = elapsed
            results.append(result)
            
            # Update progress
            self.progress_tracker.finish_model(
                model_name=model_name,
                elapsed_sec=elapsed,
                success=result.is_success()
            )
        
        # Finalize progress
        self.progress_tracker.finish_all()
        
        # Aggregate results
        summary = self._aggregate_results(results, time.time() - start_time)
        
        # Save results
        formatter = ResultFormatter(self.output_dir)
        formatter.save_results(summary)
        
        return summary
    
    def _evaluate_single_model(self, metadata: ModelMetadata) -> ModelEvaluationResult:
        """
        Evaluate a single model.
        
        Handles errors gracefully: skips if model invalid, dataset unavailable,
        or evaluator unsupported.
        
        Args:
            metadata: Model metadata
            
        Returns:
            ModelEvaluationResult with metrics or error status
        """
        try:
            # 1. Validate model file
            if not self._validate_model_file(metadata.model_path):
                logger.warning(f"Skipping {metadata.name}: model file invalid or missing")
                return ModelEvaluationResult(
                    model_name=metadata.name,
                    target=metadata.target,
                    precision=metadata.precision,
                    status=ModelStatus.SKIPPED_CORRUPTED,
                    elapsed_time_sec=0,
                    error_message="Model file corrupted or missing",
                )
            
            # 2. Get evaluation data
            dataset = self.dataset_manager.get_evaluation_data(
                task=metadata.task,
                model_name=metadata.name,
            )
            
            if dataset is None:
                logger.warning(f"Skipping {metadata.name}: dataset unavailable for task '{metadata.task}'")
                return ModelEvaluationResult(
                    model_name=metadata.name,
                    target=metadata.target,
                    precision=metadata.precision,
                    status=ModelStatus.SKIPPED_NO_DATASET,
                    elapsed_time_sec=0,
                    error_message=f"Dataset unavailable for task '{metadata.task}'",
                )
            
            # 3. Build metrics config
            metrics = MetricConfigBuilder.build_for_task(metadata.task)
            
            if not metrics:
                logger.warning(f"Skipping {metadata.name}: no evaluator for task '{metadata.task}'")
                return ModelEvaluationResult(
                    model_name=metadata.name,
                    target=metadata.target,
                    precision=metadata.precision,
                    status=ModelStatus.SKIPPED_EVALUATOR_MISMATCH,
                    elapsed_time_sec=0,
                    error_message=f"No evaluator for task '{metadata.task}'",
                )
            
            # 4. Run evaluation
            computed_metrics = {}
            accuracy_method = None
            accuracy_confidence = None
            
            for metric in metrics:
                if metric.metric_type == "accuracy":
                    # Try primary method first, fallback if needed
                    primary_value = self._compute_accuracy_metric(
                        model_path=metadata.model_path,
                        dataset=dataset,
                        method=metric.evaluator_method,
                    )
                    
                    if primary_value is not None:
                        computed_metrics["accuracy"] = primary_value
                        accuracy_method = AccuracyMethod.LM_EVAL if metric.evaluator_method == "lm-eval" else AccuracyMethod.PERPLEXITY
                        accuracy_confidence = "high" if metric.evaluator_method == "lm-eval" else "medium"
                    elif metric.fallback_method:
                        # Try fallback
                        fallback_value = self._compute_accuracy_metric(
                            model_path=metadata.model_path,
                            dataset=dataset,
                            method=metric.fallback_method,
                        )
                        
                        if fallback_value is not None:
                            computed_metrics["accuracy"] = fallback_value
                            accuracy_method = AccuracyMethod.PERPLEXITY
                            accuracy_confidence = "medium"
                    
                    if "accuracy" not in computed_metrics:
                        logger.warning(f"Could not compute accuracy for {metadata.name}")
                
                elif metric.metric_type == "latency":
                    # Compute latency percentiles
                    latencies = self._compute_latency_metric(
                        model_path=metadata.model_path,
                        dataset=dataset,
                    )
                    if latencies:
                        computed_metrics.update(latencies)
                
                elif metric.metric_type == "memory":
                    # Compute memory metrics
                    memory_metrics = self._compute_memory_metric(
                        model_path=metadata.model_path,
                        dataset=dataset,
                    )
                    if memory_metrics:
                        computed_metrics.update(memory_metrics)
                
                elif metric.metric_type == "size":
                    # Compute model size
                    size_mb = metadata.model_path.stat().st_size / (1024 * 1024)
                    computed_metrics["model_size_mb"] = size_mb
            
            return ModelEvaluationResult(
                model_name=metadata.name,
                target=metadata.target,
                precision=metadata.precision,
                status=ModelStatus.SUCCESS,
                elapsed_time_sec=0,  # Will be set by caller
                metrics=computed_metrics,
                accuracy_method=accuracy_method,
                accuracy_confidence=accuracy_confidence,
                num_samples_evaluated=len(dataset),
            )
        
        except Exception as e:
            logger.error(f"Evaluation failed for {metadata.name}: {e}")
            return ModelEvaluationResult(
                model_name=metadata.name,
                target=metadata.target,
                precision=metadata.precision,
                status=ModelStatus.FAILED,
                elapsed_time_sec=0,
                error_message=str(e),
            )
    
    @staticmethod
    def _validate_model_file(model_path: Path) -> bool:
        """
        Validate ONNX model file.
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if valid, False otherwise
        """
        if not model_path.exists():
            return False
        
        try:
            onnx.load(str(model_path))
            return True
        except Exception:
            return False
    
    def _compute_accuracy_metric(
        self,
        model_path: Path,
        dataset: List[Dict],
        method: Optional[str] = None,
    ) -> Optional[float]:
        """
        Compute accuracy metric (placeholder).
        
        In production, this would integrate with lm-eval or sklearn.
        For now, returns a mock value.
        
        Args:
            model_path: Path to model
            dataset: Evaluation dataset
            method: Evaluation method (lm-eval, sklearn, perplexity, etc.)
            
        Returns:
            Accuracy value or None if not available
        """
        # TODO: Implement actual accuracy computation
        # For now, return mock value
        logger.debug(f"Computing accuracy for {model_path.name} using {method}")
        return 0.85  # Mock value
    
    def _compute_latency_metric(
        self,
        model_path: Path,
        dataset: List[Dict],
    ) -> Optional[Dict[str, float]]:
        """
        Compute latency metrics (placeholder).
        
        In production, this would run inference and measure timing.
        
        Args:
            model_path: Path to model
            dataset: Evaluation dataset
            
        Returns:
            Dictionary with latency_p50_ms, latency_p99_ms, etc.
        """
        # TODO: Implement actual latency computation
        logger.debug(f"Computing latency for {model_path.name}")
        return {
            "latency_p50_ms": 15.2,
            "latency_p99_ms": 45.8,
        }
    
    def _compute_memory_metric(
        self,
        model_path: Path,
        dataset: List[Dict],
    ) -> Optional[Dict[str, float]]:
        """
        Compute memory metrics (placeholder).
        
        In production, this would profile memory during inference.
        
        Args:
            model_path: Path to model
            dataset: Evaluation dataset
            
        Returns:
            Dictionary with memory measurements
        """
        # TODO: Implement actual memory profiling
        logger.debug(f"Computing memory for {model_path.name}")
        return {
            "memory_onnx_runtime_peak_mb": 234.5,
            "memory_system_peak_mb": 256.0,
            "memory_average_mb": 200.0,
        }
    
    def _aggregate_results(
        self,
        results: List[ModelEvaluationResult],
        total_time: float,
    ) -> EvaluationSummary:
        """
        Aggregate results and compute recommendations.
        
        Args:
            results: List of model evaluation results
            total_time: Total evaluation time
            
        Returns:
            EvaluationSummary with aggregated results and recommendations
        """
        successful = [r for r in results if r.is_success()]
        skipped = [r for r in results if r.is_skipped()]
        failed = [r for r in results if r.status == ModelStatus.FAILED]
        
        # Compute recommendations
        recommendations = self._compute_recommendations(successful)
        
        summary = EvaluationSummary(
            timestamp=datetime.now().isoformat(),
            total_models=len(results),
            evaluated_models=len(successful),
            skipped_models=len(skipped),
            failed_models=len(failed),
            total_time_sec=total_time,
            models=results,
            recommendations=recommendations,
            sample_fraction_used=self.sample_fraction,
            pipeline_dir=self.pipeline_dir,
        )
        
        return summary
    
    @staticmethod
    def _compute_recommendations(
        successful_models: List[ModelEvaluationResult],
    ) -> RecommendationSet:
        """
        Compute smart recommendations based on results.
        
        Args:
            successful_models: List of successfully evaluated models
            
        Returns:
            RecommendationSet with recommendations
        """
        if not successful_models:
            return RecommendationSet()
        
        recommendations = RecommendationSet()
        
        # Best accuracy
        if any(m.metrics and "accuracy" in m.metrics for m in successful_models):
            best_accuracy = max(
                (m for m in successful_models if m.metrics and "accuracy" in m.metrics),
                key=lambda m: m.metrics.get("accuracy", 0),
                default=None
            )
            if best_accuracy:
                recommendations.best_accuracy = best_accuracy.model_name
                recommendations.best_overall = best_accuracy.model_name
        
        # Best latency
        if any(m.metrics and "latency_p50_ms" in m.metrics for m in successful_models):
            best_latency = min(
                (m for m in successful_models if m.metrics and "latency_p50_ms" in m.metrics),
                key=lambda m: m.metrics.get("latency_p50_ms", float("inf")),
                default=None
            )
            if best_latency:
                recommendations.best_latency = best_latency.model_name
        
        # Best efficiency (accuracy / size)
        if any(m.metrics for m in successful_models):
            efficiencies = []
            for m in successful_models:
                if m.metrics and "accuracy" in m.metrics and "model_size_mb" in m.metrics:
                    size_gb = m.metrics.get("model_size_mb", 1) / 1024
                    efficiency = m.metrics.get("accuracy", 0) / max(size_gb, 0.001)
                    efficiencies.append((m, efficiency))
            
            if efficiencies:
                best_efficiency = max(efficiencies, key=lambda x: x[1])[0]
                recommendations.best_efficiency = best_efficiency.model_name
        
        return recommendations
