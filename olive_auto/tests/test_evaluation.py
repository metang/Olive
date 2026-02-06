"""
Unit tests for olive-auto evaluation module.

Tests cover:
- Config data models
- Progress tracking
- Dataset management  
- Metric building
- Result formatting
- Main evaluator orchestration
"""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from olive_auto.evaluation.config import (
    AccuracyMethod,
    EvaluationConfig,
    EvaluationSummary,
    Metric,
    ModelEvaluationResult,
    ModelMetadata,
    ModelStatus,
    RecommendationSet,
)
from olive_auto.evaluation.progress_tracker import ProgressTracker
from olive_auto.evaluation.dataset_manager import DatasetManager
from olive_auto.evaluation.metrics_builder import MetricConfigBuilder as MetricsBuilder
from olive_auto.evaluation.result_formatter import ResultFormatter
from olive_auto.evaluation.evaluator import ManifestParser, PipelineEvaluator


class TestConfig:
    """Test configuration data models."""
    
    def test_evaluation_config_sample_fraction(self):
        """Test sample fraction calculation."""
        config_normal = EvaluationConfig(test_mode=False)
        assert config_normal.get_sample_fraction() == 0.10
        
        config_test = EvaluationConfig(test_mode=True)
        assert config_test.get_sample_fraction() == 0.01
        
        config_explicit = EvaluationConfig(sample_fraction=0.05)
        assert config_explicit.get_sample_fraction() == 0.05
    
    def test_model_evaluation_result_success(self):
        """Test successful model evaluation result."""
        result = ModelEvaluationResult(
            model_name="cpu_fp32",
            target="cpu",
            precision="fp32",
            status=ModelStatus.SUCCESS,
            elapsed_time_sec=10.5,
            metrics={"accuracy": 0.95, "latency_p50_ms": 12.3},
            accuracy_method=AccuracyMethod.LM_EVAL,
            accuracy_confidence="high",
            num_samples_evaluated=100,
        )
        
        assert result.is_success()
        assert not result.is_skipped()
        assert result.metrics["accuracy"] == 0.95
    
    def test_model_evaluation_result_skipped(self):
        """Test skipped model evaluation result."""
        result = ModelEvaluationResult(
            model_name="cuda_int4",
            target="cuda",
            precision="int4",
            status=ModelStatus.SKIPPED_NO_DATASET,
            elapsed_time_sec=0,
            error_message="Dataset unavailable",
        )
        
        assert not result.is_success()
        assert result.is_skipped()
        assert result.status == ModelStatus.SKIPPED_NO_DATASET
    
    def test_metric_configuration(self):
        """Test metric data model."""
        metric = Metric(
            name="accuracy",
            metric_type="accuracy",
            sub_types=["perplexity"],
            priority=1,
            evaluator_method="lm-eval",
            fallback_method="perplexity",
        )
        
        assert metric.name == "accuracy"
        assert metric.priority == 1
        assert metric.enabled
        assert metric.evaluator_method == "lm-eval"


class TestProgressTracker:
    """Test progress tracking functionality."""
    
    def test_progress_initialization(self):
        """Test progress tracker initialization."""
        tracker = ProgressTracker(total_models=10, refresh_interval_sec=15)
        
        assert tracker.total_models == 10
        assert tracker.completed == 0
        assert tracker.refresh_interval == 15
    
    def test_time_formatting(self):
        """Test time formatting utility."""
        assert ProgressTracker._format_time(45) == "45s"
        assert ProgressTracker._format_time(75) == "1m15s"
        assert ProgressTracker._format_time(3725) == "1h2m5s"


class TestDatasetManager:
    """Test dataset management."""
    
    def test_initialization(self):
        """Test dataset manager initialization."""
        manager_normal = DatasetManager(test_mode=False)
        assert manager_normal.sample_fraction == 0.10
        
        manager_test = DatasetManager(test_mode=True)
        assert manager_test.sample_fraction == 0.01
    
    def test_dataset_availability(self):
        """Test checking dataset availability."""
        manager = DatasetManager()
        
        assert manager.verify_dataset_available("text-generation")
        assert manager.verify_dataset_available("text-classification")
        assert not manager.verify_dataset_available("unknown-task")


class TestMetricBuilder:
    """Test metric configuration building."""
    
    def test_build_for_text_generation(self):
        """Test building metrics for text-generation task."""
        metrics = MetricsBuilder.build_for_task("text-generation")
        
        assert len(metrics) > 0
        # Check that accuracy is first (highest priority)
        assert metrics[0].metric_type == "accuracy"
        assert metrics[0].priority == 1
    
    def test_build_for_text_classification(self):
        """Test building metrics for text-classification task."""
        metrics = MetricsBuilder.build_for_task("text-classification")
        
        assert len(metrics) > 0
        # Check that accuracy is first
        assert metrics[0].metric_type == "accuracy"
    
    def test_build_for_unknown_task(self):
        """Test building metrics for unknown task."""
        metrics = MetricsBuilder.build_for_task("unknown-task")
        
        # Unknown tasks don't have accuracy metric
        assert all(m.metric_type != "accuracy" for m in metrics)
    
    def test_supported_tasks(self):
        """Test getting list of supported tasks."""
        tasks = MetricsBuilder.get_supported_tasks()
        
        assert "text-generation" in tasks
        assert "text-classification" in tasks
    
    def test_accuracy_metric_query(self):
        """Test querying accuracy metric for specific task."""
        acc_metric = MetricsBuilder.get_accuracy_metric("text-generation")
        
        assert acc_metric is not None
        assert acc_metric.metric_type == "accuracy"
        assert acc_metric.evaluator_method == "lm-eval"


class TestResultFormatter:
    """Test result formatting functionality."""
    
    def test_formatter_initialization(self):
        """Test result formatter initialization."""
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            formatter = ResultFormatter(output_dir)
            
            assert formatter.output_dir == output_dir
            assert output_dir.exists()
    
    def test_json_formatting(self):
        """Test JSON result formatting."""
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            formatter = ResultFormatter(output_dir)
            
            # Create sample results
            models = [
                ModelEvaluationResult(
                    model_name="cpu_fp32",
                    target="cpu",
                    precision="fp32",
                    status=ModelStatus.SUCCESS,
                    elapsed_time_sec=10.0,
                    metrics={"accuracy": 0.95, "latency_p50_ms": 15.2},
                    accuracy_method=AccuracyMethod.LM_EVAL,
                    accuracy_confidence="high",
                    num_samples_evaluated=100,
                )
            ]
            
            summary = EvaluationSummary(
                timestamp="2026-02-05T10:00:00",
                total_models=1,
                evaluated_models=1,
                skipped_models=0,
                failed_models=0,
                total_time_sec=10.0,
                models=models,
                sample_fraction_used=0.10,
            )
            
            json_path = output_dir / "results.json"
            formatter.to_json(summary, json_path)
            
            assert json_path.exists()
            
            # Verify JSON structure
            with open(json_path) as f:
                data = json.load(f)
            
            assert "metadata" in data
            assert "models" in data
            assert len(data["models"]) == 1
            assert data["models"][0]["name"] == "cpu_fp32"
    
    def test_csv_formatting(self):
        """Test CSV result formatting."""
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            formatter = ResultFormatter(output_dir)
            
            # Create sample results
            models = [
                ModelEvaluationResult(
                    model_name="cpu_fp32",
                    target="cpu",
                    precision="fp32",
                    status=ModelStatus.SUCCESS,
                    elapsed_time_sec=10.0,
                    metrics={"accuracy": 0.95, "latency_p50_ms": 15.2},
                    accuracy_method=AccuracyMethod.LM_EVAL,
                    accuracy_confidence="high",
                    num_samples_evaluated=100,
                )
            ]
            
            summary = EvaluationSummary(
                timestamp="2026-02-05T10:00:00",
                total_models=1,
                evaluated_models=1,
                skipped_models=0,
                failed_models=0,
                total_time_sec=10.0,
                models=models,
                sample_fraction_used=0.10,
            )
            
            csv_path = output_dir / "results.csv"
            formatter.to_csv(summary, csv_path)
            
            assert csv_path.exists()
            
            # Verify CSV content
            with open(csv_path) as f:
                lines = f.readlines()
            
            assert len(lines) == 2  # Header + 1 data row
            assert "cpu_fp32" in lines[1]


class TestRecommendationSet:
    """Test recommendation generation."""
    
    def test_recommendation_initialization(self):
        """Test recommendation set initialization."""
        recs = RecommendationSet(
            best_overall="cpu_fp32",
            best_accuracy="cpu_fp32",
            best_latency="cuda_int4",
            best_efficiency="cuda_int4",
        )
        
        assert recs.best_overall == "cpu_fp32"
        assert recs.best_latency == "cuda_int4"


class TestEvaluationSummary:
    """Test evaluation summary aggregation."""
    
    def test_summary_creation(self):
        """Test creating evaluation summary."""
        models = [
            ModelEvaluationResult(
                model_name="cpu_fp32",
                target="cpu",
                precision="fp32",
                status=ModelStatus.SUCCESS,
                elapsed_time_sec=10.0,
                metrics={"accuracy": 0.95},
                num_samples_evaluated=100,
            ),
            ModelEvaluationResult(
                model_name="cuda_int4",
                target="cuda",
                precision="int4",
                status=ModelStatus.SKIPPED_NO_DATASET,
                elapsed_time_sec=0,
                error_message="Dataset unavailable",
            ),
        ]
        
        summary = EvaluationSummary(
            timestamp="2026-02-05T10:00:00",
            total_models=2,
            evaluated_models=1,
            skipped_models=1,
            failed_models=0,
            total_time_sec=10.0,
            models=models,
            sample_fraction_used=0.10,
        )
        
        assert summary.total_models == 2
        assert summary.evaluated_models == 1
        assert len(summary.get_successful_models()) == 1
        assert len(summary.get_skipped_models()) == 1
    
    def test_filter_methods(self):
        """Test filtering methods on summary."""
        models = [
            ModelEvaluationResult(
                model_name="cpu_fp32",
                target="cpu",
                precision="fp32",
                status=ModelStatus.SUCCESS,
                elapsed_time_sec=10.0,
                num_samples_evaluated=100,
            ),
            ModelEvaluationResult(
                model_name="cuda_int4",
                target="cuda",
                precision="int4",
                status=ModelStatus.FAILED,
                elapsed_time_sec=5.0,
                error_message="OOM",
            ),
        ]
        
        summary = EvaluationSummary(
            timestamp="2026-02-05T10:00:00",
            total_models=2,
            evaluated_models=1,
            skipped_models=0,
            failed_models=1,
            total_time_sec=15.0,
            models=models,
        )
        
        assert len(summary.get_successful_models()) == 1
        assert len(summary.get_failed_models()) == 1


class TestManifestParser:
    """Test manifest parsing."""
    
    def test_parse_manifest(self):
        """Test parsing a manifest.json file."""
        with TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            
            # Create sample manifest
            manifest_data = {
                "models": {
                    "cpu_fp32": {
                        "model_path": "/path/to/model.onnx",
                        "task": "text-generation",
                        "timestamp": "2026-02-05T10:00:00",
                    },
                    "cuda_int4": {
                        "model_path": "/path/to/model_int4.onnx",
                        "task": "text-generation",
                        "timestamp": "2026-02-05T10:00:00",
                    },
                }
            }
            
            manifest_path.write_text(json.dumps(manifest_data))
            
            # Parse manifest
            models = ManifestParser.parse(manifest_path)
            
            assert len(models) == 2
            assert "cpu_fp32" in models
            assert models["cpu_fp32"].target == "cpu"
            assert models["cpu_fp32"].precision == "fp32"
            assert models["cuda_int4"].target == "cuda"
            assert models["cuda_int4"].precision == "int4"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
