"""Tests for executor module."""
import json
import pytest
from pathlib import Path

from olive_auto.executor import (
    ExecutionResult,
    PipelineExecutor,
)


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_successful_result(self):
        """Test successful execution result."""
        result = ExecutionResult(
            target="cpu",
            precision="fp32",
            success=True,
            return_code=0,
            log_file=Path("test.log"),
        )
        assert result.success is True
        assert result.return_code == 0
        assert result.error_message is None

    def test_failed_result(self):
        """Test failed execution result."""
        result = ExecutionResult(
            target="cpu",
            precision="fp32",
            success=False,
            return_code=1,
            log_file=Path("test.log"),
            error_message="Test error",
        )
        assert result.success is False
        assert result.return_code == 1
        assert result.error_message == "Test error"


class TestPipelineExecutor:
    """Tests for PipelineExecutor class."""

    @pytest.fixture
    def pipeline_dir(self, tmp_path):
        """Create a mock pipeline directory with manifest."""
        # Create manifest
        manifest = {
            "model_id": "bert-base-uncased",
            "task": "fill-mask",
            "category": "nlp",
            "targets": ["cpu"],
            "precisions": {"cpu": ["fp32"]},
            "generated_by": "olive-auto",
            "version": "0.1.0",
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        # Create directories
        (tmp_path / "commands").mkdir()
        (tmp_path / "logs").mkdir()

        return tmp_path

    def test_init(self, pipeline_dir):
        """Test executor initialization."""
        executor = PipelineExecutor(pipeline_dir)
        assert executor.pipeline_dir == pipeline_dir
        assert executor.parallel == 1
        assert executor.manifest["model_id"] == "bert-base-uncased"

    def test_init_parallel(self, pipeline_dir):
        """Test executor initialization with parallel option."""
        executor = PipelineExecutor(pipeline_dir, parallel=4)
        assert executor.parallel == 4

    def test_load_manifest(self, pipeline_dir):
        """Test manifest loading."""
        executor = PipelineExecutor(pipeline_dir)
        assert executor.manifest["task"] == "fill-mask"
        assert executor.manifest["category"] == "nlp"

    def test_missing_manifest(self, tmp_path):
        """Test error when manifest is missing."""
        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            PipelineExecutor(tmp_path)

    def test_get_commands_empty(self, pipeline_dir):
        """Test getting commands when no scripts exist."""
        executor = PipelineExecutor(pipeline_dir)
        commands = executor._get_commands()
        assert len(commands) == 0

    def test_get_commands_with_scripts(self, pipeline_dir):
        """Test getting commands when scripts exist."""
        import sys

        # Create a mock script with platform-appropriate extension
        ext = ".ps1" if sys.platform == "win32" else ".sh"
        script_path = pipeline_dir / "commands" / f"cpu_fp32{ext}"
        script_path.write_text("echo test")

        executor = PipelineExecutor(pipeline_dir)
        commands = executor._get_commands()

        assert len(commands) == 1
        assert commands[0][0] == "cpu"
        assert commands[0][1] == "fp32"


class TestPipelineExecutorSummary:
    """Tests for PipelineExecutor summary functionality."""

    @pytest.fixture
    def executor_with_results(self, tmp_path):
        """Create an executor with mock results."""
        # Create manifest
        manifest = {
            "model_id": "bert-base-uncased",
            "task": "fill-mask",
            "category": "nlp",
            "targets": ["cpu"],
            "precisions": {"cpu": ["fp32", "int8"]},
            "generated_by": "olive-auto",
            "version": "0.1.0",
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        (tmp_path / "commands").mkdir()
        (tmp_path / "logs").mkdir()

        executor = PipelineExecutor(tmp_path)

        # Add mock results
        executor.results = [
            ExecutionResult(
                target="cpu",
                precision="fp32",
                success=True,
                return_code=0,
                log_file=tmp_path / "logs" / "cpu_fp32.log",
            ),
            ExecutionResult(
                target="cpu",
                precision="int8",
                success=False,
                return_code=1,
                log_file=tmp_path / "logs" / "cpu_int8.log",
                error_message="Test error",
            ),
        ]

        return executor

    def test_print_summary_mixed_results(self, executor_with_results):
        """Test summary with mixed results."""
        # Create log files
        (executor_with_results.pipeline_dir / "logs" / "cpu_fp32.log").write_text("Success")
        (executor_with_results.pipeline_dir / "logs" / "cpu_int8.log").write_text("Failed")

        success = executor_with_results._print_summary()

        assert success is False  # Because there's a failure
        assert len(executor_with_results.results) == 2
