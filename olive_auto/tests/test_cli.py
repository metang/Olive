"""Tests for CLI module."""
import json
import pytest
from pathlib import Path
from typer.testing import CliRunner

from olive_auto.cli import app

runner = CliRunner()


class TestCLIHelp:
    """Tests for CLI help and version."""

    def test_help(self):
        """Test help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "olive-auto" in result.output or "model" in result.output

    def test_version(self):
        """Test version command."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestCLIDryRun:
    """Tests for CLI dry run mode."""

    def test_dry_run_basic(self):
        """Test basic dry run."""
        result = runner.invoke(app, [
            "-m", "bert-base-uncased",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "Dry run" in result.output

    def test_dry_run_with_targets(self):
        """Test dry run with specific targets."""
        result = runner.invoke(app, [
            "-m", "bert-base-uncased",
            "-t", "cpu",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "cpu" in result.output


class TestCLIPipelineGeneration:
    """Tests for CLI pipeline generation."""

    def test_generate_pipeline_basic(self, tmp_path):
        """Test basic pipeline generation."""
        result = runner.invoke(app, [
            "-m", "bert-base-uncased",
            "-o", str(tmp_path),
            "-t", "cpu",
            "-p", "fp32",
        ])

        assert result.exit_code == 0
        assert (tmp_path / "manifest.json").exists()
        assert (tmp_path / "commands").exists()
        assert (tmp_path / "environments").exists()

    def test_generate_pipeline_manifest(self, tmp_path):
        """Test generated manifest content."""
        result = runner.invoke(app, [
            "-m", "bert-base-uncased",
            "-o", str(tmp_path),
            "-t", "cpu",
            "-p", "fp32",
        ])

        assert result.exit_code == 0

        manifest_path = tmp_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())

        assert manifest["model_id"] == "bert-base-uncased"
        assert manifest["generated_by"] == "olive-auto"
        assert "cpu" in manifest["targets"]

    def test_generate_pipeline_scripts(self, tmp_path):
        """Test generated scripts."""
        result = runner.invoke(app, [
            "-m", "bert-base-uncased",
            "-o", str(tmp_path),
            "-t", "cpu",
            "-p", "fp32",
        ])

        assert result.exit_code == 0

        # Check for command scripts
        commands_dir = tmp_path / "commands"
        assert commands_dir.exists()

        # Should have cpu_fp32 script
        scripts = list(commands_dir.glob("cpu_fp32.*"))
        assert len(scripts) == 1


class TestCLIPlatform:
    """Tests for CLI platform options."""

    def test_linux_platform(self, tmp_path):
        """Test Linux platform generation."""
        result = runner.invoke(app, [
            "-m", "bert-base-uncased",
            "-o", str(tmp_path),
            "-t", "cpu",
            "-p", "fp32",
            "--platform", "linux",
        ])

        assert result.exit_code == 0
        assert (tmp_path / "run_all_optimizations.sh").exists()
        assert (tmp_path / "setup_environments.sh").exists()
        assert (tmp_path / "cleanup.sh").exists()

    def test_windows_platform(self, tmp_path):
        """Test Windows platform generation."""
        result = runner.invoke(app, [
            "-m", "bert-base-uncased",
            "-o", str(tmp_path),
            "-t", "cpu",
            "-p", "fp32",
            "--platform", "windows",
        ])

        assert result.exit_code == 0
        assert (tmp_path / "run_all_optimizations.ps1").exists()
        assert (tmp_path / "setup_environments.ps1").exists()
        assert (tmp_path / "cleanup.ps1").exists()


class TestCLITargetPrecision:
    """Tests for CLI target and precision options."""

    def test_multiple_targets(self, tmp_path):
        """Test multiple targets."""
        result = runner.invoke(app, [
            "-m", "bert-base-uncased",
            "-o", str(tmp_path),
            "-t", "cpu",
            "-t", "cuda",
            "-p", "fp32",
        ])

        assert result.exit_code == 0

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert "cpu" in manifest["targets"]
        # cuda may or may not be included depending on compatibility

    def test_multiple_precisions(self, tmp_path):
        """Test multiple precisions."""
        result = runner.invoke(app, [
            "-m", "bert-base-uncased",
            "-o", str(tmp_path),
            "-t", "cpu",
            "-p", "fp32",
            "-p", "int8",
        ])

        assert result.exit_code == 0

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert "fp32" in manifest["precisions"]["cpu"]
        assert "int8" in manifest["precisions"]["cpu"]

    def test_exclude_targets(self, tmp_path):
        """Test excluding targets."""
        result = runner.invoke(app, [
            "-m", "bert-base-uncased",
            "-o", str(tmp_path),
            "--exclude-targets", "cuda",
            "-p", "fp32",
        ])

        assert result.exit_code == 0

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert "cuda" not in manifest["targets"]


class TestCLIVerbose:
    """Tests for CLI verbose option."""

    def test_verbose_output(self, tmp_path):
        """Test verbose output."""
        result = runner.invoke(app, [
            "-m", "bert-base-uncased",
            "-o", str(tmp_path),
            "-t", "cpu",
            "-p", "fp32",
            "-v",
        ])

        assert result.exit_code == 0
