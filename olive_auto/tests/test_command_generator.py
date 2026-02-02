"""Tests for command generator module."""
import pytest
from pathlib import Path

from olive_auto.command_generator import (
    OptimizationCommand,
    CommandGenerator,
)


class TestOptimizationCommand:
    """Tests for OptimizationCommand dataclass."""

    def test_basic_command(self):
        """Test basic optimization command creation."""
        cmd = OptimizationCommand(
            target="cpu",
            precision="fp32",
            model_id="bert-base-uncased",
            task="fill-mask",
            output_dir=Path("./output"),
        )
        assert cmd.target == "cpu"
        assert cmd.precision == "fp32"
        assert cmd.model_id == "bert-base-uncased"

    def test_to_olive_command(self):
        """Test olive command generation."""
        cmd = OptimizationCommand(
            target="cpu",
            precision="fp32",
            model_id="bert-base-uncased",
            task="fill-mask",
            output_dir=Path("./output"),
        )
        olive_cmd = cmd.to_olive_command()

        assert "olive optimize" in olive_cmd
        assert "bert-base-uncased" in olive_cmd
        assert "fill-mask" in olive_cmd
        assert "CPUExecutionProvider" in olive_cmd
        assert "fp32" in olive_cmd

    def test_to_olive_command_with_device(self):
        """Test olive command with device flag."""
        cmd = OptimizationCommand(
            target="cuda",
            precision="fp16",
            model_id="bert-base-uncased",
            task="fill-mask",
            output_dir=Path("./output"),
        )
        olive_cmd = cmd.to_olive_command()

        assert "--device gpu" in olive_cmd

    def test_to_olive_command_windows(self):
        """Test Windows olive command generation."""
        cmd = OptimizationCommand(
            target="cpu",
            precision="fp32",
            model_id="bert-base-uncased",
            task="fill-mask",
            output_dir=Path("./output"),
        )
        olive_cmd = cmd.to_olive_command_windows()

        assert "olive optimize" in olive_cmd
        assert "`" in olive_cmd  # PowerShell line continuation


class TestCommandGenerator:
    """Tests for CommandGenerator class."""

    def test_init(self, tmp_path):
        """Test generator initialization."""
        generator = CommandGenerator(
            model_id="bert-base-uncased",
            task="fill-mask",
            category="nlp",
            output_dir=tmp_path,
            targets=["cpu"],
            platform="linux",
        )
        assert generator.model_id == "bert-base-uncased"
        assert generator.task == "fill-mask"
        assert generator.category == "nlp"
        assert generator.commands_dir.exists()

    def test_get_compatible_targets(self, tmp_path):
        """Test compatible targets retrieval."""
        generator = CommandGenerator(
            model_id="bert-base-uncased",
            task="fill-mask",
            category="nlp",
            output_dir=tmp_path,
            platform="linux",
        )
        # nlp category should have cpu and cuda
        assert "cpu" in generator.targets
        assert "cuda" in generator.targets

    def test_generate_all_commands(self, tmp_path):
        """Test generation of all commands."""
        generator = CommandGenerator(
            model_id="bert-base-uncased",
            task="fill-mask",
            category="nlp",
            output_dir=tmp_path,
            targets=["cpu"],
            platform="linux",
        )
        commands = generator.generate_all_commands()

        # CPU has 4 precisions: fp32, fp16, int8, int4
        assert len(commands) == 4

        precisions = [cmd.precision for cmd in commands]
        assert "fp32" in precisions
        assert "int8" in precisions

    def test_generate_command_scripts_linux(self, tmp_path):
        """Test Linux command script generation."""
        generator = CommandGenerator(
            model_id="bert-base-uncased",
            task="fill-mask",
            category="nlp",
            output_dir=tmp_path,
            targets=["cpu"],
            precisions={"cpu": ["fp32"]},
            platform="linux",
        )
        scripts = generator.generate_command_scripts()

        assert "cpu_fp32" in scripts
        assert scripts["cpu_fp32"].suffix == ".sh"
        assert scripts["cpu_fp32"].exists()

        content = scripts["cpu_fp32"].read_text()
        assert "#!/bin/bash" in content
        assert "olive optimize" in content

    def test_generate_command_scripts_windows(self, tmp_path):
        """Test Windows command script generation."""
        generator = CommandGenerator(
            model_id="bert-base-uncased",
            task="fill-mask",
            category="nlp",
            output_dir=tmp_path,
            targets=["cpu"],
            precisions={"cpu": ["fp32"]},
            platform="windows",
        )
        scripts = generator.generate_command_scripts()

        assert "cpu_fp32" in scripts
        assert scripts["cpu_fp32"].suffix == ".ps1"
        assert scripts["cpu_fp32"].exists()

        content = scripts["cpu_fp32"].read_text()
        assert "$ErrorActionPreference" in content

    def test_generate_master_script_linux(self, tmp_path):
        """Test master script generation for Linux."""
        generator = CommandGenerator(
            model_id="bert-base-uncased",
            task="fill-mask",
            category="nlp",
            output_dir=tmp_path,
            targets=["cpu"],
            precisions={"cpu": ["fp32", "int8"]},
            platform="linux",
        )
        script_path = generator.generate_master_script()

        assert script_path.exists()
        assert script_path.name == "run_all_optimizations.sh"

        content = script_path.read_text()
        assert "cpu_fp32.sh" in content
        assert "cpu_int8.sh" in content

    def test_generate_master_script_windows(self, tmp_path):
        """Test master script generation for Windows."""
        generator = CommandGenerator(
            model_id="bert-base-uncased",
            task="fill-mask",
            category="nlp",
            output_dir=tmp_path,
            targets=["cpu"],
            precisions={"cpu": ["fp32"]},
            platform="windows",
        )
        script_path = generator.generate_master_script()

        assert script_path.exists()
        assert script_path.name == "run_all_optimizations.ps1"


class TestCommandGeneratorMultipleTargets:
    """Tests for CommandGenerator with multiple targets."""

    def test_multiple_targets(self, tmp_path):
        """Test generation with multiple targets."""
        generator = CommandGenerator(
            model_id="bert-base-uncased",
            task="fill-mask",
            category="nlp",
            output_dir=tmp_path,
            targets=["cpu", "cuda"],
            platform="linux",
        )
        commands = generator.generate_all_commands()

        # CPU: 4 precisions, CUDA: 6 precisions
        assert len(commands) == 10

        targets = set(cmd.target for cmd in commands)
        assert "cpu" in targets
        assert "cuda" in targets

    def test_custom_precisions(self, tmp_path):
        """Test generation with custom precisions."""
        generator = CommandGenerator(
            model_id="bert-base-uncased",
            task="fill-mask",
            category="nlp",
            output_dir=tmp_path,
            targets=["cpu", "cuda"],
            precisions={"cpu": ["fp32"], "cuda": ["fp16"]},
            platform="linux",
        )
        commands = generator.generate_all_commands()

        assert len(commands) == 2

        cpu_cmds = [c for c in commands if c.target == "cpu"]
        cuda_cmds = [c for c in commands if c.target == "cuda"]

        assert len(cpu_cmds) == 1
        assert cpu_cmds[0].precision == "fp32"
        assert len(cuda_cmds) == 1
        assert cuda_cmds[0].precision == "fp16"
