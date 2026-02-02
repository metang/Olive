"""Tests for environment module."""
import pytest
from pathlib import Path

from olive_auto.environment import (
    EnvironmentConfig,
    ENVIRONMENT_CONFIGS,
    EnvironmentScriptGenerator,
)


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig dataclass."""

    def test_default_values(self):
        """Test default values for EnvironmentConfig."""
        config = EnvironmentConfig(target="test")
        assert config.target == "test"
        assert config.python_version == "3.10"
        assert config.olive_extras == []
        assert config.pip_packages == []
        assert config.env_vars == {}

    def test_custom_values(self):
        """Test custom values for EnvironmentConfig."""
        config = EnvironmentConfig(
            target="custom",
            python_version="3.11",
            pip_packages=["package1", "package2"],
            env_vars={"KEY": "value"},
        )
        assert config.target == "custom"
        assert config.python_version == "3.11"
        assert len(config.pip_packages) == 2


class TestEnvironmentConfigs:
    """Tests for predefined environment configurations."""

    def test_all_targets_have_config(self):
        """Each target should have an environment configuration."""
        expected_targets = [
            "cpu", "cuda", "qnn", "openvino", "vitisai",
            "webgpu", "tensorrt", "directml", "rocm"
        ]
        for target in expected_targets:
            assert target in ENVIRONMENT_CONFIGS
            assert ENVIRONMENT_CONFIGS[target].target == target

    def test_cpu_config(self):
        """Test CPU environment configuration."""
        cpu = ENVIRONMENT_CONFIGS["cpu"]
        assert "olive-ai" in cpu.pip_packages
        assert "onnxruntime" in cpu.pip_packages

    def test_cuda_config(self):
        """Test CUDA environment configuration."""
        cuda = ENVIRONMENT_CONFIGS["cuda"]
        assert "gpu" in cuda.olive_extras
        assert "onnxruntime-gpu" in cuda.pip_packages
        assert len(cuda.system_requirements) > 0


class TestEnvironmentScriptGenerator:
    """Tests for EnvironmentScriptGenerator class."""

    def test_init(self, tmp_path):
        """Test generator initialization."""
        generator = EnvironmentScriptGenerator(tmp_path, platform="linux")
        assert generator.output_dir == tmp_path
        assert generator.platform == "linux"
        assert generator.env_dir.exists()

    def test_generate_linux_setup(self, tmp_path):
        """Test Linux setup script generation."""
        generator = EnvironmentScriptGenerator(tmp_path, platform="linux")
        script_path = generator.generate_setup_script("cpu")

        assert script_path.exists()
        assert script_path.suffix == ".sh"
        content = script_path.read_text()
        assert "#!/bin/bash" in content
        assert "olive_cpu" in content

    def test_generate_windows_setup(self, tmp_path):
        """Test Windows setup script generation."""
        generator = EnvironmentScriptGenerator(tmp_path, platform="windows")
        script_path = generator.generate_setup_script("cpu")

        assert script_path.exists()
        assert script_path.suffix == ".ps1"
        content = script_path.read_text()
        assert "$ErrorActionPreference" in content

    def test_generate_master_setup_linux(self, tmp_path):
        """Test master setup script generation for Linux."""
        generator = EnvironmentScriptGenerator(tmp_path, platform="linux")
        script_path = generator.generate_master_setup_script(["cpu", "cuda"])

        assert script_path.exists()
        content = script_path.read_text()
        assert "setup_cpu.sh" in content
        assert "setup_cuda.sh" in content

    def test_generate_cleanup_linux(self, tmp_path):
        """Test cleanup script generation for Linux."""
        generator = EnvironmentScriptGenerator(tmp_path, platform="linux")
        script_path = generator.generate_cleanup_script(["cpu", "cuda"])

        assert script_path.exists()
        content = script_path.read_text()
        assert "olive_cpu" in content
        assert "olive_cuda" in content

    def test_generate_cleanup_windows(self, tmp_path):
        """Test cleanup script generation for Windows."""
        generator = EnvironmentScriptGenerator(tmp_path, platform="windows")
        script_path = generator.generate_cleanup_script(["cpu", "cuda"])

        assert script_path.exists()
        content = script_path.read_text()
        assert "olive_cpu" in content
        assert "olive_cuda" in content

    def test_invalid_target_raises(self, tmp_path):
        """Test that invalid target raises ValueError."""
        generator = EnvironmentScriptGenerator(tmp_path, platform="linux")
        with pytest.raises(ValueError, match="Unknown target"):
            generator.generate_setup_script("invalid_target")
