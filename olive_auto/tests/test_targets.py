"""Tests for targets module."""
import pytest

from olive_auto.targets import (
    TARGET_PRECISION_MATRIX,
    CATEGORY_TARGET_COMPATIBILITY,
    TARGET_PROVIDER_MAP,
    ALL_TARGETS,
    ALL_PRECISIONS,
    get_provider_for_target,
    get_compatible_targets,
    get_precisions_for_target,
    get_default_precision,
    get_device_for_target,
    validate_target,
    validate_precision,
)


class TestTargetPrecisionMatrix:
    """Tests for target precision matrix."""

    def test_all_targets_have_precisions(self):
        """Each target should have a list of supported precisions."""
        for target, config in TARGET_PRECISION_MATRIX.items():
            assert "precisions" in config
            assert len(config["precisions"]) > 0
            assert isinstance(config["precisions"], list)

    def test_all_targets_have_default(self):
        """Each target should have a default precision."""
        for target, config in TARGET_PRECISION_MATRIX.items():
            assert "default" in config
            assert config["default"] in config["precisions"]

    def test_cpu_target(self):
        """Test CPU target configuration."""
        cpu = TARGET_PRECISION_MATRIX["cpu"]
        assert "fp32" in cpu["precisions"]
        assert "int8" in cpu["precisions"]
        assert cpu["device"] == "cpu"

    def test_cuda_target(self):
        """Test CUDA target configuration."""
        cuda = TARGET_PRECISION_MATRIX["cuda"]
        assert "fp16" in cuda["precisions"]
        assert "int4" in cuda["precisions"]
        assert cuda["device"] == "gpu"


class TestCategoryTargetCompatibility:
    """Tests for category to target compatibility."""

    def test_all_categories_have_targets(self):
        """Each category should have at least one target."""
        for category, targets in CATEGORY_TARGET_COMPATIBILITY.items():
            assert len(targets) > 0
            for target in targets:
                assert target in TARGET_PRECISION_MATRIX

    def test_nlp_has_common_targets(self):
        """NLP category should include common targets."""
        nlp_targets = CATEGORY_TARGET_COMPATIBILITY["nlp"]
        assert "cpu" in nlp_targets
        assert "cuda" in nlp_targets

    def test_vision_has_common_targets(self):
        """Vision category should include common targets."""
        vision_targets = CATEGORY_TARGET_COMPATIBILITY["vision"]
        assert "cpu" in vision_targets
        assert "cuda" in vision_targets


class TestTargetProviderMap:
    """Tests for target to provider mapping."""

    def test_all_targets_have_provider(self):
        """Each target should map to an execution provider."""
        for target in ALL_TARGETS:
            assert target in TARGET_PROVIDER_MAP

    def test_cpu_provider(self):
        """Test CPU provider mapping."""
        assert TARGET_PROVIDER_MAP["cpu"] == "CPUExecutionProvider"

    def test_cuda_provider(self):
        """Test CUDA provider mapping."""
        assert TARGET_PROVIDER_MAP["cuda"] == "CUDAExecutionProvider"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_provider_for_target(self):
        """Test get_provider_for_target function."""
        assert get_provider_for_target("cpu") == "CPUExecutionProvider"
        assert get_provider_for_target("cuda") == "CUDAExecutionProvider"
        assert get_provider_for_target("unknown") == "CPUExecutionProvider"

    def test_get_compatible_targets(self):
        """Test get_compatible_targets function."""
        nlp_targets = get_compatible_targets("nlp")
        assert "cpu" in nlp_targets

        unknown_targets = get_compatible_targets("unknown")
        assert "cpu" in unknown_targets

    def test_get_precisions_for_target(self):
        """Test get_precisions_for_target function."""
        cpu_precisions = get_precisions_for_target("cpu")
        assert "fp32" in cpu_precisions

        unknown_precisions = get_precisions_for_target("unknown")
        assert unknown_precisions == ["fp32"]

    def test_get_default_precision(self):
        """Test get_default_precision function."""
        assert get_default_precision("cpu") == "int8"
        assert get_default_precision("cuda") == "fp16"
        assert get_default_precision("unknown") == "fp32"

    def test_get_device_for_target(self):
        """Test get_device_for_target function."""
        assert get_device_for_target("cpu") == "cpu"
        assert get_device_for_target("cuda") == "gpu"
        assert get_device_for_target("qnn") == "npu"

    def test_validate_target(self):
        """Test validate_target function."""
        assert validate_target("cpu") is True
        assert validate_target("cuda") is True
        assert validate_target("unknown") is False

    def test_validate_precision(self):
        """Test validate_precision function."""
        assert validate_precision("fp32", "cpu") is True
        assert validate_precision("bf16", "cpu") is False
        assert validate_precision("bf16", "cuda") is True
        assert validate_precision("fp32", "unknown") is False
