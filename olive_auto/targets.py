"""Target and precision definitions for olive-auto."""
from __future__ import annotations

from typing import Dict, List

# Target to execution provider mapping
TARGET_PROVIDER_MAP = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "qnn": "QNNExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
    "webgpu": "WebGpuExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "directml": "DmlExecutionProvider",
    "rocm": "ROCMExecutionProvider",
}

# Target-precision compatibility matrix
TARGET_PRECISION_MATRIX: Dict[str, Dict] = {
    "cpu": {
        "precisions": ["fp32", "fp16", "int4"],
        "default": "fp16",
        "olive_extra": None,
        "system_requirements": [],
        "device": "cpu",
    },
    "cuda": {
        "precisions": ["fp32", "fp16", "bf16", "int8", "int4", "bnb4"],
        "default": "fp16",
        "olive_extra": "gpu",
        "system_requirements": ["CUDA Toolkit >= 11.8", "cuDNN >= 8.6"],
        "device": "gpu",
    },
    "qnn": {
        "precisions": ["fp16", "int8", "int4"],
        "default": "int8",
        "olive_extra": "qualcomm",
        "system_requirements": ["Qualcomm AI Engine Direct SDK >= 2.19"],
        "device": "npu",
    },
    "openvino": {
        "precisions": ["fp32", "fp16", "int8", "int4"],
        "default": "int8",
        "olive_extra": "openvino",
        "system_requirements": ["OpenVINO Toolkit >= 2024.0"],
        "device": "cpu",
    },
    "vitisai": {
        "precisions": ["fp32", "int8"],
        "default": "int8",
        "olive_extra": None,
        "system_requirements": ["Vitis AI >= 3.5"],
        "device": "npu",
    },
    "webgpu": {
        "precisions": ["fp32", "fp16", "int4"],
        "default": "fp16",
        "olive_extra": None,
        "system_requirements": [],
        "device": "gpu",
    },
    "tensorrt": {
        "precisions": ["fp32", "fp16", "int8"],
        "default": "fp16",
        "olive_extra": None,
        "system_requirements": ["TensorRT >= 8.6", "CUDA Toolkit >= 11.8"],
        "device": "gpu",
    },
    "directml": {
        "precisions": ["fp32", "fp16"],
        "default": "fp16",
        "olive_extra": "directml",
        "system_requirements": ["Windows 10/11", "DirectX 12 compatible GPU"],
        "device": "gpu",
    },
    "rocm": {
        "precisions": ["fp32", "fp16", "int8"],
        "default": "fp16",
        "olive_extra": None,
        "system_requirements": ["ROCm >= 5.7", "AMD GPU (gfx9 or later)"],
        "device": "gpu",
    },
}

# Category to compatible targets mapping
CATEGORY_TARGET_COMPATIBILITY: Dict[str, List[str]] = {
    "nlp": [
        "cpu",
        "cuda",
        "qnn",
        "openvino",
        "vitisai",
        "webgpu",
        "tensorrt",
        "directml",
        "rocm",
    ],
    "vision": [
        "cpu",
        "cuda",
        "qnn",
        "openvino",
        "vitisai",
        "tensorrt",
        "directml",
        "rocm",
    ],
    "audio": [
        "cpu",
        "cuda",
        "openvino",
        "tensorrt",
    ],
    "multimodal": [
        "cpu",
        "cuda",
        "openvino",
    ],
    "diffusers": [
        "cpu",
        "cuda",
        "openvino",
        "directml",
        "rocm",
    ],
    "unknown": [
        "cpu",
    ],
}

# All available targets
ALL_TARGETS = list(TARGET_PRECISION_MATRIX.keys())

# All available precisions
ALL_PRECISIONS = ["fp32", "fp16", "bf16", "int8", "int4", "bnb4"]


def get_provider_for_target(target: str) -> str:
    """Get the ONNX Runtime execution provider name for a target."""
    return TARGET_PROVIDER_MAP.get(target, "CPUExecutionProvider")


def get_compatible_targets(category: str) -> List[str]:
    """Get list of compatible targets for a model category."""
    return CATEGORY_TARGET_COMPATIBILITY.get(category, ["cpu"])


def get_precisions_for_target(target: str) -> List[str]:
    """Get list of supported precisions for a target."""
    if target in TARGET_PRECISION_MATRIX:
        return TARGET_PRECISION_MATRIX[target]["precisions"]
    return ["fp32"]


def get_default_precision(target: str) -> str:
    """Get the default precision for a target."""
    if target in TARGET_PRECISION_MATRIX:
        return TARGET_PRECISION_MATRIX[target]["default"]
    return "fp32"


def get_device_for_target(target: str) -> str:
    """Get the device type for a target."""
    if target in TARGET_PRECISION_MATRIX:
        return TARGET_PRECISION_MATRIX[target]["device"]
    return "cpu"


def validate_target(target: str) -> bool:
    """Check if a target is valid."""
    return target in TARGET_PRECISION_MATRIX


def validate_precision(precision: str, target: str) -> bool:
    """Check if a precision is valid for a given target."""
    if target not in TARGET_PRECISION_MATRIX:
        return False
    return precision in TARGET_PRECISION_MATRIX[target]["precisions"]
