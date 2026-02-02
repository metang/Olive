# Auto Olive: Unified Model Optimization Pipeline

## Executive Summary

This document proposes **Auto Olive** - a standalone CLI tool (`olive-auto`) that takes a HuggingFace model ID and generates a complete batch pipeline to convert and optimize the model for all supported targets and precisions, including environment setup and cleanup scripts.

```bash
# Single command to generate everything
olive-auto -m microsoft/phi-2 --output-dir ./optimization_pipeline
```

**Why a separate command?**
- Clean separation from the core `olive` CLI
- Independent versioning and updates
- Simpler installation for users who only need batch generation
- Avoids bloating the main olive command with complex orchestration logic

**Generated Output:**
```
optimization_pipeline/
├── setup_environments.sh        # Setup all required environments
├── run_all_optimizations.sh     # Master orchestration script
├── cleanup.sh                   # Cleanup environments and temp files
├── environments/
│   ├── setup_cpu.sh
│   ├── setup_cuda.sh
│   ├── setup_qnn.sh
│   ├── setup_openvino.sh
│   └── ...
├── commands/
│   ├── cpu_fp32.sh
│   ├── cpu_int8.sh
│   ├── cuda_fp16.sh
│   ├── cuda_int4.sh
│   ├── qnn_int8.sh
│   └── ...
├── outputs/                     # Optimized models (after execution)
│   ├── cpu_fp32/
│   ├── cpu_int8/
│   └── ...
└── logs/                        # Execution logs
    └── ...
```

---

## Part 1: Auto Model Detection

### 1.1 Problem Statement

Currently, Olive's `olive optimize` command:
- Defaults to `text-generation-with-past` task for all HuggingFace models
- Requires users to manually specify `--task` for non-LLM models
- Only OpenVINO passes use Optimum's TasksManager for auto-detection

### 1.2 Task Detection Module

**New Module:** `olive/common/hf/task_detection.py`

```python
"""Auto task detection for HuggingFace models."""
from __future__ import annotations

import logging
from typing import Optional, Tuple

from olive.common.constants import DEFAULT_HF_TASK
from olive.common.hf.model_io import get_hf_model_config

logger = logging.getLogger(__name__)

# Task to category mapping
TASK_CATEGORY_MAP = {
    # Text/NLP
    "text-generation": "nlp",
    "text-generation-with-past": "nlp",
    "text2text-generation": "nlp",
    "text2text-generation-with-past": "nlp",
    "text-classification": "nlp",
    "token-classification": "nlp",
    "question-answering": "nlp",
    "fill-mask": "nlp",
    "feature-extraction": "nlp",
    "zero-shot-classification": "nlp",

    # Vision
    "image-classification": "vision",
    "image-segmentation": "vision",
    "object-detection": "vision",
    "image-to-text": "vision",
    "zero-shot-image-classification": "vision",
    "zero-shot-object-detection": "vision",
    "depth-estimation": "vision",

    # Audio
    "automatic-speech-recognition": "audio",
    "audio-classification": "audio",
    "text-to-audio": "audio",
    "audio-to-audio": "audio",

    # Multimodal
    "image-text-to-text": "multimodal",
    "visual-question-answering": "multimodal",
    "document-question-answering": "multimodal",

    # Generative
    "stable-diffusion": "diffusers",
    "stable-diffusion-xl": "diffusers",
}


def infer_task_from_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
) -> Optional[str]:
    """
    Infer the task from a HuggingFace model.

    Uses multiple strategies:
    1. Optimum's TasksManager (most reliable)
    2. Model config's architectures field
    3. Model card/README metadata

    Returns:
        Inferred task string or None if detection fails
    """
    task = None

    # Strategy 1: Use Optimum's TasksManager
    task = _infer_via_tasks_manager(model_name_or_path)
    if task:
        return task

    # Strategy 2: Infer from model architecture name
    task = _infer_from_architecture(model_name_or_path, trust_remote_code)
    if task:
        return task

    # Strategy 3: Check model card metadata
    task = _infer_from_model_card(model_name_or_path)
    if task:
        return task

    return None


def _infer_via_tasks_manager(model_name_or_path: str) -> Optional[str]:
    """Use Optimum's TasksManager for task inference."""
    try:
        from optimum.exporters.tasks import TasksManager
        task = TasksManager.infer_task_from_model(model_name_or_path)
        if task in ("text-generation", "text2text-generation"):
            task = f"{task}-with-past"
        return task
    except Exception:
        return None


def _infer_from_architecture(
    model_name_or_path: str,
    trust_remote_code: bool = False
) -> Optional[str]:
    """Infer task from model architecture name pattern."""
    ARCHITECTURE_TASK_MAP = {
        "ForCausalLM": "text-generation-with-past",
        "ForSeq2SeqLM": "text2text-generation-with-past",
        "ForSequenceClassification": "text-classification",
        "ForTokenClassification": "token-classification",
        "ForQuestionAnswering": "question-answering",
        "ForMaskedLM": "fill-mask",
        "ForImageClassification": "image-classification",
        "ForObjectDetection": "object-detection",
        "ForSemanticSegmentation": "image-segmentation",
        "ForAudioClassification": "audio-classification",
        "ForCTC": "automatic-speech-recognition",
        "ForSpeechSeq2Seq": "automatic-speech-recognition",
        "ForVision2Seq": "image-to-text",
        "ForConditionalGeneration": "text2text-generation-with-past",
    }

    try:
        config = get_hf_model_config(model_name_or_path, trust_remote_code)
        architectures = getattr(config, "architectures", []) or []
        for arch in architectures:
            for suffix, task in ARCHITECTURE_TASK_MAP.items():
                if arch.endswith(suffix):
                    return task
        return None
    except Exception:
        return None


def _infer_from_model_card(model_name_or_path: str) -> Optional[str]:
    """Infer task from HuggingFace model card metadata."""
    try:
        from huggingface_hub import model_info
        info = model_info(model_name_or_path)
        if info.pipeline_tag:
            task = info.pipeline_tag
            if task in ("text-generation", "text2text-generation"):
                task = f"{task}-with-past"
            return task
        return None
    except Exception:
        return None


def get_model_category(task: str) -> str:
    """Get the model category from task."""
    return TASK_CATEGORY_MAP.get(task, "unknown")


def detect_model_info(
    model_name_or_path: str,
    user_task: Optional[str] = None,
    trust_remote_code: bool = False,
) -> Tuple[str, str]:
    """
    Detect model task and category.

    Returns:
        Tuple of (task, category)
    """
    if user_task:
        task = user_task
    else:
        task = infer_task_from_model(model_name_or_path, trust_remote_code)
        if task is None:
            task = DEFAULT_HF_TASK

    category = get_model_category(task)
    return task, category
```

---

## Part 2: Target and Precision Matrix

### 2.1 Execution Providers (Targets)

| Target | Description | Platform | SDK Required |
|--------|-------------|----------|--------------|
| `cpu` | CPU inference via ONNX Runtime | All | None |
| `cuda` | NVIDIA GPU via CUDA | Linux/Windows | CUDA Toolkit |
| `qnn` | Qualcomm NPU | Windows/Android | Qualcomm AI Engine Direct SDK |
| `openvino` | Intel CPU/GPU/VPU | Linux/Windows | OpenVINO Toolkit |
| `vitisai` | AMD/Xilinx FPGA/NPU | Linux | Vitis AI SDK |
| `webgpu` | Browser WebGPU | Web | None (browser) |
| `tensorrt` | NVIDIA TensorRT | Linux/Windows | TensorRT SDK |
| `directml` | Windows DirectML | Windows | DirectX 12 |
| `rocm` | AMD GPU via ROCm | Linux | ROCm SDK |

### 2.2 Precision Options

| Precision | Description | Size Reduction | Typical Use |
|-----------|-------------|----------------|-------------|
| `fp32` | 32-bit floating point | 1x (baseline) | Maximum accuracy |
| `fp16` | 16-bit floating point | 2x | GPU inference |
| `bf16` | Brain floating point 16 | 2x | Training/inference |
| `int8` | 8-bit integer | 4x | CPU/edge deployment |
| `int4` | 4-bit integer | 8x | LLM compression |
| `bnb4` | bitsandbytes 4-bit | 8x | LLM with GPU |

### 2.3 Target-Precision Compatibility Matrix

```python
TARGET_PRECISION_MATRIX = {
    "cpu": {
        "precisions": ["fp32", "fp16", "int8", "int4"],
        "default": "int8",
        "olive_extra": None,
        "system_requirements": [],
    },
    "cuda": {
        "precisions": ["fp32", "fp16", "bf16", "int8", "int4", "bnb4"],
        "default": "fp16",
        "olive_extra": "gpu",
        "system_requirements": ["CUDA Toolkit >= 11.8", "cuDNN >= 8.6"],
    },
    "qnn": {
        "precisions": ["fp16", "int8", "int4"],
        "default": "int8",
        "olive_extra": "qualcomm",
        "system_requirements": ["Qualcomm AI Engine Direct SDK >= 2.19"],
    },
    "openvino": {
        "precisions": ["fp32", "fp16", "int8", "int4"],
        "default": "int8",
        "olive_extra": "openvino",
        "system_requirements": ["OpenVINO Toolkit >= 2024.0"],
    },
    "vitisai": {
        "precisions": ["fp32", "int8"],
        "default": "int8",
        "olive_extra": None,
        "system_requirements": ["Vitis AI >= 3.5"],
    },
    "webgpu": {
        "precisions": ["fp32", "fp16", "int4"],
        "default": "fp16",
        "olive_extra": None,
        "system_requirements": [],
    },
    "tensorrt": {
        "precisions": ["fp32", "fp16", "int8"],
        "default": "fp16",
        "olive_extra": None,
        "system_requirements": ["TensorRT >= 8.6", "CUDA Toolkit >= 11.8"],
    },
    "directml": {
        "precisions": ["fp32", "fp16"],
        "default": "fp16",
        "olive_extra": "directml",
        "system_requirements": ["Windows 10/11", "DirectX 12 GPU"],
    },
    "rocm": {
        "precisions": ["fp32", "fp16", "int8"],
        "default": "fp16",
        "olive_extra": None,
        "system_requirements": ["ROCm >= 5.7", "AMD GPU (gfx9+)"],
    },
}
```

### 2.4 Category-Target Compatibility

Not all model categories work with all targets:

```python
CATEGORY_TARGET_COMPATIBILITY = {
    "nlp": ["cpu", "cuda", "qnn", "openvino", "vitisai", "webgpu", "tensorrt", "directml", "rocm"],
    "vision": ["cpu", "cuda", "qnn", "openvino", "vitisai", "tensorrt", "directml", "rocm"],
    "audio": ["cpu", "cuda", "openvino", "tensorrt"],
    "multimodal": ["cpu", "cuda", "openvino"],
    "diffusers": ["cpu", "cuda", "openvino", "directml", "rocm"],
}
```

---

## Part 3: Environment Management

### 3.1 Environment Configuration

**New Module:** `olive/auto/environment.py`

```python
"""Environment setup and management for Auto Olive."""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class EnvironmentType(Enum):
    VENV = "venv"
    CONDA = "conda"


@dataclass
class EnvironmentConfig:
    """Configuration for a target-specific environment."""
    target: str
    python_version: str = "3.10"
    olive_extras: List[str] = None
    pip_packages: List[str] = None
    conda_packages: List[str] = None
    system_requirements: List[str] = None
    env_vars: Dict[str, str] = None
    pre_install_script: Optional[str] = None
    post_install_script: Optional[str] = None


# Environment configurations per target
ENVIRONMENT_CONFIGS = {
    "cpu": EnvironmentConfig(
        target="cpu",
        olive_extras=[],
        pip_packages=[
            "olive-ai",
            "onnxruntime",
            "transformers",
            "torch",
        ],
    ),

    "cuda": EnvironmentConfig(
        target="cuda",
        olive_extras=["gpu"],
        pip_packages=[
            "olive-ai[gpu]",
            "onnxruntime-gpu",
            "transformers",
            "torch",
            "bitsandbytes",
        ],
        env_vars={
            "CUDA_HOME": "/usr/local/cuda",
            "LD_LIBRARY_PATH": "${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}",
        },
        system_requirements=[
            "CUDA Toolkit >= 11.8",
            "cuDNN >= 8.6",
        ],
    ),

    "qnn": EnvironmentConfig(
        target="qnn",
        olive_extras=["qualcomm"],
        pip_packages=[
            "olive-ai[qualcomm]",
            "onnxruntime-qnn",
            "transformers",
        ],
        env_vars={
            "QNN_SDK_ROOT": "/opt/qnn-sdk",
            "PATH": "${QNN_SDK_ROOT}/bin:${PATH}",
            "LD_LIBRARY_PATH": "${QNN_SDK_ROOT}/lib:${LD_LIBRARY_PATH}",
        },
        system_requirements=[
            "Qualcomm AI Engine Direct SDK >= 2.19",
        ],
        pre_install_script="""
# Verify QNN SDK installation
if [ -z "$QNN_SDK_ROOT" ]; then
    echo "ERROR: QNN_SDK_ROOT not set. Please install Qualcomm AI Engine Direct SDK."
    exit 1
fi
""",
    ),

    "openvino": EnvironmentConfig(
        target="openvino",
        olive_extras=["openvino"],
        pip_packages=[
            "olive-ai[openvino]",
            "openvino",
            "openvino-dev",
            "optimum[openvino]",
            "transformers",
        ],
        env_vars={
            "OPENVINO_HOME": "/opt/intel/openvino",
        },
        system_requirements=[
            "OpenVINO Toolkit >= 2024.0",
        ],
    ),

    "vitisai": EnvironmentConfig(
        target="vitisai",
        olive_extras=[],
        pip_packages=[
            "olive-ai",
            "vai-q-onnx",
            "transformers",
        ],
        conda_packages=[
            "vitis-ai-runtime",
        ],
        env_vars={
            "VITIS_AI_HOME": "/opt/vitis-ai",
            "XLNX_VART_FIRMWARE": "/opt/vitis-ai/firmware",
        },
        system_requirements=[
            "Vitis AI >= 3.5",
            "XRT (Xilinx Runtime)",
        ],
    ),

    "webgpu": EnvironmentConfig(
        target="webgpu",
        olive_extras=[],
        pip_packages=[
            "olive-ai",
            "onnxruntime-web",
            "transformers",
        ],
    ),

    "tensorrt": EnvironmentConfig(
        target="tensorrt",
        olive_extras=[],
        pip_packages=[
            "olive-ai",
            "onnxruntime-gpu",
            "tensorrt",
            "transformers",
            "torch",
        ],
        env_vars={
            "CUDA_HOME": "/usr/local/cuda",
            "TENSORRT_HOME": "/usr/local/tensorrt",
            "LD_LIBRARY_PATH": "${TENSORRT_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}",
        },
        system_requirements=[
            "TensorRT >= 8.6",
            "CUDA Toolkit >= 11.8",
        ],
    ),

    "directml": EnvironmentConfig(
        target="directml",
        olive_extras=["directml"],
        pip_packages=[
            "olive-ai[directml]",
            "onnxruntime-directml",
            "torch-directml",
            "transformers",
        ],
        system_requirements=[
            "Windows 10/11",
            "DirectX 12 compatible GPU",
        ],
    ),

    "rocm": EnvironmentConfig(
        target="rocm",
        olive_extras=[],
        pip_packages=[
            "olive-ai",
            "torch-rocm",
            "transformers",
        ],
        env_vars={
            "ROCM_HOME": "/opt/rocm",
            "HIP_PATH": "/opt/rocm/hip",
            "PATH": "${ROCM_HOME}/bin:${PATH}",
            "LD_LIBRARY_PATH": "${ROCM_HOME}/lib:${LD_LIBRARY_PATH}",
        },
        system_requirements=[
            "ROCm >= 5.7",
            "AMD GPU (gfx9 or later)",
        ],
    ),
}
```

### 3.2 Environment Script Generator

```python
"""Generate environment setup scripts."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

from olive.auto.environment import ENVIRONMENT_CONFIGS, EnvironmentConfig


class EnvironmentScriptGenerator:
    """Generate setup/cleanup scripts for target environments."""

    def __init__(self, output_dir: Path, env_type: str = "venv"):
        self.output_dir = output_dir
        self.env_type = env_type
        self.env_dir = output_dir / "environments"
        self.env_dir.mkdir(parents=True, exist_ok=True)

    def generate_setup_script(self, target: str) -> Path:
        """Generate environment setup script for a target."""
        config = ENVIRONMENT_CONFIGS[target]
        script_path = self.env_dir / f"setup_{target}.sh"

        script_content = self._generate_bash_setup(config)
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        return script_path

    def _generate_bash_setup(self, config: EnvironmentConfig) -> str:
        """Generate bash setup script."""
        lines = [
            "#!/bin/bash",
            f"# Environment setup for {config.target}",
            "set -e",
            "",
            f"ENV_NAME=\"olive_{config.target}\"",
            f"ENV_DIR=\"$(dirname $0)/$ENV_NAME\"",
            "",
            "echo \"========================================\"",
            f"echo \"Setting up environment for: {config.target}\"",
            "echo \"========================================\"",
            "",
        ]

        # System requirements check
        if config.system_requirements:
            lines.extend([
                "# Check system requirements",
                "echo \"Checking system requirements...\"",
            ])
            for req in config.system_requirements:
                lines.append(f"echo \"  - Required: {req}\"")
            lines.append("")

        # Pre-install script
        if config.pre_install_script:
            lines.extend([
                "# Pre-install checks",
                config.pre_install_script,
                "",
            ])

        # Create virtual environment
        lines.extend([
            "# Create virtual environment",
            "if [ ! -d \"$ENV_DIR\" ]; then",
            f"    python{config.python_version} -m venv \"$ENV_DIR\"",
            "fi",
            "",
            "# Activate environment",
            "source \"$ENV_DIR/bin/activate\"",
            "",
            "# Upgrade pip",
            "pip install --upgrade pip",
            "",
        ])

        # Install packages
        if config.pip_packages:
            lines.extend([
                "# Install pip packages",
            ])
            for pkg in config.pip_packages:
                lines.append(f"pip install \"{pkg}\"")
            lines.append("")

        # Set environment variables
        if config.env_vars:
            lines.extend([
                "# Set environment variables",
                "cat >> \"$ENV_DIR/bin/activate\" << 'ENVVARS'",
            ])
            for key, value in config.env_vars.items():
                lines.append(f"export {key}=\"{value}\"")
            lines.extend([
                "ENVVARS",
                "",
            ])

        # Post-install script
        if config.post_install_script:
            lines.extend([
                "# Post-install setup",
                config.post_install_script,
                "",
            ])

        lines.extend([
            "echo \"\"",
            f"echo \"Environment '{config.target}' setup complete!\"",
            "echo \"Activate with: source $ENV_DIR/bin/activate\"",
        ])

        return "\n".join(lines)

    def generate_windows_setup(self, config: EnvironmentConfig) -> str:
        """Generate PowerShell setup script for Windows."""
        lines = [
            "# Environment setup for " + config.target,
            "$ErrorActionPreference = 'Stop'",
            "",
            f"$ENV_NAME = 'olive_{config.target}'",
            "$ENV_DIR = Join-Path $PSScriptRoot $ENV_NAME",
            "",
            "Write-Host '========================================'",
            f"Write-Host 'Setting up environment for: {config.target}'",
            "Write-Host '========================================'",
            "",
        ]

        # Create virtual environment
        lines.extend([
            "# Create virtual environment",
            "if (-not (Test-Path $ENV_DIR)) {",
            f"    python -m venv $ENV_DIR",
            "}",
            "",
            "# Activate environment",
            "& \"$ENV_DIR\\Scripts\\Activate.ps1\"",
            "",
            "# Upgrade pip",
            "pip install --upgrade pip",
            "",
        ])

        # Install packages
        if config.pip_packages:
            lines.append("# Install pip packages")
            for pkg in config.pip_packages:
                lines.append(f"pip install '{pkg}'")
            lines.append("")

        # Set environment variables
        if config.env_vars:
            lines.append("# Set environment variables")
            for key, value in config.env_vars.items():
                # Convert bash-style variable expansion to PowerShell
                ps_value = value.replace("${", "$env:").replace("}", "")
                lines.append(f"$env:{key} = \"{ps_value}\"")
            lines.append("")

        lines.extend([
            "Write-Host ''",
            f"Write-Host 'Environment {config.target} setup complete!'",
        ])

        return "\n".join(lines)

    def generate_cleanup_script(self, targets: List[str]) -> Path:
        """Generate cleanup script for all environments."""
        script_path = self.output_dir / "cleanup.sh"

        lines = [
            "#!/bin/bash",
            "# Cleanup script for Auto Olive environments",
            "set -e",
            "",
            "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"",
            "",
            "echo \"Cleaning up Auto Olive environments...\"",
            "",
        ]

        for target in targets:
            lines.extend([
                f"# Cleanup {target} environment",
                f"if [ -d \"$SCRIPT_DIR/environments/olive_{target}\" ]; then",
                f"    echo \"Removing olive_{target} environment...\"",
                f"    rm -rf \"$SCRIPT_DIR/environments/olive_{target}\"",
                "fi",
                "",
            ])

        lines.extend([
            "# Cleanup temporary files",
            "if [ -d \"$SCRIPT_DIR/temp\" ]; then",
            "    echo \"Removing temporary files...\"",
            "    rm -rf \"$SCRIPT_DIR/temp\"",
            "fi",
            "",
            "# Optional: Cleanup output models",
            "read -p \"Remove output models? (y/N) \" -n 1 -r",
            "echo",
            "if [[ $REPLY =~ ^[Yy]$ ]]; then",
            "    rm -rf \"$SCRIPT_DIR/outputs\"",
            "    echo \"Output models removed.\"",
            "fi",
            "",
            "echo \"Cleanup complete!\"",
        ])

        script_path.write_text("\n".join(lines))
        script_path.chmod(0o755)

        return script_path
```

---

## Part 4: Command Generation

### 4.1 Optimization Command Generator

**New Module:** `olive/auto/command_generator.py`

```python
"""Generate olive optimize commands for all target/precision combinations."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from olive.auto.environment import ENVIRONMENT_CONFIGS
from olive.auto.targets import TARGET_PRECISION_MATRIX, CATEGORY_TARGET_COMPATIBILITY


@dataclass
class OptimizationCommand:
    """Represents a single optimization command."""
    target: str
    precision: str
    model_id: str
    task: str
    output_dir: Path
    extra_args: Dict[str, str] = None

    def to_olive_command(self) -> str:
        """Generate the olive optimize command string."""
        cmd_parts = [
            "olive optimize",
            f"-m \"{self.model_id}\"",
            f"--task {self.task}",
            f"--provider {self._get_provider()}",
            f"--precision {self.precision}",
            f"-o \"{self.output_dir}\"",
        ]

        # Add target-specific flags
        if self.target == "cuda":
            cmd_parts.append("--device gpu")
        elif self.target == "qnn":
            cmd_parts.append("--device npu")

        # Add extra arguments
        if self.extra_args:
            for key, value in self.extra_args.items():
                cmd_parts.append(f"--{key} {value}")

        return " \\\n    ".join(cmd_parts)

    def _get_provider(self) -> str:
        """Map target to execution provider name."""
        provider_map = {
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
        return provider_map.get(self.target, "CPUExecutionProvider")


class CommandGenerator:
    """Generate optimization commands and scripts with Jinja2 templates."""

    def __init__(
        self,
        model_id: str,
        task: str,
        category: str,
        output_dir: Path,
        targets: Optional[List[str]] = None,
        precisions: Optional[Dict[str, List[str]]] = None,
        platform: str = "linux",
    ):
        self.model_id = model_id
        self.task = task
        self.category = category
        self.output_dir = output_dir
        self.platform = platform
        self.commands_dir = output_dir / "commands"
        self.commands_dir.mkdir(parents=True, exist_ok=True)

        # Determine compatible targets
        self.targets = targets or self._get_compatible_targets()

        # Determine precisions per target
        self.precisions = precisions or self._get_default_precisions()

        # Setup Jinja2 environment
        self._setup_templates()

    def _setup_templates(self):
        """Setup Jinja2 template environment."""
        from jinja2 import Environment, PackageLoader, select_autoescape

        template_subdir = "powershell" if self.platform == "windows" else "bash"
        self.env = Environment(
            loader=PackageLoader("olive_auto", f"templates/{template_subdir}"),
            autoescape=select_autoescape(),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def _get_compatible_targets(self) -> List[str]:
        """Get targets compatible with the model category."""
        return CATEGORY_TARGET_COMPATIBILITY.get(self.category, ["cpu"])

    def _get_default_precisions(self) -> Dict[str, List[str]]:
        """Get default precisions for each target."""
        precisions = {}
        for target in self.targets:
            if target in TARGET_PRECISION_MATRIX:
                precisions[target] = TARGET_PRECISION_MATRIX[target]["precisions"]
        return precisions

    def generate_all_commands(self) -> List[OptimizationCommand]:
        """Generate all optimization commands."""
        commands = []

        for target in self.targets:
            for precision in self.precisions.get(target, ["fp32"]):
                output_path = self.output_dir / "outputs" / f"{target}_{precision}"

                cmd = OptimizationCommand(
                    target=target,
                    precision=precision,
                    model_id=self.model_id,
                    task=self.task,
                    output_dir=output_path,
                )
                commands.append(cmd)

        return commands

    def generate_command_scripts(self) -> Dict[str, Path]:
        """Generate individual command scripts using templates."""
        commands = self.generate_all_commands()
        scripts = {}
        ext = ".ps1" if self.platform == "windows" else ".sh"

        template = self.env.get_template(f"run_command{ext}.jinja2")

        for cmd in commands:
            script_name = f"{cmd.target}_{cmd.precision}{ext}"
            script_path = self.commands_dir / script_name

            script_content = template.render(
                target=cmd.target,
                precision=cmd.precision,
                model_id=cmd.model_id,
                task=cmd.task,
                output_dir=cmd.output_dir,
                olive_command=cmd.to_olive_command(),
                provider=cmd._get_provider(),
            )

            script_path.write_text(script_content)
            if self.platform != "windows":
                script_path.chmod(0o755)

            scripts[f"{cmd.target}_{cmd.precision}"] = script_path

        return scripts

    def generate_master_script(self) -> Path:
        """Generate master orchestration script."""
        ext = ".ps1" if self.platform == "windows" else ".sh"
        script_path = self.output_dir / f"run_all_optimizations{ext}"
        commands = self.generate_all_commands()

        template = self.env.get_template(f"master_run{ext}.jinja2")

        script_content = template.render(
            model_id=self.model_id,
            task=self.task,
            category=self.category,
            commands=[
                {
                    "target": cmd.target,
                    "precision": cmd.precision,
                    "script_name": f"{cmd.target}_{cmd.precision}{ext}",
                }
                for cmd in commands
            ],
            total_commands=len(commands),
        )

        script_path.write_text(script_content)
        if self.platform != "windows":
            script_path.chmod(0o755)

        return script_path
```

### 4.2 Jinja2 Templates

**File:** `olive_auto/templates/bash/run_command.sh.jinja2`

```jinja2
#!/bin/bash
# Optimization: {{ target }} with {{ precision }} precision
# Model: {{ model_id }}
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/../logs/{{ target }}_{{ precision }}.log"

echo "========================================"
echo "Target: {{ target }}"
echo "Precision: {{ precision }}"
echo "Model: {{ model_id }}"
echo "========================================"

# Activate environment
source "$SCRIPT_DIR/../environments/olive_{{ target }}/bin/activate"

# Create output directory
mkdir -p "{{ output_dir }}"

echo "Starting optimization..."
echo "Log file: $LOG_FILE"

# Run optimization
{{ olive_command }} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Optimization complete: {{ target }} {{ precision }}"
```

**File:** `olive_auto/templates/bash/master_run.sh.jinja2`

```jinja2
#!/bin/bash
# Master orchestration script for olive-auto
# Model: {{ model_id }}
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

echo "========================================"
echo "olive-auto Pipeline"
echo "========================================"
echo "Model: {{ model_id }}"
echo "Task: {{ task }}"
echo "Category: {{ category }}"
echo "Total combinations: {{ total_commands }}"
echo "========================================"

# Track results
PASSED=0
FAILED=0

{% for cmd in commands %}
# {{ cmd.target }} with {{ cmd.precision }}
echo ""
echo "[$(date)] Running: {{ cmd.target }} {{ cmd.precision }}"
if "$SCRIPT_DIR/commands/{{ cmd.script_name }}"; then
    echo "[PASS] {{ cmd.target }} {{ cmd.precision }}"
    ((PASSED++))
else
    echo "[FAIL] {{ cmd.target }} {{ cmd.precision }}"
    ((FAILED++))
fi

{% endfor %}
# Print summary
echo ""
echo "========================================"
echo "Optimization Summary"
echo "========================================"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

if [ $FAILED -gt 0 ]; then
    exit 1
fi
```

**File:** `olive_auto/templates/powershell/run_command.ps1.jinja2`

```jinja2
# Optimization: {{ target }} with {{ precision }} precision
# Model: {{ model_id }}
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogFile = Join-Path $ScriptDir "..\logs\{{ target }}_{{ precision }}.log"

Write-Host "========================================"
Write-Host "Target: {{ target }}"
Write-Host "Precision: {{ precision }}"
Write-Host "Model: {{ model_id }}"
Write-Host "========================================"

# Activate environment
& "$ScriptDir\..\environments\olive_{{ target }}\Scripts\Activate.ps1"

# Create output directory
New-Item -ItemType Directory -Force -Path "{{ output_dir }}" | Out-Null

Write-Host "Starting optimization..."
Write-Host "Log file: $LogFile"

# Run optimization
{{ olive_command }} 2>&1 | Tee-Object -FilePath $LogFile

Write-Host ""
Write-Host "Optimization complete: {{ target }} {{ precision }}"
```

**File:** `olive_auto/templates/powershell/master_run.ps1.jinja2`

```jinja2
# Master orchestration script for olive-auto
# Model: {{ model_id }}
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Create logs directory
New-Item -ItemType Directory -Force -Path "$ScriptDir\logs" | Out-Null

Write-Host "========================================"
Write-Host "olive-auto Pipeline"
Write-Host "========================================"
Write-Host "Model: {{ model_id }}"
Write-Host "Task: {{ task }}"
Write-Host "Category: {{ category }}"
Write-Host "Total combinations: {{ total_commands }}"
Write-Host "========================================"

# Track results
$Passed = 0
$Failed = 0

{% for cmd in commands %}
# {{ cmd.target }} with {{ cmd.precision }}
Write-Host ""
Write-Host "[$(Get-Date)] Running: {{ cmd.target }} {{ cmd.precision }}"
try {
    & "$ScriptDir\commands\{{ cmd.script_name }}"
    Write-Host "[PASS] {{ cmd.target }} {{ cmd.precision }}"
    $Passed++
} catch {
    Write-Host "[FAIL] {{ cmd.target }} {{ cmd.precision }}"
    $Failed++
}

{% endfor %}
# Print summary
Write-Host ""
Write-Host "========================================"
Write-Host "Optimization Summary"
Write-Host "========================================"
Write-Host "Passed: $Passed"
Write-Host "Failed: $Failed"
Write-Host ""

if ($Failed -gt 0) {
    exit 1
}
```

---

## Part 5: Standalone CLI - `olive-auto`

### 5.1 Entry Point and Package Structure

**New Package:** `olive_auto/` (standalone package)

```
olive_auto/
├── __init__.py
├── __main__.py          # Entry point for `python -m olive_auto`
├── cli.py               # Main CLI implementation
├── task_detection.py    # Model task/category detection
├── targets.py           # Target/precision matrix
├── environment.py       # Environment configurations
├── command_generator.py # Command script generation
├── executor.py          # Pipeline execution
└── templates/           # Script templates
    ├── bash/
    │   ├── setup_env.sh.jinja2
    │   ├── run_command.sh.jinja2
    │   └── cleanup.sh.jinja2
    └── powershell/
        ├── setup_env.ps1.jinja2
        ├── run_command.ps1.jinja2
        └── cleanup.ps1.jinja2
```

### 5.2 Package Configuration

**File:** `pyproject.toml` (for olive-auto package)

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "olive-auto"
version = "0.1.0"
description = "Batch optimization pipeline generator for Olive"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
dependencies = [
    "huggingface_hub>=0.20.0",
    "jinja2>=3.0.0",
    "pyyaml>=6.0",
    "rich>=13.0.0",      # For beautiful CLI output
    "typer>=0.9.0",      # Modern CLI framework
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]
# Optional: include olive for direct execution
full = [
    "olive-ai>=0.7.0",
    "optimum>=1.17.0",
]

[project.scripts]
olive-auto = "olive_auto.cli:app"

[project.urls]
Homepage = "https://github.com/microsoft/Olive"
Documentation = "https://microsoft.github.io/Olive/"
Repository = "https://github.com/microsoft/Olive"
```

### 5.3 Main CLI Implementation

**File:** `olive_auto/cli.py`

```python
"""
olive-auto: Standalone CLI for batch model optimization pipeline generation.

Usage:
    olive-auto -m microsoft/phi-2 -o ./pipeline
    olive-auto -m openai/whisper-tiny --targets cpu cuda --platform both
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from olive_auto.task_detection import detect_model_info
from olive_auto.targets import TARGET_PRECISION_MATRIX, CATEGORY_TARGET_COMPATIBILITY
from olive_auto.environment import EnvironmentScriptGenerator
from olive_auto.command_generator import CommandGenerator
from olive_auto.executor import PipelineExecutor

# Initialize CLI app
app = typer.Typer(
    name="olive-auto",
    help="Generate optimization pipelines for HuggingFace models across all targets and precisions.",
    add_completion=False,
)

console = Console()
logger = logging.getLogger(__name__)


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        from olive_auto import __version__
        console.print(f"olive-auto version {__version__}")
        raise typer.Exit()


@app.command()
def main(
    # Model specification
    model: str = typer.Option(
        ...,
        "-m", "--model",
        help="HuggingFace model ID or local path",
    ),
    task: Optional[str] = typer.Option(
        None,
        "--task",
        help="Model task (auto-detected if not specified)",
    ),
    trust_remote_code: bool = typer.Option(
        False,
        "--trust-remote-code",
        help="Trust remote code from HuggingFace",
    ),

    # Target/precision selection
    targets: Optional[List[str]] = typer.Option(
        None,
        "--targets", "-t",
        help="Specific targets (default: all compatible). Options: cpu, cuda, qnn, openvino, tensorrt, directml, webgpu, vitisai, rocm",
    ),
    precisions: Optional[List[str]] = typer.Option(
        None,
        "--precisions", "-p",
        help="Specific precisions (default: all per target). Options: fp32, fp16, bf16, int8, int4, bnb4",
    ),
    exclude_targets: Optional[List[str]] = typer.Option(
        None,
        "--exclude-targets",
        help="Targets to exclude from generation",
    ),

    # Output configuration
    output_dir: Path = typer.Option(
        Path("./olive_auto_pipeline"),
        "-o", "--output-dir",
        help="Output directory for generated scripts",
    ),
    platform: str = typer.Option(
        "auto",
        "--platform",
        help="Target platform: linux, windows, or auto (detect current)",
    ),

    # Execution options
    parallel: int = typer.Option(
        1,
        "--parallel",
        help="Number of parallel optimizations (for --run)",
    ),
    run: bool = typer.Option(
        False,
        "--run",
        help="Execute pipeline after generation",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be generated without creating files",
    ),

    # Misc options
    verbose: bool = typer.Option(
        False,
        "-v", "--verbose",
        help="Enable verbose output",
    ),
    version: bool = typer.Option(
        False,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
):
    """
    Generate optimization pipeline for a HuggingFace model.

    This command analyzes a model, determines compatible targets and precisions,
    and generates a complete batch pipeline with environment setup and cleanup scripts.

    Examples:

        # Generate pipeline for all targets
        olive-auto -m microsoft/phi-2 -o ./phi2_pipeline

        # Generate for specific targets only
        olive-auto -m openai/whisper-tiny --targets cpu cuda

        # Generate and run immediately
        olive-auto -m bert-base-uncased --targets cpu --run

        # Generate for Windows
        olive-auto -m microsoft/phi-2 --platform windows
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")

    # Detect platform if auto
    if platform == "auto":
        import os
        platform = "windows" if os.name == "nt" else "linux"

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        # Step 1: Analyze model
        task_id = progress.add_task("Analyzing model...", total=None)
        detected_task, category = detect_model_info(
            model,
            user_task=task,
            trust_remote_code=trust_remote_code,
        )
        progress.update(task_id, completed=True)

        # Step 2: Determine targets
        progress.add_task("Selecting targets...", total=None)
        selected_targets = _get_targets(category, targets, exclude_targets)

        if not selected_targets:
            console.print("[red]Error: No compatible targets found for this model category.[/red]")
            raise typer.Exit(1)

        # Step 3: Determine precisions
        target_precisions = _get_precisions(selected_targets, precisions)

        # Count total commands
        total_commands = sum(len(p) for p in target_precisions.values())

    # Display analysis results
    _display_analysis(model, detected_task, category, selected_targets, target_precisions)

    if dry_run:
        console.print("\n[yellow]Dry run mode - no files created.[/yellow]")
        raise typer.Exit(0)

    # Step 4: Generate pipeline
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Create output directory
        task_id = progress.add_task("Creating output directory...", total=None)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "logs").mkdir(exist_ok=True)
        (output_dir / "outputs").mkdir(exist_ok=True)
        (output_dir / "environments").mkdir(exist_ok=True)
        (output_dir / "commands").mkdir(exist_ok=True)
        progress.update(task_id, completed=True)

        # Generate environment scripts
        task_id = progress.add_task("Generating environment scripts...", total=None)
        env_generator = EnvironmentScriptGenerator(output_dir, platform)
        for target in selected_targets:
            env_generator.generate_setup_script(target)
        env_generator.generate_master_setup_script(selected_targets)
        progress.update(task_id, completed=True)

        # Generate command scripts
        task_id = progress.add_task("Generating optimization commands...", total=None)
        cmd_generator = CommandGenerator(
            model_id=model,
            task=detected_task,
            category=category,
            output_dir=output_dir,
            targets=selected_targets,
            precisions=target_precisions,
            platform=platform,
        )
        cmd_generator.generate_command_scripts()
        cmd_generator.generate_master_script()
        progress.update(task_id, completed=True)

        # Generate cleanup script
        task_id = progress.add_task("Generating cleanup script...", total=None)
        env_generator.generate_cleanup_script(selected_targets)
        progress.update(task_id, completed=True)

        # Generate manifest
        task_id = progress.add_task("Generating manifest...", total=None)
        _generate_manifest(
            output_dir, model, detected_task, category,
            selected_targets, target_precisions
        )
        progress.update(task_id, completed=True)

    # Display summary
    _display_summary(output_dir, selected_targets, total_commands, platform)

    # Execute if requested
    if run:
        console.print("\n[bold]Executing pipeline...[/bold]")
        executor = PipelineExecutor(output_dir, parallel=parallel)
        success = executor.run()
        if not success:
            raise typer.Exit(1)


def _get_targets(
    category: str,
    user_targets: Optional[List[str]],
    exclude_targets: Optional[List[str]],
) -> List[str]:
    """Determine which targets to use based on category and user preferences."""
    compatible = CATEGORY_TARGET_COMPATIBILITY.get(category, ["cpu"])

    if user_targets:
        # Validate user targets
        invalid = [t for t in user_targets if t not in TARGET_PRECISION_MATRIX]
        if invalid:
            console.print(f"[yellow]Warning: Unknown targets ignored: {invalid}[/yellow]")
        targets = [t for t in user_targets if t in compatible]
    else:
        targets = list(compatible)

    if exclude_targets:
        targets = [t for t in targets if t not in exclude_targets]

    return targets


def _get_precisions(
    targets: List[str],
    user_precisions: Optional[List[str]],
) -> dict:
    """Determine precisions for each target."""
    result = {}
    for target in targets:
        available = TARGET_PRECISION_MATRIX[target]["precisions"]
        if user_precisions:
            result[target] = [p for p in user_precisions if p in available]
            if not result[target]:
                result[target] = available  # Fallback to all if none match
        else:
            result[target] = available
    return result


def _display_analysis(
    model: str,
    task: str,
    category: str,
    targets: List[str],
    precisions: dict,
):
    """Display model analysis results."""
    console.print()
    console.print(Panel.fit(
        f"[bold]Model:[/bold] {model}\n"
        f"[bold]Task:[/bold] {task}\n"
        f"[bold]Category:[/bold] {category}",
        title="Model Analysis",
        border_style="blue",
    ))

    # Create targets table
    table = Table(title="Target/Precision Matrix", show_header=True, header_style="bold")
    table.add_column("Target", style="cyan")
    table.add_column("Precisions", style="green")
    table.add_column("Commands", justify="right")

    for target in targets:
        precs = precisions[target]
        table.add_row(target, ", ".join(precs), str(len(precs)))

    console.print(table)


def _display_summary(
    output_dir: Path,
    targets: List[str],
    total_commands: int,
    platform: str,
):
    """Display generation summary."""
    ext = ".ps1" if platform == "windows" else ".sh"

    console.print()
    console.print(Panel.fit(
        f"[bold green]Pipeline generated successfully![/bold green]\n\n"
        f"[bold]Output:[/bold] {output_dir.absolute()}\n"
        f"[bold]Targets:[/bold] {len(targets)}\n"
        f"[bold]Total commands:[/bold] {total_commands}\n\n"
        f"[bold]Generated files:[/bold]\n"
        f"  - setup_environments{ext}\n"
        f"  - run_all_optimizations{ext}\n"
        f"  - cleanup{ext}\n"
        f"  - environments/ ({len(targets)} scripts)\n"
        f"  - commands/ ({total_commands} scripts)\n"
        f"  - manifest.json",
        title="Generation Complete",
        border_style="green",
    ))

    # Usage instructions
    if platform == "windows":
        console.print("\n[bold]To run the pipeline:[/bold]")
        console.print(f"  cd {output_dir}")
        console.print("  .\\setup_environments.ps1   # First time only")
        console.print("  .\\run_all_optimizations.ps1")
    else:
        console.print("\n[bold]To run the pipeline:[/bold]")
        console.print(f"  cd {output_dir}")
        console.print("  ./setup_environments.sh   # First time only")
        console.print("  ./run_all_optimizations.sh")


def _generate_manifest(
    output_dir: Path,
    model: str,
    task: str,
    category: str,
    targets: List[str],
    precisions: dict,
):
    """Generate pipeline manifest JSON."""
    manifest = {
        "model_id": model,
        "task": task,
        "category": category,
        "targets": targets,
        "precisions": precisions,
        "generated_by": "olive-auto",
        "version": "0.1.0",
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))


# Entry point
if __name__ == "__main__":
    app()
```

### 5.4 Entry Point Module

**File:** `olive_auto/__main__.py`

```python
"""Entry point for python -m olive_auto."""
from olive_auto.cli import app

if __name__ == "__main__":
    app()
```

### 5.5 Package Init

**File:** `olive_auto/__init__.py`

```python
"""
olive-auto: Batch optimization pipeline generator for Olive.

Generate complete optimization pipelines for HuggingFace models
across all supported targets and precisions.
"""

__version__ = "0.1.0"
__author__ = "Microsoft"

from olive_auto.cli import app
from olive_auto.task_detection import detect_model_info
from olive_auto.targets import TARGET_PRECISION_MATRIX, CATEGORY_TARGET_COMPATIBILITY
from olive_auto.command_generator import CommandGenerator
from olive_auto.environment import EnvironmentScriptGenerator

__all__ = [
    "app",
    "detect_model_info",
    "TARGET_PRECISION_MATRIX",
    "CATEGORY_TARGET_COMPATIBILITY",
    "CommandGenerator",
    "EnvironmentScriptGenerator",
]
```

---

## Part 6: Execution Plan

### Phase 1: Package Setup and Core Infrastructure (Week 1)

| Task | Description | Files |
|------|-------------|-------|
| 1.1 | Create `olive_auto/` package structure | `olive_auto/__init__.py`, `olive_auto/__main__.py` |
| 1.2 | Setup `pyproject.toml` with dependencies | `olive_auto/pyproject.toml` |
| 1.3 | Create task detection module | `olive_auto/task_detection.py` |
| 1.4 | Create target/precision matrix | `olive_auto/targets.py` |
| 1.5 | Unit tests for task detection | `olive_auto/tests/test_task_detection.py` |

### Phase 2: Script Generation (Week 2)

| Task | Description | Files |
|------|-------------|-------|
| 2.1 | Create Jinja2 templates for bash scripts | `olive_auto/templates/bash/*.jinja2` |
| 2.2 | Create Jinja2 templates for PowerShell | `olive_auto/templates/powershell/*.jinja2` |
| 2.3 | Implement environment script generator | `olive_auto/environment.py` |
| 2.4 | Implement command generator | `olive_auto/command_generator.py` |
| 2.5 | Unit tests for generators | `olive_auto/tests/test_*.py` |

### Phase 3: CLI Implementation (Week 3)

| Task | Description | Files |
|------|-------------|-------|
| 3.1 | Implement CLI with Typer | `olive_auto/cli.py` |
| 3.2 | Add Rich console output | `olive_auto/cli.py` |
| 3.3 | Implement pipeline executor | `olive_auto/executor.py` |
| 3.4 | Add parallel execution support | `olive_auto/executor.py` |
| 3.5 | CLI integration tests | `olive_auto/tests/test_cli.py` |

### Phase 4: Testing and Validation (Week 4)

| Task | Description | Files |
|------|-------------|-------|
| 4.1 | Test with LLM models (Phi-2, Llama) | - |
| 4.2 | Test with vision models (ViT, ResNet) | - |
| 4.3 | Test with audio models (Whisper) | - |
| 4.4 | Test with diffusion models (SD) | - |
| 4.5 | Cross-platform testing (Linux/Windows) | - |
| 4.6 | End-to-end pipeline execution tests | - |

### Phase 5: Documentation and Release (Week 5)

| Task | Description | Files |
|------|-------------|-------|
| 5.1 | Package README | `olive_auto/README.md` |
| 5.2 | User documentation | `docs/features/olive-auto.md` |
| 5.3 | Example pipelines | `examples/auto_optimize/` |
| 5.4 | PyPI packaging and release | `olive_auto/pyproject.toml` |
| 5.5 | Update main Olive README | `README.md` |

### Milestone Summary

| Milestone | Deliverable | Target |
|-----------|-------------|--------|
| M1 | Package structure + task detection | End of Week 1 |
| M2 | Script generation working | End of Week 2 |
| M3 | `olive-auto` CLI functional | End of Week 3 |
| M4 | All model types tested | End of Week 4 |
| M5 | Documentation + PyPI release | End of Week 5 |

---

## Part 7: Architecture Diagram

```
+------------------------------------------------------------------+
|                           olive-auto                               |
|                    -m microsoft/phi-2 -o ./pipeline                |
+------------------------------------------------------------------+
                                  |
                                  v
+------------------------------------------------------------------+
|                     1. Model Analysis                              |
|  +------------------+  +------------------+  +------------------+  |
|  | TasksManager     |  | Architecture     |  | Model Card       |  |
|  | Inference        |  | Pattern Match    |  | Metadata         |  |
|  +------------------+  +------------------+  +------------------+  |
|                          |                                         |
|                          v                                         |
|              Task: text-generation-with-past                       |
|              Category: nlp                                         |
+------------------------------------------------------------------+
                                  |
                                  v
+------------------------------------------------------------------+
|                    2. Target Selection                             |
|  +----------------------------------------------------------+    |
|  | Compatible targets for 'nlp':                              |    |
|  | cpu, cuda, qnn, openvino, tensorrt, directml, webgpu      |    |
|  +----------------------------------------------------------+    |
+------------------------------------------------------------------+
                                  |
                                  v
+------------------------------------------------------------------+
|                  3. Script Generation                              |
|                                                                    |
|  +------------------------+  +---------------------------+        |
|  | Environment Scripts    |  | Command Scripts           |        |
|  |------------------------|  |---------------------------|        |
|  | setup_cpu.sh           |  | cpu_fp32.sh               |        |
|  | setup_cuda.sh          |  | cpu_int8.sh               |        |
|  | setup_qnn.sh           |  | cpu_int4.sh               |        |
|  | setup_openvino.sh      |  | cuda_fp16.sh              |        |
|  | setup_tensorrt.sh      |  | cuda_int4.sh              |        |
|  | setup_directml.ps1     |  | qnn_int8.sh               |        |
|  | ...                    |  | openvino_int8.sh          |        |
|  +------------------------+  | ...                       |        |
|                              +---------------------------+        |
|                                                                    |
|  +----------------------------------------------------------+    |
|  | Master Scripts                                             |    |
|  |----------------------------------------------------------|    |
|  | setup_environments.sh    - Setup all environments         |    |
|  | run_all_optimizations.sh - Run all optimizations          |    |
|  | cleanup.sh               - Cleanup all environments       |    |
|  +----------------------------------------------------------+    |
+------------------------------------------------------------------+
                                  |
                                  v
+------------------------------------------------------------------+
|                       4. Output Structure                          |
|                                                                    |
|  pipeline/                                                         |
|  ├── manifest.json                                                 |
|  ├── setup_environments.sh                                         |
|  ├── run_all_optimizations.sh                                      |
|  ├── cleanup.sh                                                    |
|  ├── environments/                                                 |
|  │   ├── setup_cpu.sh                                              |
|  │   ├── setup_cuda.sh                                             |
|  │   └── ...                                                       |
|  ├── commands/                                                     |
|  │   ├── cpu_fp32.sh                                               |
|  │   ├── cpu_int8.sh                                               |
|  │   ├── cuda_fp16.sh                                              |
|  │   └── ...                                                       |
|  ├── outputs/           (after execution)                          |
|  │   ├── cpu_fp32/                                                 |
|  │   ├── cpu_int8/                                                 |
|  │   └── ...                                                       |
|  └── logs/              (after execution)                          |
|      ├── cpu_fp32.log                                              |
|      └── ...                                                       |
+------------------------------------------------------------------+
```

---

## Part 8: Usage Examples

### 8.1 Installation

```bash
# Install from PyPI (when published)
pip install olive-auto

# Install with olive integration for direct execution
pip install olive-auto[full]

# Install from source
git clone https://github.com/microsoft/Olive.git
cd Olive/olive_auto
pip install -e .
```

### 8.2 Basic Usage

```bash
# Generate pipeline for a HuggingFace model
olive-auto -m microsoft/phi-2 -o ./phi2_pipeline

# Output:
# +------------------+
# | Model Analysis   |
# +------------------+
# Model: microsoft/phi-2
# Task: text-generation-with-past
# Category: nlp
#
# +------------------------+
# | Target/Precision Matrix|
# +------------------------+
# | Target    | Precisions              | Commands |
# |-----------|-------------------------|----------|
# | cpu       | fp32, fp16, int8, int4  | 4        |
# | cuda      | fp32, fp16, bf16, ...   | 6        |
# | qnn       | fp16, int8, int4        | 3        |
# | ...       | ...                     | ...      |
#
# +----------------------+
# | Generation Complete  |
# +----------------------+
# Output: /path/to/phi2_pipeline
# Total commands: 28
```

### 8.3 Selective Targets

```bash
# Only CPU and CUDA targets
olive-auto -m openai/whisper-tiny \
    --targets cpu cuda \
    -o ./whisper_pipeline

# Short form with -t
olive-auto -m openai/whisper-tiny -t cpu cuda -o ./whisper_pipeline

# Exclude specific targets
olive-auto -m google/vit-base-patch16-224 \
    --exclude-targets qnn vitisai \
    -o ./vit_pipeline
```

### 8.4 Specific Precisions

```bash
# Only int8 quantization across all targets
olive-auto -m microsoft/phi-2 -p int8 -o ./phi2_int8

# Multiple precisions
olive-auto -m microsoft/phi-2 -p fp16 int8 int4 -o ./phi2_quantized
```

### 8.5 Run After Generation

```bash
# Generate and immediately run
olive-auto -m bert-base-uncased \
    --targets cpu \
    --run \
    -o ./bert_pipeline

# Run with parallel execution
olive-auto -m bert-base-uncased \
    --targets cpu cuda \
    --run \
    --parallel 4 \
    -o ./bert_pipeline
```

### 8.6 Platform-Specific Generation

```bash
# Generate for Linux (default)
olive-auto -m microsoft/phi-2 --platform linux -o ./phi2_linux

# Generate for Windows
olive-auto -m microsoft/phi-2 --platform windows -o ./phi2_windows

# Auto-detect current platform
olive-auto -m microsoft/phi-2 --platform auto -o ./phi2_pipeline
```

### 8.7 Dry Run

```bash
# Preview what would be generated without creating files
olive-auto -m microsoft/phi-2 --dry-run

# Verbose dry run
olive-auto -m microsoft/phi-2 --dry-run -v
```

### 8.8 Manual Execution

```bash
# Step 1: Generate pipeline
olive-auto -m microsoft/phi-2 -o ./phi2_pipeline

# Step 2: Setup environments (first time only)
cd ./phi2_pipeline
./setup_environments.sh

# Step 3: Run all optimizations
./run_all_optimizations.sh

# Or run individual optimizations
./commands/cuda_fp16.sh
./commands/cpu_int8.sh

# Step 4: Cleanup when done
./cleanup.sh
```

### 8.9 Windows PowerShell Execution

```powershell
# Step 1: Generate pipeline
olive-auto -m microsoft/phi-2 --platform windows -o .\phi2_pipeline

# Step 2: Setup environments
cd .\phi2_pipeline
.\setup_environments.ps1

# Step 3: Run all optimizations
.\run_all_optimizations.ps1

# Step 4: Cleanup
.\cleanup.ps1
```

### 8.10 Help and Version

```bash
# Show help
olive-auto --help

# Show version
olive-auto --version

# Verbose output
olive-auto -m microsoft/phi-2 -v -o ./pipeline
```

---

## Part 9: Test Plan

### 9.1 Unit Tests

```python
# olive_auto/tests/test_task_detection.py
import pytest
from olive_auto.task_detection import detect_model_info, infer_task_from_model


class TestTaskDetection:
    @pytest.mark.parametrize("model_id,expected_task,expected_category", [
        ("microsoft/phi-2", "text-generation-with-past", "nlp"),
        ("openai/whisper-tiny", "automatic-speech-recognition", "audio"),
        ("google/vit-base-patch16-224", "image-classification", "vision"),
        ("stabilityai/stable-diffusion-2", "stable-diffusion", "diffusers"),
    ])
    def test_detect_model_info(self, model_id, expected_task, expected_category):
        task, category = detect_model_info(model_id)
        assert task == expected_task
        assert category == expected_category

    def test_user_task_override(self):
        """User-specified task should override detection."""
        task, category = detect_model_info(
            "microsoft/phi-2",
            user_task="text-classification"
        )
        assert task == "text-classification"
        assert category == "nlp"


# olive_auto/tests/test_targets.py
import pytest
from olive_auto.targets import TARGET_PRECISION_MATRIX, CATEGORY_TARGET_COMPATIBILITY


class TestTargets:
    def test_all_targets_have_precisions(self):
        for target, config in TARGET_PRECISION_MATRIX.items():
            assert "precisions" in config
            assert len(config["precisions"]) > 0

    def test_all_categories_have_targets(self):
        for category, targets in CATEGORY_TARGET_COMPATIBILITY.items():
            assert len(targets) > 0
            for target in targets:
                assert target in TARGET_PRECISION_MATRIX


# olive_auto/tests/test_command_generator.py
import pytest
from pathlib import Path
from olive_auto.command_generator import CommandGenerator


class TestCommandGenerator:
    def test_generate_all_commands(self, tmp_path):
        generator = CommandGenerator(
            model_id="bert-base-uncased",
            task="fill-mask",
            category="nlp",
            output_dir=tmp_path,
            targets=["cpu", "cuda"],
            platform="linux",
        )
        commands = generator.generate_all_commands()

        # CPU: fp32, fp16, int8, int4 = 4
        # CUDA: fp32, fp16, bf16, int8, int4, bnb4 = 6
        assert len(commands) == 10

    def test_generate_command_scripts_linux(self, tmp_path):
        generator = CommandGenerator(
            model_id="bert-base-uncased",
            task="fill-mask",
            category="nlp",
            output_dir=tmp_path,
            targets=["cpu"],
            platform="linux",
        )
        scripts = generator.generate_command_scripts()

        assert (tmp_path / "commands" / "cpu_fp32.sh").exists()
        assert (tmp_path / "commands" / "cpu_int8.sh").exists()

    def test_generate_command_scripts_windows(self, tmp_path):
        generator = CommandGenerator(
            model_id="bert-base-uncased",
            task="fill-mask",
            category="nlp",
            output_dir=tmp_path,
            targets=["cpu"],
            platform="windows",
        )
        scripts = generator.generate_command_scripts()

        assert (tmp_path / "commands" / "cpu_fp32.ps1").exists()
        assert (tmp_path / "commands" / "cpu_int8.ps1").exists()


# olive_auto/tests/test_environment.py
import pytest
from pathlib import Path
from olive_auto.environment import EnvironmentScriptGenerator


class TestEnvironmentGenerator:
    def test_generate_linux_setup(self, tmp_path):
        generator = EnvironmentScriptGenerator(tmp_path, platform="linux")
        script_path = generator.generate_setup_script("cpu")

        assert script_path.exists()
        assert script_path.suffix == ".sh"
        content = script_path.read_text()
        assert "#!/bin/bash" in content
        assert "olive_cpu" in content

    def test_generate_windows_setup(self, tmp_path):
        generator = EnvironmentScriptGenerator(tmp_path, platform="windows")
        script_path = generator.generate_setup_script("cpu")

        assert script_path.exists()
        assert script_path.suffix == ".ps1"
        content = script_path.read_text()
        assert "$ErrorActionPreference" in content

    def test_generate_cleanup(self, tmp_path):
        generator = EnvironmentScriptGenerator(tmp_path, platform="linux")
        script_path = generator.generate_cleanup_script(["cpu", "cuda"])

        assert script_path.exists()
        content = script_path.read_text()
        assert "olive_cpu" in content
        assert "olive_cuda" in content
```

### 9.2 Integration Tests

```python
# olive_auto/tests/test_cli.py
import json
import pytest
from pathlib import Path
from typer.testing import CliRunner
from olive_auto.cli import app

runner = CliRunner()


class TestCLI:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "olive-auto" in result.output or "model" in result.output

    def test_version(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0

    def test_dry_run(self, tmp_path):
        result = runner.invoke(app, [
            "-m", "bert-base-uncased",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "Dry run" in result.output

    def test_generate_pipeline_llm(self, tmp_path):
        """Test pipeline generation for LLM model."""
        result = runner.invoke(app, [
            "-m", "microsoft/phi-2",
            "-o", str(tmp_path),
            "-t", "cpu",
        ])

        assert result.exit_code == 0
        assert (tmp_path / "manifest.json").exists()
        assert (tmp_path / "run_all_optimizations.sh").exists()
        assert (tmp_path / "environments" / "setup_cpu.sh").exists()

        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["model_id"] == "microsoft/phi-2"
        assert manifest["generated_by"] == "olive-auto"

    def test_generate_pipeline_vision(self, tmp_path):
        """Test pipeline generation for vision model."""
        result = runner.invoke(app, [
            "-m", "google/vit-base-patch16-224",
            "-o", str(tmp_path),
            "-t", "cpu", "cuda",
        ])

        assert result.exit_code == 0
        manifest = json.loads((tmp_path / "manifest.json").read_text())
        assert manifest["category"] == "vision"
        assert manifest["task"] == "image-classification"

    def test_windows_platform(self, tmp_path):
        """Test Windows script generation."""
        result = runner.invoke(app, [
            "-m", "bert-base-uncased",
            "-o", str(tmp_path),
            "-t", "cpu",
            "--platform", "windows",
        ])

        assert result.exit_code == 0
        assert (tmp_path / "run_all_optimizations.ps1").exists()
        assert (tmp_path / "setup_environments.ps1").exists()
```

### 9.3 End-to-End Tests

```python
# olive_auto/tests/test_e2e.py
import pytest
import subprocess
from pathlib import Path


@pytest.mark.e2e
@pytest.mark.slow
class TestE2E:
    def test_full_pipeline_cpu(self, tmp_path):
        """Full end-to-end test with CPU target."""
        # Generate pipeline
        result = subprocess.run([
            "olive-auto",
            "-m", "bert-base-uncased",
            "-o", str(tmp_path),
            "-t", "cpu",
            "-p", "fp32",
        ], capture_output=True, text=True)
        assert result.returncode == 0

        # Verify structure
        assert (tmp_path / "manifest.json").exists()
        assert (tmp_path / "commands" / "cpu_fp32.sh").exists()

        # Run setup (requires olive installed)
        # setup_result = subprocess.run(
        #     [str(tmp_path / "setup_environments.sh")],
        #     capture_output=True, text=True
        # )
        # assert setup_result.returncode == 0
```

---

## Part 9.5: Pipeline Executor Module

**File:** `olive_auto/executor.py`

```python
"""Pipeline execution engine for olive-auto."""
from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

console = Console()


@dataclass
class ExecutionResult:
    """Result of a single optimization command."""
    target: str
    precision: str
    success: bool
    return_code: int
    log_file: Path
    error_message: Optional[str] = None


class PipelineExecutor:
    """Execute generated optimization pipeline."""

    def __init__(self, pipeline_dir: Path, parallel: int = 1):
        self.pipeline_dir = pipeline_dir
        self.parallel = parallel
        self.manifest = self._load_manifest()
        self.results: List[ExecutionResult] = []

    def _load_manifest(self) -> dict:
        """Load pipeline manifest."""
        manifest_path = self.pipeline_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        return json.loads(manifest_path.read_text())

    def run(self) -> bool:
        """Execute the full pipeline."""
        console.print("\n[bold]Starting pipeline execution...[/bold]")

        # Get all commands to run
        commands = self._get_commands()
        total = len(commands)

        console.print(f"Total commands: {total}")
        console.print(f"Parallel workers: {self.parallel}\n")

        if self.parallel == 1:
            # Sequential execution
            self._run_sequential(commands)
        else:
            # Parallel execution
            self._run_parallel(commands)

        # Print summary
        return self._print_summary()

    def _get_commands(self) -> List[tuple]:
        """Get list of (target, precision, script_path) tuples."""
        commands = []
        commands_dir = self.pipeline_dir / "commands"

        for target, precisions in self.manifest["precisions"].items():
            for precision in precisions:
                # Determine script extension
                ext = ".ps1" if sys.platform == "win32" else ".sh"
                script_path = commands_dir / f"{target}_{precision}{ext}"
                if script_path.exists():
                    commands.append((target, precision, script_path))

        return commands

    def _run_sequential(self, commands: List[tuple]):
        """Run commands sequentially."""
        with Progress(console=console) as progress:
            task = progress.add_task("Running optimizations...", total=len(commands))

            for target, precision, script_path in commands:
                progress.update(task, description=f"Running {target}/{precision}...")
                result = self._execute_command(target, precision, script_path)
                self.results.append(result)
                progress.advance(task)

    def _run_parallel(self, commands: List[tuple]):
        """Run commands in parallel."""
        with Progress(console=console) as progress:
            task = progress.add_task("Running optimizations...", total=len(commands))

            with ThreadPoolExecutor(max_workers=self.parallel) as executor:
                futures = {
                    executor.submit(
                        self._execute_command, target, precision, script_path
                    ): (target, precision)
                    for target, precision, script_path in commands
                }

                for future in as_completed(futures):
                    target, precision = futures[future]
                    try:
                        result = future.result()
                        self.results.append(result)
                    except Exception as e:
                        self.results.append(ExecutionResult(
                            target=target,
                            precision=precision,
                            success=False,
                            return_code=-1,
                            log_file=Path(),
                            error_message=str(e),
                        ))
                    progress.advance(task)

    def _execute_command(
        self,
        target: str,
        precision: str,
        script_path: Path,
    ) -> ExecutionResult:
        """Execute a single optimization command."""
        log_file = self.pipeline_dir / "logs" / f"{target}_{precision}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(log_file, "w") as log:
                if sys.platform == "win32":
                    result = subprocess.run(
                        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script_path)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=3600,  # 1 hour timeout
                    )
                else:
                    result = subprocess.run(
                        ["bash", str(script_path)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=3600,
                    )

            return ExecutionResult(
                target=target,
                precision=precision,
                success=result.returncode == 0,
                return_code=result.returncode,
                log_file=log_file,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                target=target,
                precision=precision,
                success=False,
                return_code=-1,
                log_file=log_file,
                error_message="Timeout after 1 hour",
            )
        except Exception as e:
            return ExecutionResult(
                target=target,
                precision=precision,
                success=False,
                return_code=-1,
                log_file=log_file,
                error_message=str(e),
            )

    def _print_summary(self) -> bool:
        """Print execution summary and return success status."""
        passed = sum(1 for r in self.results if r.success)
        failed = sum(1 for r in self.results if not r.success)

        console.print("\n")

        # Create results table
        table = Table(title="Execution Results", show_header=True)
        table.add_column("Target", style="cyan")
        table.add_column("Precision", style="magenta")
        table.add_column("Status")
        table.add_column("Log File")

        for result in self.results:
            status = "[green]PASS[/green]" if result.success else "[red]FAIL[/red]"
            if result.error_message:
                status += f" ({result.error_message})"
            table.add_row(
                result.target,
                result.precision,
                status,
                str(result.log_file.name) if result.log_file.exists() else "-",
            )

        console.print(table)

        # Summary
        console.print(f"\n[bold]Summary:[/bold] {passed} passed, {failed} failed")

        if failed > 0:
            console.print("\n[yellow]Check log files for failure details:[/yellow]")
            for result in self.results:
                if not result.success:
                    console.print(f"  - {result.log_file}")

        return failed == 0
```

---

## Part 10: Future Enhancements

### 10.1 Cloud Integration
- Generate Azure ML pipeline YAML
- Generate AWS SageMaker pipeline
- Generate GCP Vertex AI pipeline

### 10.2 CI/CD Integration
- GitHub Actions workflow generation
- Azure DevOps pipeline generation
- Jenkins pipeline generation

### 10.3 Docker Support
- Generate Dockerfile per target
- Generate docker-compose.yml for parallel execution
- Pre-built Docker images with SDKs

### 10.4 Benchmarking
- Add benchmark scripts for each output
- Generate comparison reports
- Track latency, throughput, model size

### 10.5 Model Hub Integration
- Upload optimized models to HuggingFace Hub
- Generate model cards for optimized models
- Version tracking and lineage

---

## Appendix A: File Structure

**Standalone Package: `olive_auto/`**

```
olive_auto/
├── __init__.py              # Package init with exports
├── __main__.py              # Entry point for python -m olive_auto
├── cli.py                   # Main CLI using Typer
├── task_detection.py        # Model task/category detection
├── targets.py               # Target/precision matrix
├── environment.py           # Environment configurations & script generation
├── command_generator.py     # Command script generation
├── executor.py              # Pipeline execution engine
├── templates/               # Jinja2 script templates
│   ├── bash/
│   │   ├── setup_env.sh.jinja2
│   │   ├── run_command.sh.jinja2
│   │   ├── master_setup.sh.jinja2
│   │   ├── master_run.sh.jinja2
│   │   └── cleanup.sh.jinja2
│   └── powershell/
│       ├── setup_env.ps1.jinja2
│       ├── run_command.ps1.jinja2
│       ├── master_setup.ps1.jinja2
│       ├── master_run.ps1.jinja2
│       └── cleanup.ps1.jinja2
├── pyproject.toml           # Package configuration
├── README.md                # Package documentation
└── tests/
    ├── __init__.py
    ├── test_cli.py
    ├── test_task_detection.py
    ├── test_targets.py
    ├── test_environment.py
    ├── test_command_generator.py
    └── test_executor.py
```

**Location in Olive Repository:**

```
Olive/
├── olive/                   # Core Olive package (unchanged)
│   └── ...
├── olive_auto/              # NEW: Standalone olive-auto package
│   └── ...
├── examples/
│   └── auto_optimize/       # Example pipelines
│       ├── llm_example/
│       ├── vision_example/
│       └── audio_example/
└── docs/
    └── features/
        └── auto-optimize.md # Documentation
```

## Appendix B: Dependencies

**File:** `olive_auto/pyproject.toml`

```toml
[project]
name = "olive-auto"
version = "0.1.0"
description = "Batch optimization pipeline generator for Olive"
requires-python = ">=3.8"

dependencies = [
    # Core dependencies
    "huggingface_hub>=0.20.0",    # Model info fetching
    "jinja2>=3.0.0",               # Template rendering
    "pyyaml>=6.0",                 # YAML configuration
    "rich>=13.0.0",                # Beautiful CLI output
    "typer>=0.9.0",                # Modern CLI framework
]

[project.optional-dependencies]
# Development dependencies
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
]

# Full installation with Olive for direct execution
full = [
    "olive-ai>=0.7.0",
    "optimum>=1.17.0",
    "transformers>=4.36.0",
]

# Target-specific extras (for running generated pipelines)
cpu = ["onnxruntime>=1.16.0"]
cuda = ["onnxruntime-gpu>=1.16.0", "bitsandbytes>=0.41.0"]
openvino = ["openvino>=2024.0", "optimum-intel>=1.14.0"]
directml = ["onnxruntime-directml>=1.16.0"]

[project.scripts]
olive-auto = "olive_auto.cli:app"
```

## Appendix C: CLI Reference

```
olive-auto - Generate optimization pipelines for HuggingFace models

USAGE:
    olive-auto [OPTIONS]

OPTIONS:
    -m, --model TEXT              HuggingFace model ID or local path [required]
    --task TEXT                   Model task (auto-detected if not specified)
    --trust-remote-code           Trust remote code from HuggingFace

    -t, --targets TEXT            Specific targets (can specify multiple)
    -p, --precisions TEXT         Specific precisions (can specify multiple)
    --exclude-targets TEXT        Targets to exclude

    -o, --output-dir PATH         Output directory [default: ./olive_auto_pipeline]
    --platform [linux|windows|auto]  Target platform [default: auto]

    --parallel INTEGER            Number of parallel optimizations [default: 1]
    --run                         Execute pipeline after generation
    --dry-run                     Show what would be generated

    -v, --verbose                 Enable verbose output
    --version                     Show version and exit
    --help                        Show this message and exit

TARGETS:
    cpu         CPU inference via ONNX Runtime
    cuda        NVIDIA GPU via CUDA
    qnn         Qualcomm NPU
    openvino    Intel CPU/GPU/VPU
    tensorrt    NVIDIA TensorRT
    directml    Windows DirectML
    webgpu      Browser WebGPU
    vitisai     AMD/Xilinx FPGA/NPU
    rocm        AMD GPU via ROCm

PRECISIONS:
    fp32        32-bit floating point
    fp16        16-bit floating point
    bf16        Brain floating point 16
    int8        8-bit integer quantization
    int4        4-bit integer quantization
    bnb4        bitsandbytes 4-bit

EXAMPLES:
    # Basic usage
    olive-auto -m microsoft/phi-2 -o ./pipeline

    # Specific targets and precisions
    olive-auto -m microsoft/phi-2 -t cpu cuda -p int8 int4

    # Generate and run
    olive-auto -m bert-base-uncased -t cpu --run

    # Windows PowerShell scripts
    olive-auto -m microsoft/phi-2 --platform windows
```
