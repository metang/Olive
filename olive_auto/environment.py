"""Environment setup and management for olive-auto."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class EnvironmentConfig:
    """Configuration for a target-specific environment."""

    target: str
    python_version: str = "3.10"
    olive_extras: List[str] = field(default_factory=list)
    pip_packages: List[str] = field(default_factory=list)
    conda_packages: List[str] = field(default_factory=list)
    system_requirements: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    pre_install_script: Optional[str] = None
    post_install_script: Optional[str] = None


# Environment configurations per target
ENVIRONMENT_CONFIGS: Dict[str, EnvironmentConfig] = {
    "cpu": EnvironmentConfig(
        target="cpu",
        olive_extras=[],
        pip_packages=[
            "olive-ai",
            "onnxruntime",
            "onnxruntime-genai",
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
            "onnxruntime-genai-cuda",
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
    echo "WARNING: QNN_SDK_ROOT not set. Please install Qualcomm AI Engine Direct SDK."
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
            "onnxruntime-genai-directml",
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


class EnvironmentScriptGenerator:
    """Generate setup/cleanup scripts for target environments."""

    def __init__(self, output_dir: Path, platform: str = "linux"):
        self.output_dir = Path(output_dir)
        self.platform = platform
        self.env_dir = self.output_dir / "environments"
        self.env_dir.mkdir(parents=True, exist_ok=True)

    def generate_setup_script(self, target: str) -> Path:
        """Generate environment setup script for a target."""
        if target not in ENVIRONMENT_CONFIGS:
            raise ValueError(f"Unknown target: {target}")

        config = ENVIRONMENT_CONFIGS[target]

        if self.platform == "windows":
            return self._generate_powershell_setup(config)
        else:
            return self._generate_bash_setup(config)

    def _generate_bash_setup(self, config: EnvironmentConfig) -> Path:
        """Generate bash setup script."""
        script_path = self.env_dir / f"setup_{config.target}.sh"

        lines = [
            "#!/bin/bash",
            f"# Environment setup for {config.target}",
            "set -e",
            "",
            f'ENV_NAME="olive_{config.target}"',
            'ENV_DIR="$(dirname $0)/$ENV_NAME"',
            "",
            'echo "========================================"',
            f'echo "Setting up environment for: {config.target}"',
            'echo "========================================"',
            "",
        ]

        # System requirements check
        if config.system_requirements:
            lines.extend([
                "# System requirements",
                'echo "System requirements:"',
            ])
            for req in config.system_requirements:
                lines.append(f'echo "  - {req}"')
            lines.append("")

        # Pre-install script
        if config.pre_install_script:
            lines.extend([
                "# Pre-install checks",
                config.pre_install_script.strip(),
                "",
            ])

        # Create virtual environment
        lines.extend([
            "# Create virtual environment",
            'if [ ! -d "$ENV_DIR" ]; then',
            f'    python3 -m venv "$ENV_DIR"',
            "fi",
            "",
            "# Activate environment",
            'source "$ENV_DIR/bin/activate"',
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
                lines.append(f'pip install "{pkg}"')
            lines.append("")

        # Set environment variables
        if config.env_vars:
            lines.extend([
                "# Set environment variables",
                'cat >> "$ENV_DIR/bin/activate" << \'ENVVARS\'',
            ])
            for key, value in config.env_vars.items():
                lines.append(f'export {key}="{value}"')
            lines.extend([
                "ENVVARS",
                "",
            ])

        # Post-install script
        if config.post_install_script:
            lines.extend([
                "# Post-install setup",
                config.post_install_script.strip(),
                "",
            ])

        lines.extend([
            'echo ""',
            f'echo "Environment \'{config.target}\' setup complete!"',
            'echo "Activate with: source $ENV_DIR/bin/activate"',
        ])

        script_path.write_text("\n".join(lines))
        # Make executable on non-Windows
        if os.name != "nt":
            script_path.chmod(0o755)

        return script_path

    def _generate_powershell_setup(self, config: EnvironmentConfig) -> Path:
        """Generate PowerShell setup script for Windows."""
        script_path = self.env_dir / f"setup_{config.target}.ps1"

        lines = [
            f"# Environment setup for {config.target}",
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

        # System requirements check
        if config.system_requirements:
            lines.extend([
                "# System requirements",
                "Write-Host 'System requirements:'",
            ])
            for req in config.system_requirements:
                lines.append(f"Write-Host '  - {req}'")
            lines.append("")

        # Create virtual environment
        lines.extend([
            "# Create virtual environment",
            "if (-not (Test-Path $ENV_DIR)) {",
            "    python -m venv $ENV_DIR",
            "}",
            "",
            "# Activate environment",
            '& "$ENV_DIR\\Scripts\\Activate.ps1"',
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
                lines.append(f'$env:{key} = "{ps_value}"')
            lines.append("")

        lines.extend([
            "Write-Host ''",
            f"Write-Host 'Environment {config.target} setup complete!'",
        ])

        script_path.write_text("\n".join(lines))
        return script_path

    def generate_master_setup_script(self, targets: List[str]) -> Path:
        """Generate master script to setup all environments."""
        if self.platform == "windows":
            return self._generate_master_setup_powershell(targets)
        else:
            return self._generate_master_setup_bash(targets)

    def _generate_master_setup_bash(self, targets: List[str]) -> Path:
        """Generate master bash setup script."""
        script_path = self.output_dir / "setup_environments.sh"

        lines = [
            "#!/bin/bash",
            "# Master setup script for all olive-auto environments",
            "set -e",
            "",
            'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            "",
            'echo "========================================"',
            'echo "Setting up all olive-auto environments"',
            'echo "========================================"',
            "",
        ]

        for target in targets:
            lines.extend([
                f"# Setup {target}",
                f'echo ""',
                f'echo "Setting up {target}..."',
                f'"$SCRIPT_DIR/environments/setup_{target}.sh"',
                "",
            ])

        lines.extend([
            'echo ""',
            'echo "========================================"',
            'echo "All environments setup complete!"',
            'echo "========================================"',
        ])

        script_path.write_text("\n".join(lines))
        if os.name != "nt":
            script_path.chmod(0o755)

        return script_path

    def _generate_master_setup_powershell(self, targets: List[str]) -> Path:
        """Generate master PowerShell setup script."""
        script_path = self.output_dir / "setup_environments.ps1"

        lines = [
            "# Master setup script for all olive-auto environments",
            "$ErrorActionPreference = 'Stop'",
            "",
            "$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path",
            "",
            "Write-Host '========================================'",
            "Write-Host 'Setting up all olive-auto environments'",
            "Write-Host '========================================'",
            "",
        ]

        for target in targets:
            lines.extend([
                f"# Setup {target}",
                "Write-Host ''",
                f"Write-Host 'Setting up {target}...'",
                f'& "$ScriptDir\\environments\\setup_{target}.ps1"',
                "",
            ])

        lines.extend([
            "Write-Host ''",
            "Write-Host '========================================'",
            "Write-Host 'All environments setup complete!'",
            "Write-Host '========================================'",
        ])

        script_path.write_text("\n".join(lines))
        return script_path

    def generate_cleanup_script(self, targets: List[str]) -> Path:
        """Generate cleanup script for all environments."""
        if self.platform == "windows":
            return self._generate_cleanup_powershell(targets)
        else:
            return self._generate_cleanup_bash(targets)

    def _generate_cleanup_bash(self, targets: List[str]) -> Path:
        """Generate bash cleanup script."""
        script_path = self.output_dir / "cleanup.sh"

        lines = [
            "#!/bin/bash",
            "# Cleanup script for olive-auto environments",
            "set -e",
            "",
            'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
            "",
            'echo "Cleaning up olive-auto environments..."',
            "",
        ]

        for target in targets:
            lines.extend([
                f"# Cleanup {target} environment",
                f'if [ -d "$SCRIPT_DIR/environments/olive_{target}" ]; then',
                f'    echo "Removing olive_{target} environment..."',
                f'    rm -rf "$SCRIPT_DIR/environments/olive_{target}"',
                "fi",
                "",
            ])

        lines.extend([
            "# Cleanup temporary files",
            'if [ -d "$SCRIPT_DIR/temp" ]; then',
            '    echo "Removing temporary files..."',
            '    rm -rf "$SCRIPT_DIR/temp"',
            "fi",
            "",
            "# Optional: Cleanup output models",
            'read -p "Remove output models? (y/N) " -n 1 -r',
            "echo",
            'if [[ $REPLY =~ ^[Yy]$ ]]; then',
            '    rm -rf "$SCRIPT_DIR/outputs"',
            '    echo "Output models removed."',
            "fi",
            "",
            'echo "Cleanup complete!"',
        ])

        script_path.write_text("\n".join(lines))
        if os.name != "nt":
            script_path.chmod(0o755)

        return script_path

    def _generate_cleanup_powershell(self, targets: List[str]) -> Path:
        """Generate PowerShell cleanup script."""
        script_path = self.output_dir / "cleanup.ps1"

        lines = [
            "# Cleanup script for olive-auto environments",
            "$ErrorActionPreference = 'Stop'",
            "",
            "$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path",
            "",
            "Write-Host 'Cleaning up olive-auto environments...'",
            "",
        ]

        for target in targets:
            lines.extend([
                f"# Cleanup {target} environment",
                f'$envPath = Join-Path $ScriptDir "environments\\olive_{target}"',
                "if (Test-Path $envPath) {",
                f"    Write-Host 'Removing olive_{target} environment...'",
                "    Remove-Item -Recurse -Force $envPath",
                "}",
                "",
            ])

        lines.extend([
            "# Cleanup temporary files",
            '$tempPath = Join-Path $ScriptDir "temp"',
            "if (Test-Path $tempPath) {",
            "    Write-Host 'Removing temporary files...'",
            "    Remove-Item -Recurse -Force $tempPath",
            "}",
            "",
            "# Optional: Cleanup output models",
            "$response = Read-Host 'Remove output models? (y/N)'",
            "if ($response -eq 'y' -or $response -eq 'Y') {",
            '    $outputPath = Join-Path $ScriptDir "outputs"',
            "    if (Test-Path $outputPath) {",
            "        Remove-Item -Recurse -Force $outputPath",
            "        Write-Host 'Output models removed.'",
            "    }",
            "}",
            "",
            "Write-Host 'Cleanup complete!'",
        ])

        script_path.write_text("\n".join(lines))
        return script_path
