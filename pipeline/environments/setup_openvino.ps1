# Environment setup for openvino
$ErrorActionPreference = 'Stop'

$ENV_NAME = 'olive_openvino'
$ENV_DIR = Join-Path $PSScriptRoot $ENV_NAME

# Use Python 3.11-3.13 (pydantic v1 incompatible with 3.14+)
$PYTHON = $env:OLIVE_PYTHON ?? "$env:LOCALAPPDATA\miniconda3\envs\olive\python.exe"
if (-not (Test-Path $PYTHON)) {
    Write-Error "Python not found at $PYTHON. Set OLIVE_PYTHON env var to a Python 3.11-3.13 executable."
    exit 1
}

Write-Host '========================================'
Write-Host 'Setting up environment for: openvino'
Write-Host '========================================'

# System requirements
Write-Host 'System requirements:'
Write-Host '  - OpenVINO Toolkit >= 2024.0'

# Create virtual environment
if (-not (Test-Path $ENV_DIR)) {
    & $PYTHON -m venv $ENV_DIR
}

# Activate environment
& "$ENV_DIR\Scripts\Activate.ps1"

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# Install pip packages
pip install 'olive-ai[openvino]'
pip install 'openvino'
pip install 'openvino-dev'
pip install 'optimum[openvino]'
pip install 'onnxruntime-genai'
pip install 'transformers'

# Set environment variables
$env:OPENVINO_HOME = "/opt/intel/openvino"

Write-Host ''
Write-Host 'Environment openvino setup complete!'