# Environment setup for qnn
$ErrorActionPreference = 'Stop'

$ENV_NAME = 'olive_qnn'
$ENV_DIR = Join-Path $PSScriptRoot $ENV_NAME

# Use Python 3.11-3.13 (pydantic v1 incompatible with 3.14+)
$PYTHON = $env:OLIVE_PYTHON ?? "$env:LOCALAPPDATA\miniconda3\envs\olive\python.exe"
if (-not (Test-Path $PYTHON)) {
    Write-Error "Python not found at $PYTHON. Set OLIVE_PYTHON env var to a Python 3.11-3.13 executable."
    exit 1
}

Write-Host '========================================'
Write-Host 'Setting up environment for: qnn'
Write-Host '========================================'

# System requirements
Write-Host 'System requirements:'
Write-Host '  - Qualcomm AI Engine Direct SDK >= 2.19'

# Create virtual environment
if (-not (Test-Path $ENV_DIR)) {
    & $PYTHON -m venv $ENV_DIR
}

# Activate environment
& "$ENV_DIR\Scripts\Activate.ps1"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install pip packages
pip install 'olive-ai[qualcomm]'
pip install 'onnxruntime-qnn'
pip install 'transformers'

# Set environment variables
$env:QNN_SDK_ROOT = "/opt/qnn-sdk"
$env:PATH = "$env:QNN_SDK_ROOT/bin:$env:PATH"
$env:LD_LIBRARY_PATH = "$env:QNN_SDK_ROOT/lib:$env:LD_LIBRARY_PATH"

Write-Host ''
Write-Host 'Environment qnn setup complete!'