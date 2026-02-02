# Environment setup for cuda
$ErrorActionPreference = 'Stop'

$ENV_NAME = 'olive_cuda'
$ENV_DIR = Join-Path $PSScriptRoot $ENV_NAME

# Use Python 3.11-3.13 (pydantic v1 incompatible with 3.14+)
$PYTHON = $env:OLIVE_PYTHON ?? "$env:LOCALAPPDATA\miniconda3\envs\olive\python.exe"
if (-not (Test-Path $PYTHON)) {
    Write-Error "Python not found at $PYTHON. Set OLIVE_PYTHON env var to a Python 3.11-3.13 executable."
    exit 1
}

Write-Host '========================================'
Write-Host 'Setting up environment for: cuda'
Write-Host '========================================'

# System requirements
Write-Host 'System requirements:'
Write-Host '  - CUDA Toolkit >= 11.8'
Write-Host '  - cuDNN >= 8.6'

# Create virtual environment
if (-not (Test-Path $ENV_DIR)) {
    & $PYTHON -m venv $ENV_DIR
}

# Activate environment
& "$ENV_DIR\Scripts\Activate.ps1"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install pip packages
pip install 'olive-ai[gpu]'
pip install 'onnxruntime-gpu'
pip install 'transformers'
pip install 'torch'
pip install 'bitsandbytes'

# Set environment variables
$env:CUDA_HOME = "/usr/local/cuda"
$env:LD_LIBRARY_PATH = "$env:CUDA_HOME/lib64:$env:LD_LIBRARY_PATH"

Write-Host ''
Write-Host 'Environment cuda setup complete!'