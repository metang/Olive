# Environment setup for cuda
$ErrorActionPreference = 'Stop'

$ENV_NAME = 'olive_cuda'
$ENV_DIR = Join-Path $PSScriptRoot $ENV_NAME

Write-Host '========================================'
Write-Host 'Setting up environment for: cuda'
Write-Host '========================================'

# System requirements
Write-Host 'System requirements:'
Write-Host '  - CUDA Toolkit >= 11.8'
Write-Host '  - cuDNN >= 8.6'

# Create virtual environment
if (-not (Test-Path $ENV_DIR)) {
    python -m venv $ENV_DIR
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