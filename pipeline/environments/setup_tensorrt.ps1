# Environment setup for tensorrt
$ErrorActionPreference = 'Stop'

$ENV_NAME = 'olive_tensorrt'
$ENV_DIR = Join-Path $PSScriptRoot $ENV_NAME

Write-Host '========================================'
Write-Host 'Setting up environment for: tensorrt'
Write-Host '========================================'

# System requirements
Write-Host 'System requirements:'
Write-Host '  - TensorRT >= 8.6'
Write-Host '  - CUDA Toolkit >= 11.8'

# Create virtual environment
if (-not (Test-Path $ENV_DIR)) {
    python -m venv $ENV_DIR
}

# Activate environment
& "$ENV_DIR\Scripts\Activate.ps1"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install pip packages
pip install 'olive-ai'
pip install 'onnxruntime-gpu'
pip install 'tensorrt'
pip install 'transformers'
pip install 'torch'

# Set environment variables
$env:CUDA_HOME = "/usr/local/cuda"
$env:TENSORRT_HOME = "/usr/local/tensorrt"
$env:LD_LIBRARY_PATH = "$env:TENSORRT_HOME/lib:$env:CUDA_HOME/lib64:$env:LD_LIBRARY_PATH"

Write-Host ''
Write-Host 'Environment tensorrt setup complete!'