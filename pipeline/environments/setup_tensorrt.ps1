# Environment setup for tensorrt
$ErrorActionPreference = 'Stop'

$ENV_NAME = 'olive_tensorrt'
$ENV_DIR = Join-Path $PSScriptRoot $ENV_NAME

# Use Python 3.11-3.13 (pydantic v1 incompatible with 3.14+)
$PYTHON = $env:OLIVE_PYTHON ?? "$env:LOCALAPPDATA\miniconda3\envs\olive\python.exe"
if (-not (Test-Path $PYTHON)) {
    Write-Error "Python not found at $PYTHON. Set OLIVE_PYTHON env var to a Python 3.11-3.13 executable."
    exit 1
}

Write-Host '========================================'
Write-Host 'Setting up environment for: tensorrt'
Write-Host '========================================'

# System requirements
Write-Host 'System requirements:'
Write-Host '  - TensorRT >= 8.6'
Write-Host '  - CUDA Toolkit >= 11.8'

# Create virtual environment (check for activation script, not just directory)
$ActivateScript = "$ENV_DIR\Scripts\Activate.ps1"
if (-not (Test-Path $ActivateScript)) {
    if (Test-Path $ENV_DIR) {
        Remove-Item -Recurse -Force $ENV_DIR
    }
    & $PYTHON -m venv $ENV_DIR
}

# Activate environment
& "$ENV_DIR\Scripts\Activate.ps1"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install pip packages
pip install 'olive-ai'
pip install 'onnxruntime-gpu'
pip install 'onnxruntime-genai-cuda'
pip install 'tensorrt'
pip install 'transformers'
pip install 'torch'

# Set environment variables
$env:CUDA_HOME = "/usr/local/cuda"
$env:TENSORRT_HOME = "/usr/local/tensorrt"
$env:LD_LIBRARY_PATH = "$env:TENSORRT_HOME/lib:$env:CUDA_HOME/lib64:$env:LD_LIBRARY_PATH"

Write-Host ''
Write-Host 'Environment tensorrt setup complete!'