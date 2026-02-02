# Environment setup for rocm
$ErrorActionPreference = 'Stop'

$ENV_NAME = 'olive_rocm'
$ENV_DIR = Join-Path $PSScriptRoot $ENV_NAME

Write-Host '========================================'
Write-Host 'Setting up environment for: rocm'
Write-Host '========================================'

# System requirements
Write-Host 'System requirements:'
Write-Host '  - ROCm >= 5.7'
Write-Host '  - AMD GPU (gfx9 or later)'

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
pip install 'torch-rocm'
pip install 'transformers'

# Set environment variables
$env:ROCM_HOME = "/opt/rocm"
$env:HIP_PATH = "/opt/rocm/hip"
$env:PATH = "$env:ROCM_HOME/bin:$env:PATH"
$env:LD_LIBRARY_PATH = "$env:ROCM_HOME/lib:$env:LD_LIBRARY_PATH"

Write-Host ''
Write-Host 'Environment rocm setup complete!'