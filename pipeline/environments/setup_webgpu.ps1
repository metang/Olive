# Environment setup for webgpu
$ErrorActionPreference = 'Stop'

$ENV_NAME = 'olive_webgpu'
$ENV_DIR = Join-Path $PSScriptRoot $ENV_NAME

Write-Host '========================================'
Write-Host 'Setting up environment for: webgpu'
Write-Host '========================================'

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
pip install 'onnxruntime-web'
pip install 'transformers'

Write-Host ''
Write-Host 'Environment webgpu setup complete!'