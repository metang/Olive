# Environment setup for directml
$ErrorActionPreference = 'Stop'

$ENV_NAME = 'olive_directml'
$ENV_DIR = Join-Path $PSScriptRoot $ENV_NAME

Write-Host '========================================'
Write-Host 'Setting up environment for: directml'
Write-Host '========================================'

# System requirements
Write-Host 'System requirements:'
Write-Host '  - Windows 10/11'
Write-Host '  - DirectX 12 compatible GPU'

# Create virtual environment
if (-not (Test-Path $ENV_DIR)) {
    python -m venv $ENV_DIR
}

# Activate environment
& "$ENV_DIR\Scripts\Activate.ps1"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install pip packages
pip install 'olive-ai[directml]'
pip install 'onnxruntime-directml'
pip install 'torch-directml'
pip install 'transformers'

Write-Host ''
Write-Host 'Environment directml setup complete!'