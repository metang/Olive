# Environment setup for directml
$ErrorActionPreference = 'Stop'

$ENV_NAME = 'olive_directml'
$ENV_DIR = Join-Path $PSScriptRoot $ENV_NAME

# Use Python 3.11-3.13 (pydantic v1 incompatible with 3.14+)
$PYTHON = $env:OLIVE_PYTHON ?? "$env:LOCALAPPDATA\miniconda3\envs\olive\python.exe"
if (-not (Test-Path $PYTHON)) {
    Write-Error "Python not found at $PYTHON. Set OLIVE_PYTHON env var to a Python 3.11-3.13 executable."
    exit 1
}

Write-Host '========================================'
Write-Host 'Setting up environment for: directml'
Write-Host '========================================'

# System requirements
Write-Host 'System requirements:'
Write-Host '  - Windows 10/11'
Write-Host '  - DirectX 12 compatible GPU'

# Create virtual environment
if (-not (Test-Path $ENV_DIR)) {
    & $PYTHON -m venv $ENV_DIR
}

# Activate environment
& "$ENV_DIR\Scripts\Activate.ps1"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install pip packages
pip install 'olive-ai[directml]'
pip install 'onnxruntime-directml'
pip install 'onnxruntime-genai-directml'
pip install 'torch-directml'
pip install 'transformers'

Write-Host ''
Write-Host 'Environment directml setup complete!'