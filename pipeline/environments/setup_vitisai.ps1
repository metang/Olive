# Environment setup for vitisai
$ErrorActionPreference = 'Stop'

$ENV_NAME = 'olive_vitisai'
$ENV_DIR = Join-Path $PSScriptRoot $ENV_NAME

# Use Python 3.11-3.13 (pydantic v1 incompatible with 3.14+)
$PYTHON = $env:OLIVE_PYTHON ?? "$env:LOCALAPPDATA\miniconda3\envs\olive\python.exe"
if (-not (Test-Path $PYTHON)) {
    Write-Error "Python not found at $PYTHON. Set OLIVE_PYTHON env var to a Python 3.11-3.13 executable."
    exit 1
}

Write-Host '========================================'
Write-Host 'Setting up environment for: vitisai'
Write-Host '========================================'

# System requirements
Write-Host 'System requirements:'
Write-Host '  - Vitis AI >= 3.5'
Write-Host '  - XRT (Xilinx Runtime)'

# Create virtual environment
if (-not (Test-Path $ENV_DIR)) {
    & $PYTHON -m venv $ENV_DIR
}

# Activate environment
& "$ENV_DIR\Scripts\Activate.ps1"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install pip packages
pip install 'olive-ai'
pip install 'vai-q-onnx'
pip install 'transformers'

# Set environment variables
$env:VITIS_AI_HOME = "/opt/vitis-ai"
$env:XLNX_VART_FIRMWARE = "/opt/vitis-ai/firmware"

Write-Host ''
Write-Host 'Environment vitisai setup complete!'