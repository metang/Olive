# Environment setup for webgpu
$ErrorActionPreference = 'Stop'

$ENV_NAME = 'olive_webgpu'
$ENV_DIR = Join-Path $PSScriptRoot $ENV_NAME

# Use Python 3.11-3.13 (pydantic v1 incompatible with 3.14+)
$PYTHON = $env:OLIVE_PYTHON ?? "$env:LOCALAPPDATA\miniconda3\envs\olive\python.exe"
if (-not (Test-Path $PYTHON)) {
    Write-Error "Python not found at $PYTHON. Set OLIVE_PYTHON env var to a Python 3.11-3.13 executable."
    exit 1
}

Write-Host '========================================'
Write-Host 'Setting up environment for: webgpu'
Write-Host '========================================'

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
pip install 'onnxruntime-web'
pip install 'onnxruntime-genai'
pip install 'transformers'

Write-Host ''
Write-Host 'Environment webgpu setup complete!'