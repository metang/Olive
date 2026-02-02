# Optimization: openvino with fp32 precision
# Model: microsoft/phi-2
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogFile = Join-Path $ScriptDir "..\logs\openvino_fp32.log"

Write-Host '========================================'
Write-Host 'Target: openvino'
Write-Host 'Precision: fp32'
Write-Host 'Model: microsoft/phi-2'
Write-Host '========================================'

# Activate environment
& "$ScriptDir\..\environments\olive_openvino\Scripts\Activate.ps1"

# Create output directory
New-Item -ItemType Directory -Force -Path "pipeline\outputs\openvino_fp32" | Out-Null

Write-Host 'Starting optimization...'
Write-Host "Log file: $LogFile"

# Run optimization
olive optimize `
    -m "microsoft/phi-2" `
    --task text-generation-with-past `
    --provider OpenVINOExecutionProvider `
    --precision fp32 `
    -o "pipeline\outputs\openvino_fp32" 2>&1 | Tee-Object -FilePath $LogFile

Write-Host ''
Write-Host 'Optimization complete: openvino fp32'