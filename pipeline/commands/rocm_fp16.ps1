# Optimization: rocm with fp16 precision
# Model: microsoft/phi-2
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogFile = Join-Path $ScriptDir "..\logs\rocm_fp16.log"

Write-Host '========================================'
Write-Host 'Target: rocm'
Write-Host 'Precision: fp16'
Write-Host 'Model: microsoft/phi-2'
Write-Host '========================================'

# Activate environment
& "$ScriptDir\..\environments\olive_rocm\Scripts\Activate.ps1"

# Create output directory
New-Item -ItemType Directory -Force -Path "pipeline\outputs\rocm_fp16" | Out-Null

Write-Host 'Starting optimization...'
Write-Host "Log file: $LogFile"

# Run optimization
olive optimize `
    -m "microsoft/phi-2" `
    --task text-generation-with-past `
    --provider ROCMExecutionProvider `
    --precision fp16 `
    -o "pipeline\outputs\rocm_fp16" `
    --device gpu 2>&1 | Tee-Object -FilePath $LogFile

Write-Host ''
Write-Host 'Optimization complete: rocm fp16'