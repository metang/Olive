# Optimization: vitisai with int4 precision
# Model: microsoft/phi-2
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogFile = Join-Path $ScriptDir "..\logs\vitisai_int4.log"

Write-Host '========================================'
Write-Host 'Target: vitisai'
Write-Host 'Precision: int4'
Write-Host 'Model: microsoft/phi-2'
Write-Host '========================================'

# Activate environment
& "$ScriptDir\..\environments\olive_vitisai\Scripts\Activate.ps1"

# Create output directory
New-Item -ItemType Directory -Force -Path "pipeline\outputs\vitisai_int4" | Out-Null

Write-Host 'Starting optimization...'
Write-Host "Log file: $LogFile"

# Run optimization
olive optimize `
    -m "microsoft/phi-2" `
    --task text-generation-with-past `
    --provider VitisAIExecutionProvider `
    --precision int4 `
    -o "pipeline\outputs\vitisai_int4" `
    --device npu 2>&1 | Tee-Object -FilePath $LogFile

Write-Host ''
Write-Host 'Optimization complete: vitisai int4'
