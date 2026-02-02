# Cleanup script for olive-auto environments
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host 'Cleaning up olive-auto environments...'

# Cleanup cpu environment
$envPath = Join-Path $ScriptDir "environments\olive_cpu"
if (Test-Path $envPath) {
    Write-Host 'Removing olive_cpu environment...'
    Remove-Item -Recurse -Force $envPath
}

# Cleanup cuda environment
$envPath = Join-Path $ScriptDir "environments\olive_cuda"
if (Test-Path $envPath) {
    Write-Host 'Removing olive_cuda environment...'
    Remove-Item -Recurse -Force $envPath
}

# Cleanup qnn environment
$envPath = Join-Path $ScriptDir "environments\olive_qnn"
if (Test-Path $envPath) {
    Write-Host 'Removing olive_qnn environment...'
    Remove-Item -Recurse -Force $envPath
}

# Cleanup openvino environment
$envPath = Join-Path $ScriptDir "environments\olive_openvino"
if (Test-Path $envPath) {
    Write-Host 'Removing olive_openvino environment...'
    Remove-Item -Recurse -Force $envPath
}

# Cleanup vitisai environment
$envPath = Join-Path $ScriptDir "environments\olive_vitisai"
if (Test-Path $envPath) {
    Write-Host 'Removing olive_vitisai environment...'
    Remove-Item -Recurse -Force $envPath
}

# Cleanup webgpu environment
$envPath = Join-Path $ScriptDir "environments\olive_webgpu"
if (Test-Path $envPath) {
    Write-Host 'Removing olive_webgpu environment...'
    Remove-Item -Recurse -Force $envPath
}

# Cleanup tensorrt environment
$envPath = Join-Path $ScriptDir "environments\olive_tensorrt"
if (Test-Path $envPath) {
    Write-Host 'Removing olive_tensorrt environment...'
    Remove-Item -Recurse -Force $envPath
}

# Cleanup directml environment
$envPath = Join-Path $ScriptDir "environments\olive_directml"
if (Test-Path $envPath) {
    Write-Host 'Removing olive_directml environment...'
    Remove-Item -Recurse -Force $envPath
}

# Cleanup rocm environment
$envPath = Join-Path $ScriptDir "environments\olive_rocm"
if (Test-Path $envPath) {
    Write-Host 'Removing olive_rocm environment...'
    Remove-Item -Recurse -Force $envPath
}

# Cleanup temporary files
$tempPath = Join-Path $ScriptDir "temp"
if (Test-Path $tempPath) {
    Write-Host 'Removing temporary files...'
    Remove-Item -Recurse -Force $tempPath
}

# Optional: Cleanup output models
$response = Read-Host 'Remove output models? (y/N)'
if ($response -eq 'y' -or $response -eq 'Y') {
    $outputPath = Join-Path $ScriptDir "outputs"
    if (Test-Path $outputPath) {
        Remove-Item -Recurse -Force $outputPath
        Write-Host 'Output models removed.'
    }
}

Write-Host 'Cleanup complete!'