# Master setup script for all olive-auto environments
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host '========================================'
Write-Host 'Setting up all olive-auto environments'
Write-Host '========================================'

# Setup cpu
Write-Host ''
Write-Host 'Setting up cpu...'
& "$ScriptDir\environments\setup_cpu.ps1"

# Setup cuda
Write-Host ''
Write-Host 'Setting up cuda...'
& "$ScriptDir\environments\setup_cuda.ps1"

# Setup qnn
Write-Host ''
Write-Host 'Setting up qnn...'
& "$ScriptDir\environments\setup_qnn.ps1"

# Setup openvino
Write-Host ''
Write-Host 'Setting up openvino...'
& "$ScriptDir\environments\setup_openvino.ps1"

# Setup vitisai
Write-Host ''
Write-Host 'Setting up vitisai...'
& "$ScriptDir\environments\setup_vitisai.ps1"

# Setup webgpu
Write-Host ''
Write-Host 'Setting up webgpu...'
& "$ScriptDir\environments\setup_webgpu.ps1"

# Setup tensorrt
Write-Host ''
Write-Host 'Setting up tensorrt...'
& "$ScriptDir\environments\setup_tensorrt.ps1"

# Setup directml
Write-Host ''
Write-Host 'Setting up directml...'
& "$ScriptDir\environments\setup_directml.ps1"

# Setup rocm
Write-Host ''
Write-Host 'Setting up rocm...'
& "$ScriptDir\environments\setup_rocm.ps1"

Write-Host ''
Write-Host '========================================'
Write-Host 'All environments setup complete!'
Write-Host '========================================'