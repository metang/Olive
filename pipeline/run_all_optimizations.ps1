# Master orchestration script for olive-auto
# Model: microsoft/phi-2
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Check for Hugging Face authentication
if (-not $env:HF_TOKEN) {
    # Check if logged in via huggingface-cli
    $hfTokenFile = Join-Path $env:USERPROFILE ".cache\huggingface\token"
    if (-not (Test-Path $hfTokenFile)) {
        Write-Host '========================================'
        Write-Host 'ERROR: Hugging Face authentication required'
        Write-Host '========================================'
        Write-Host ''
        Write-Host 'onnxruntime-genai requires HF authentication to download models.'
        Write-Host 'Please authenticate using one of these methods:'
        Write-Host ''
        Write-Host '  Option 1: Set HF_TOKEN environment variable'
        Write-Host '    $env:HF_TOKEN = "hf_your_token_here"'
        Write-Host ''
        Write-Host '  Option 2: Login via CLI'
        Write-Host '    huggingface-cli login'
        Write-Host ''
        Write-Host 'Get your token at: https://huggingface.co/settings/tokens'
        Write-Host ''
        exit 1
    }
}

# Create logs directory
New-Item -ItemType Directory -Force -Path "$ScriptDir\logs" | Out-Null

Write-Host '========================================'
Write-Host 'olive-auto Pipeline'
Write-Host '========================================'
Write-Host 'Model: microsoft/phi-2'
Write-Host 'Task: text-generation-with-past'
Write-Host 'Category: nlp'
Write-Host 'Total combinations: 22'
Write-Host '========================================'

# Track results
$Passed = 0
$Failed = 0

# cpu with fp32
Write-Host ''
Write-Host "[$(Get-Date)] Running: cpu fp32"
try {
    & "$ScriptDir\commands\cpu_fp32.ps1"
    Write-Host '[PASS] cpu fp32'
    $Passed++
} catch {
    Write-Host '[FAIL] cpu fp32'
    $Failed++
}

# cpu with fp16
Write-Host ''
Write-Host "[$(Get-Date)] Running: cpu fp16"
try {
    & "$ScriptDir\commands\cpu_fp16.ps1"
    Write-Host '[PASS] cpu fp16'
    $Passed++
} catch {
    Write-Host '[FAIL] cpu fp16'
    $Failed++
}

# cpu with int4
Write-Host ''
Write-Host "[$(Get-Date)] Running: cpu int4"
try {
    & "$ScriptDir\commands\cpu_int4.ps1"
    Write-Host '[PASS] cpu int4'
    $Passed++
} catch {
    Write-Host '[FAIL] cpu int4'
    $Failed++
}

# cuda with fp32
Write-Host ''
Write-Host "[$(Get-Date)] Running: cuda fp32"
try {
    & "$ScriptDir\commands\cuda_fp32.ps1"
    Write-Host '[PASS] cuda fp32'
    $Passed++
} catch {
    Write-Host '[FAIL] cuda fp32'
    $Failed++
}

# cuda with fp16
Write-Host ''
Write-Host "[$(Get-Date)] Running: cuda fp16"
try {
    & "$ScriptDir\commands\cuda_fp16.ps1"
    Write-Host '[PASS] cuda fp16'
    $Passed++
} catch {
    Write-Host '[FAIL] cuda fp16'
    $Failed++
}

# cuda with bf16
Write-Host ''
Write-Host "[$(Get-Date)] Running: cuda bf16"
try {
    & "$ScriptDir\commands\cuda_bf16.ps1"
    Write-Host '[PASS] cuda bf16'
    $Passed++
} catch {
    Write-Host '[FAIL] cuda bf16'
    $Failed++
}

# cuda with int4
Write-Host ''
Write-Host "[$(Get-Date)] Running: cuda int4"
try {
    & "$ScriptDir\commands\cuda_int4.ps1"
    Write-Host '[PASS] cuda int4'
    $Passed++
} catch {
    Write-Host '[FAIL] cuda int4'
    $Failed++
}

# qnn with fp16
Write-Host ''
Write-Host "[$(Get-Date)] Running: qnn fp16"
try {
    & "$ScriptDir\commands\qnn_fp16.ps1"
    Write-Host '[PASS] qnn fp16'
    $Passed++
} catch {
    Write-Host '[FAIL] qnn fp16'
    $Failed++
}

# qnn with int4
Write-Host ''
Write-Host "[$(Get-Date)] Running: qnn int4"
try {
    & "$ScriptDir\commands\qnn_int4.ps1"
    Write-Host '[PASS] qnn int4'
    $Passed++
} catch {
    Write-Host '[FAIL] qnn int4'
    $Failed++
}

# openvino with fp32
Write-Host ''
Write-Host "[$(Get-Date)] Running: openvino fp32"
try {
    & "$ScriptDir\commands\openvino_fp32.ps1"
    Write-Host '[PASS] openvino fp32'
    $Passed++
} catch {
    Write-Host '[FAIL] openvino fp32'
    $Failed++
}

# openvino with fp16
Write-Host ''
Write-Host "[$(Get-Date)] Running: openvino fp16"
try {
    & "$ScriptDir\commands\openvino_fp16.ps1"
    Write-Host '[PASS] openvino fp16'
    $Passed++
} catch {
    Write-Host '[FAIL] openvino fp16'
    $Failed++
}

# openvino with int4
Write-Host ''
Write-Host "[$(Get-Date)] Running: openvino int4"
try {
    & "$ScriptDir\commands\openvino_int4.ps1"
    Write-Host '[PASS] openvino int4'
    $Passed++
} catch {
    Write-Host '[FAIL] openvino int4'
    $Failed++
}

# vitisai with fp32
Write-Host ''
Write-Host "[$(Get-Date)] Running: vitisai fp32"
try {
    & "$ScriptDir\commands\vitisai_fp32.ps1"
    Write-Host '[PASS] vitisai fp32'
    $Passed++
} catch {
    Write-Host '[FAIL] vitisai fp32'
    $Failed++
}

# vitisai with int4
Write-Host ''
Write-Host "[$(Get-Date)] Running: vitisai int4"
try {
    & "$ScriptDir\commands\vitisai_int4.ps1"
    Write-Host '[PASS] vitisai int4'
    $Passed++
} catch {
    Write-Host '[FAIL] vitisai int4'
    $Failed++
}

# webgpu with fp32
Write-Host ''
Write-Host "[$(Get-Date)] Running: webgpu fp32"
try {
    & "$ScriptDir\commands\webgpu_fp32.ps1"
    Write-Host '[PASS] webgpu fp32'
    $Passed++
} catch {
    Write-Host '[FAIL] webgpu fp32'
    $Failed++
}

# webgpu with fp16
Write-Host ''
Write-Host "[$(Get-Date)] Running: webgpu fp16"
try {
    & "$ScriptDir\commands\webgpu_fp16.ps1"
    Write-Host '[PASS] webgpu fp16'
    $Passed++
} catch {
    Write-Host '[FAIL] webgpu fp16'
    $Failed++
}

# webgpu with int4
Write-Host ''
Write-Host "[$(Get-Date)] Running: webgpu int4"
try {
    & "$ScriptDir\commands\webgpu_int4.ps1"
    Write-Host '[PASS] webgpu int4'
    $Passed++
} catch {
    Write-Host '[FAIL] webgpu int4'
    $Failed++
}

# tensorrt with fp32
Write-Host ''
Write-Host "[$(Get-Date)] Running: tensorrt fp32"
try {
    & "$ScriptDir\commands\tensorrt_fp32.ps1"
    Write-Host '[PASS] tensorrt fp32'
    $Passed++
} catch {
    Write-Host '[FAIL] tensorrt fp32'
    $Failed++
}

# tensorrt with fp16
Write-Host ''
Write-Host "[$(Get-Date)] Running: tensorrt fp16"
try {
    & "$ScriptDir\commands\tensorrt_fp16.ps1"
    Write-Host '[PASS] tensorrt fp16'
    $Passed++
} catch {
    Write-Host '[FAIL] tensorrt fp16'
    $Failed++
}

# directml with fp32
Write-Host ''
Write-Host "[$(Get-Date)] Running: directml fp32"
try {
    & "$ScriptDir\commands\directml_fp32.ps1"
    Write-Host '[PASS] directml fp32'
    $Passed++
} catch {
    Write-Host '[FAIL] directml fp32'
    $Failed++
}

# directml with fp16
Write-Host ''
Write-Host "[$(Get-Date)] Running: directml fp16"
try {
    & "$ScriptDir\commands\directml_fp16.ps1"
    Write-Host '[PASS] directml fp16'
    $Passed++
} catch {
    Write-Host '[FAIL] directml fp16'
    $Failed++
}

# rocm with fp32
Write-Host ''
Write-Host "[$(Get-Date)] Running: rocm fp32"
try {
    & "$ScriptDir\commands\rocm_fp32.ps1"
    Write-Host '[PASS] rocm fp32'
    $Passed++
} catch {
    Write-Host '[FAIL] rocm fp32'
    $Failed++
}

# rocm with fp16
Write-Host ''
Write-Host "[$(Get-Date)] Running: rocm fp16"
try {
    & "$ScriptDir\commands\rocm_fp16.ps1"
    Write-Host '[PASS] rocm fp16'
    $Passed++
} catch {
    Write-Host '[FAIL] rocm fp16'
    $Failed++
}

# Print summary
Write-Host ''
Write-Host '========================================'
Write-Host 'Optimization Summary'
Write-Host '========================================'
Write-Host "Passed: $Passed"
Write-Host "Failed: $Failed"
Write-Host ''

if ($Failed -gt 0) {
    exit 1
}