# Fixes Applied to olive-auto

## Summary
Fixed multiple issues with the olive-auto pipeline generator that were preventing proper model optimization and output generation.

## Issues Fixed

### 1. Path Handling (FIXED)
**Problem**: Generated scripts used relative paths like `test_pipeline\outputs\cpu_fp32` which were relative to the script directory, causing them to create nested paths like `test_pipeline\test_pipeline\outputs\cpu_fp32` instead of writing to the intended location.

**Solution**: Modified `command_generator.py` to convert all output, log, and environment paths to absolute paths using `.resolve()`. This ensures scripts write to the correct directories regardless of where they're executed from.

**Changes in `command_generator.py`**:
- `_generate_bash_script()`: Now converts paths to absolute paths
- `_generate_powershell_script()`: Now converts paths to absolute paths
- Both methods now pass absolute paths to PowerShell/Bash variables

### 2. Error Handling and Debugging (FIXED)
**Problem**: When optimization failed, there was minimal error information. The scripts didn't capture stderr properly or provide feedback about whether models were produced.

**Solution**: Added comprehensive error handling and feedback:

**For PowerShell scripts**:
- Wrapped olive optimize command in try-catch block
- Added file verification after optimization (checks if output files were created)
- Added explicit error messages when optimization fails
- Display output directory path and log file path for debugging

**For Bash scripts**:
- Wrapped optimization in error handling block
- Capture command exit status separately
- Check if output files exist after optimization
- Display file count when successful

### 3. Missing Dependency (FIXED)
**Problem**: The `onnxruntime-genai` package is required for generative AI models like qwen3-0.6b but was missing from the environment configuration.

**Solution**: Updated `environment.py` to include:
- `onnxruntime-genai` for CPU target
- `onnxruntime-genai-cuda` for CUDA target
- `onnxruntime-genai-directml` for DirectML target

### 4. Generated Scripts Improvements
The new generated scripts now include:
- Absolute paths for all directories (output, logs, environments)
- Log directory creation before running optimization
- Output directory creation with explicit verification
- Try-catch error handling with informative messages
- Post-optimization verification that outputs were produced
- Clear display of all paths for debugging

## Files Modified
1. `olive_auto/command_generator.py` - Path handling and error handling improvements
2. `olive_auto/environment.py` - Added onnxruntime-genai dependencies

## Testing
Generated a new test pipeline with fixed code:
```bash
olive-auto -m qwen/qwen3-0.6b -t cpu -o test_pipeline --platform windows
```

Generated scripts now have:
- Absolute paths (verified in test_pipeline/commands/cpu_fp32.ps1)
- Comprehensive error handling (try-catch blocks)
- Output file verification
- Better logging and debugging information

## Example Generated Script
The new `cpu_fp32.ps1` now includes:
```powershell
$LogFile = "D:\Code\Olive\olive_auto\test_pipeline\logs\cpu_fp32.log"
$OutputDir = "D:\Code\Olive\olive_auto\test_pipeline\outputs\cpu_fp32"
$EnvDir = "D:\Code\Olive\olive_auto\test_pipeline\environments\olive_cpu"

# ... 

try {
    olive optimize `
        -m "qwen/qwen3-0.6b" `
        --task text-generation-with-past `
        --provider CPUExecutionProvider `
        --precision fp32 `
        -o $OutputDir 2>&1 | Tee-Object -FilePath $LogFile
    
    # Check if output was produced
    $outputFiles = @(Get-ChildItem -Path $OutputDir -File -Recurse -ErrorAction SilentlyContinue)
    if ($outputFiles.Count -eq 0) {
        Write-Host 'WARNING: No output files produced by optimization. Check log for details.'
    } else {
        Write-Host "Successfully produced $($outputFiles.Count) output file(s)"
    }
} catch {
    Write-Host "ERROR: Optimization failed with exception: $_"
    throw
}
```

## Next Steps
The pipeline is now ready with proper error handling and path management. When models don't produce output, users will see clear diagnostic messages about:
- Log file location for investigation
- Output directory path
- Whether any files were produced
- Any exceptions that occurred during optimization
