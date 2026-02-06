# Olive-Auto Evaluation Guide

## Overview

The olive-auto evaluation system automatically benchmarks all converted models in a pipeline, measuring:

1. **Accuracy** (Primary metric) - Model prediction quality
2. **Latency** (Secondary) - Inference speed (p50, p99 percentiles)
3. **Memory** (Tertiary) - System memory usage during inference
4. **Model Size** (Lowest priority) - Disk space requirements

This guide explains how to use the auto-generated evaluation script.

---

## Quick Start

### 1. Generate Pipeline with Evaluation Script

```bash
olive-auto -m qwen/qwen3-0.6b -o ./my-pipeline
cd ./my-pipeline
```

This creates `run_evaluation.py` automatically.

### 2. Run Full Evaluation (10% Dataset)

```bash
python run_evaluation.py
```

This evaluates all models using 10% of the dataset (fastest evaluation).

### 3. Run Quick Test (1% Dataset)

```bash
python run_evaluation.py --test
```

Uses only 1% of dataset for very fast evaluation (useful for testing).

### 4. View Results

Results are saved to `evaluation_results/`:

- **results_summary.json** - Full results in JSON format
- **results_summary.csv** - Results in CSV (open in Excel)
- **results_dashboard.html** - Interactive HTML dashboard

Open `results_dashboard.html` in your browser to see charts and recommendations.

---

## Usage Options

### Full Help

```bash
python run_evaluation.py --help
```

### Custom Output Directory

```bash
python run_evaluation.py --output-dir ./my_results
```

### Dataset Sampling

- **Default:** Uses 10% of dataset (faster than full evaluation)
- **Test mode (`--test`):** Uses 1% of dataset (fastest)
- **Automatic:** Adapts based on available memory

---

## What Gets Evaluated

### Models Included

All models from `manifest.json` are evaluated by default:
- All targets: cpu, cuda, openvino, directml, tensorrt, etc.
- All precisions: fp32, fp16, int4, int8, etc.

### Models Skipped

Models are automatically skipped if:
- Model file is corrupted or missing
- Required dataset is unavailable
- Evaluator doesn't support the task type

All skipped models are reported in the results.

---

## Metrics Explained

### Accuracy (lm-eval → perplexity fallback)

**Text-Generation Models:**
- Primary: **lm-eval** - Official language model evaluation framework
  - Evaluates on standard benchmarks (better for comparison)
  - More accurate but slower
- Fallback: **Perplexity** - Used if lm-eval unavailable
  - Faster computation
  - Slightly less standard

**Text-Classification Models:**
- Uses **sklearn metrics**: accuracy, precision, recall, F1-score

### Latency (Inference Speed)

Measured in milliseconds (ms), with percentiles:
- **p50** - Median inference time (50% of inferences faster)
- **p99** - 99th percentile (99% of inferences faster)

Example interpretation:
- p50: 15.2ms = typical inference time
- p99: 45.8ms = worst-case inference time (1% slower than this)

### Memory

Two measurements to avoid surprises:
- **ONNX Runtime Peak** - Memory tracked by ONNX Runtime library
- **System Peak** - Actual system memory usage (usually higher)

Take the **System Peak** as the more reliable value for deployment.

### Model Size

Total size of the ONNX model file on disk in MB.

Includes quantized parameters (int4 < int8 < fp16 < fp32).

---

## Understanding Results

### JSON Results Format

```json
{
  "metadata": {
    "timestamp": "2026-02-05T14:30:00Z",
    "total_models": 8,
    "evaluated_models": 7,
    "skipped_models": 1,
    "sample_fraction": 0.10
  },
  "models": [
    {
      "name": "cpu_fp32",
      "target": "cpu",
      "precision": "fp32",
      "status": "success",
      "metrics": {
        "accuracy": 0.9523,
        "latency_p50_ms": 12.3,
        "latency_p99_ms": 45.1,
        "memory_system_peak_mb": 256.0,
        "model_size_mb": 1024
      }
    }
  ],
  "recommendations": {
    "best_overall": "cpu_fp32",
    "best_accuracy": "cpu_fp32",
    "best_latency": "cuda_int4",
    "best_efficiency": "cuda_int4"
  }
}
```

### CSV Results (Excel)

| model_name | target | precision | accuracy | latency_p50_ms | latency_p99_ms | memory_peak_mb | model_size_mb |
|-----------|--------|-----------|----------|----------------|----------------|----------------|---------------|
| cpu_fp32  | cpu    | fp32      | 0.9523   | 12.3           | 45.1           | 256.0          | 1024          |
| cuda_fp32 | cuda   | fp32      | 0.9523   | 5.1            | 12.3           | 512.0          | 1024          |
| cuda_int4 | cuda   | int4      | 0.9421   | 4.1            | 9.8            | 385.0          | 256           |

---

## Smart Recommendations

The evaluation system automatically recommends models based on different priorities:

### Best Overall
Highest accuracy with reasonable latency (best all-around choice).

### Best Accuracy
Model with highest accuracy (use when precision is critical).

### Best Latency
Fastest model (use when speed is critical).

### Most Efficient
Best accuracy per megabyte of model size (use for deployment constraints).

---

## Performance Expectations

### Evaluation Time

- **Normal mode (10% dataset):** 10-30 min per model
- **Test mode (1% dataset):** 1-5 min per model
- **Sequential evaluation:** One model at a time

Factors affecting time:
- Model size (larger = slower)
- Precision level (fp32 slower than int4)
- Target platform (CUDA faster than CPU)
- Dataset complexity

### Memory Usage

- Evaluator process: 500MB - 2GB
- Model loading: depends on model size
- Dataset caching: typically < 500MB

Total peak: Usually 2-4GB for typical models.

---

## Troubleshooting

### Issue: "Dataset unavailable"

**Cause:** Required evaluation dataset couldn't be downloaded

**Solution:**
1. Check internet connection
2. Try again (dataset may be temporarily unavailable)
3. Check disk space for caching

### Issue: "Model file corrupted"

**Cause:** ONNX model file is invalid

**Solution:**
1. Verify model optimization completed successfully
2. Check `outputs/` directory for errors
3. Try re-generating the pipeline

### Issue: "No evaluator for task"

**Cause:** Task type not supported for evaluation

**Supported:** text-generation, text-classification
**Not supported:** other vision, audio tasks

**Solution:** Manually verify model outputs instead

### Issue: Out of Memory (OOM)

**Cause:** Not enough RAM for evaluation

**Solution:**
1. Close other applications
2. Use smaller dataset: `python run_evaluation.py --test`
3. Evaluate fewer models at once
4. Increase system RAM or swap

### Issue: Evaluation very slow

**Cause:** CPU-only evaluation, large models, or disk I/O

**Solutions:**
1. Use GPU: Set CUDA_VISIBLE_DEVICES if available
2. Run with `--test` for faster iteration
3. Close other processes
4. Consider SSD for dataset caching

---

## Advanced Usage

### Evaluation Configuration

Edit configuration in `run_evaluation.py` if needed:

```python
# Dataset sampling
sample_fraction = 0.10  # 10% of dataset

# Include memory profiling
include_memory_profiling = True

# Output directory
output_dir = Path("evaluation_results")
```

### Using Evaluation Results Programmatically

```python
import json
from pathlib import Path

# Load results
with open("evaluation_results/results_summary.json") as f:
    results = json.load(f)

# Access metadata
print(f"Evaluated: {results['metadata']['evaluated_models']} models")
print(f"Total time: {results['metadata']['total_time_sec']}s")

# Find best model
best = results['recommendations']['best_overall']
for model in results['models']:
    if model['name'] == best:
        print(f"Best model: {best}")
        print(f"Accuracy: {model['metrics']['accuracy']:.4f}")
        print(f"Latency p50: {model['metrics']['latency_p50_ms']:.2f}ms")
```

### Integration with CI/CD

```bash
#!/bin/bash
# Run evaluation and fail if accuracy < threshold
python run_evaluation.py --test

ACCURACY=$(jq '.models[0].metrics.accuracy' evaluation_results/results_summary.json)
if (( $(echo "$ACCURACY < 0.90" | bc -l) )); then
  echo "Error: Accuracy below 0.90"
  exit 1
fi
```

---

## Output Files

### evaluation_results/ Directory Structure

```
evaluation_results/
├── results_summary.json           # Full results (machine-readable)
├── results_summary.csv            # Results for Excel (human-readable)
├── results_dashboard.html         # Interactive charts & recommendations
└── per_model/                     # Per-model details (optional)
    ├── cpu_fp32/
    │   ├── metrics_detailed.json
    │   └── inference_log.txt
    └── cuda_int4/
        ├── metrics_detailed.json
        └── inference_log.txt
```

### JSON Schema

See `results_summary.json` for full schema. Key sections:

- **metadata** - Evaluation config & timing
- **models** - Per-model results array
- **recommendations** - Best model for each use case

### CSV Columns

All metrics from `results_summary.json` converted to CSV format for Excel import.

---

## Next Steps

### After Getting Results

1. **Review Recommendations** - Check HTML dashboard for smart picks
2. **Compare Metrics** - Use CSV in Excel for custom analysis
3. **Validate** - Test recommended model in your application
4. **Deploy** - Use best model for your use case

### For Iterative Optimization

```bash
# Run quick test
python run_evaluation.py --test

# Review recommendations
open evaluation_results/results_dashboard.html

# Adjust pipeline if needed
cd ..
olive-auto -m [model] -o ./pipeline-v2 --targets [new-targets]

# Evaluate again
cd pipeline-v2
python run_evaluation.py
```

---

## Support & Issues

For evaluation-related issues:

1. Check logs in `evaluation_results/`
2. Review error messages in script output
3. Try `python run_evaluation.py --help`
4. Check dataset availability: `python run_evaluation.py --test`

Common issues are documented in the **Troubleshooting** section above.

---

## Reference

### Accuracy Methods

| Task | Primary | Fallback | Speed | Accuracy |
|------|---------|----------|-------|----------|
| text-generation | lm-eval | perplexity | slow | high |
| text-classification | sklearn | (none) | fast | high |

### Metrics Priority

1. **Accuracy** - Most important (model quality)
2. **Latency** - Deployment speed
3. **Memory** - Resource constraints
4. **Size** - Storage/bandwidth

### Dataset Sampling

| Mode | Samples | Time | Use Case |
|------|---------|------|----------|
| Normal | 10% | 10-30min | Production evaluation |
| Test | 1% | 1-5min | Quick iteration |
| None | 100% | 1-2hrs | Research/reference |

---

**Last Updated:** 2026-02-05
**Olive-Auto Version:** 0.1.0+
