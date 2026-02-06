# Olive Benchmark and Evaluation System Analysis

## Executive Summary

Olive provides a comprehensive benchmark and evaluation system that supports multiple frameworks (PyTorch, ONNX, OpenVINO, QNN) and measurement types (accuracy, latency, throughput, size on disk, custom metrics). The system is built around the `OliveEvaluator` abstract base class and provides both CLI-based and programmatic evaluation interfaces.

---

## 1. BENCHMARK MODULE/SYSTEM

### 1.1 Location and Structure

**Main Files:**
- **CLI Entry Point:** `D:\Code\Olive\olive\cli\benchmark.py` (133 lines)
- **Benchmark Configuration Template:** Embedded in `BenchmarkCommand` class

**Key Classes:**
- `BenchmarkCommand`: CLI command handler for benchmark operations
- Uses `LMEvaluator` framework for language model evaluation
- Integrates with `lm-eval` library for task-based evaluation

### 1.2 Benchmarking Frameworks Supported

1. **lm-eval** (Language Model Evaluation Harness)
   - File: `olive/evaluator/lmeval_ort.py` (22,602 bytes)
   - Classes: `LMEvalOnnxBase`, `LMEvalORTEvaluator`, `LMEvalORTGenAIEvaluator`
   - Supports HuggingFace models and ONNX Runtime models
   - Integrates with `lm_eval.simple_evaluate()` for standardized benchmarking

2. **Torch Metrics**
   - Built on PyTorch metrics library
   - File: `olive/evaluator/accuracy.py`
   - Supports: accuracy_score, f1_score, precision, recall, auroc, perplexity

### 1.3 Performance Measurements

#### Latency Measurement
- **Sub-types:** AVG, MAX, MIN, P50, P75, P90, P95, P99, P999
- **Unit:** Milliseconds (ms)
- **Configuration:**
  - `warmup_num`: Number of warm-up runs (default: 10)
  - `repeat_test_num`: Number of test repetitions (default: 20)
  - `sleep_num`: Sleep time between runs (default: 0)

**Latency Calculation Flow:**
```
_evaluate_raw_latency() -> latencies (list of floats in seconds)
    ↓
compute_latency() -> OliveEvaluator.latency_helper()
    ↓
Returns: {
    "avg": mean(latencies) * 1000,
    "max": max(latencies) * 1000,
    "min": min(latencies) * 1000,
    "p50": np.percentile(latencies, 50) * 1000,
    ... (percentiles)
}
```

#### Throughput Measurement
- **Sub-types:** AVG, MAX, MIN, P50, P75, P90, P95, P99, P999
- **Unit:** Tokens per second (tps)
- **Formula:** `throughput = batch_size / latency_ms * 1000`
- **Direction:** Higher is better (inversely computed from latencies)

#### Accuracy Measurement
- **Sub-types:** accuracy_score, f1_score, precision, recall, auroc, perplexity
- **Backends:** 
  - torch_metrics (default)
  - huggingface_metrics

#### Size on Disk
- **Sub-type:** bytes
- **Measurement:** Direct model file size
- **Direction:** Lower is better

### 1.4 Main APIs and Entry Points

#### CLI Entry Point
**File:** `olive/cli/benchmark.py`

```python
class BenchmarkCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        # Registers "benchmark" subcommand
        
    def run(self) -> WorkflowOutput:
        # Executes benchmark workflow
        return self._run_workflow()
```

**CLI Arguments:**
```
--tasks: List of tasks to evaluate on (required)
--device: Target device ("cpu" or "gpu", default: "cpu")
--batch_size: Batch size (default: 1)
--max_length: Maximum input + output length (default: 1024)
--limit: Number or percentage of dataset samples (default: 1)
--input-model-path: Path to input model
--output-path: Output directory
```

**Template Configuration:**
```python
TEMPLATE = {
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}]
        }
    },
    "evaluators": {
        "evaluator": {
            "type": "LMEvaluator",
            "tasks": [],
            "batch_size": 16,
            "max_length": 1024,
            "device": "cpu",
            "limit": 64
        }
    },
    "evaluator": "evaluator",
    "host": "local_system",
    "target": "local_system",
    "no_artifacts": True
}
```

---

## 2. EVALUATION MODULE

### 2.1 Location and File Structure

**Main Directory:** `D:\Code\Olive\olive\evaluator\`

**Key Files:**
```
evaluator/
├── __init__.py                      (18 lines - Public API exports)
├── accuracy.py                      (160 lines - Accuracy metric implementations)
├── metric.py                        (223 lines - Metric types and configurations)
├── metric_config.py                 (105 lines - Metric configuration classes)
├── metric_backend.py                (112 lines - Metric computation backends)
├── metric_result.py                 (49 lines - Result data structures)
├── olive_evaluator.py               (1202 lines - Core evaluator implementations)
├── registry.py                      (69 lines - Evaluator registry)
└── lmeval_ort.py                    (22602 bytes - LM evaluation integration)
```

### 2.2 Supported Evaluation Types

#### Metric Types (from `metric.py`)
```python
class MetricType(StrEnumBase):
    ACCURACY = "accuracy"           # Accuracy metrics
    LATENCY = "latency"             # Latency measurements
    THROUGHPUT = "throughput"       # Throughput measurements
    SIZE_ON_DISK = "size_on_disk"   # Model size
    CUSTOM = "custom"               # User-defined metrics
```

#### Accuracy Sub-Types
```python
class AccuracySubType(StrEnumBase):
    ACCURACY_SCORE = "accuracy_score"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    AUROC = "auroc"
    PERPLEXITY = "perplexity"
```

### 2.3 Framework-Specific Evaluators

**Evaluator Class Hierarchy:**
```
OliveEvaluator (ABC)
├── _OliveEvaluator (ABC)
│   ├── OnnxEvaluator          [ONNX models]
│   ├── PyTorchEvaluator       [PyTorch models]
│   ├── OpenVINOEvaluator      [OpenVINO models]
│   ├── QNNEvaluator           [Qualcomm QNN models]
│   └── [Registered via @Registry.register decorator]
└── LMEvaluator                [Language model evaluation]
```

**Registry Registration:**
```python
@Registry.register(str(Framework.ONNX))
@Registry.register("OnnxEvaluator")
class OnnxEvaluator(_OliveEvaluator, OnnxEvaluatorMixin):
    ...
```

### 2.4 Model Loading and Evaluation Flow

#### OnnxEvaluator Flow (Most Comprehensive)

**1. Session Creation:**
```python
def get_session_wrapper(
    model: ONNXModelHandler,
    metric: Metric,
    dataloader: DataLoader,
    device: Device,
    execution_providers: list[str],
) -> tuple[OrtInferenceSession, dict]:
    # Gets inference settings from metric or model
    inference_settings = OnnxEvaluator.get_inference_settings(metric, model)
    
    # Prepares ONNX Runtime session
    session = model.prepare_session(
        inference_settings=inference_settings,
        device=device,
        execution_providers=execution_providers,
    )
    
    # Optionally prepares IO bindings for optimization
    # Loads constant inputs if needed
    
    return OrtInferenceSession(...), inference_settings
```

**2. Accuracy Evaluation:**
```python
def _evaluate_onnx_accuracy(
    model: ONNXModelHandler,
    metric: Metric,
    dataloader: DataLoader,
    post_func=None,
    device: Device = Device.CPU,
    execution_providers: list[str] = None,
) -> MetricResult:
    # Run inference on all batches
    inference_output, targets = self._inference(
        model, metric, dataloader, post_func, device, execution_providers
    )
    
    # Compute accuracy using metric backend
    return OliveEvaluator.compute_accuracy(metric, inference_output, targets)
```

**3. Latency Evaluation:**
```python
def _evaluate_onnx_latency(
    model: ONNXModelHandler,
    metric: Metric,
    dataloader: DataLoader,
    ...
) -> list[float]:
    # Warm up runs
    for _ in range(warmup_num):
        session.run(input_feed)
    
    # Timed runs
    latencies = session.time_run(
        input_feed,
        num_runs=repeat_test_num,
        num_warmup=0,  # Already warmed up
        sleep_time=sleep_num,
    )
    
    return latencies  # List of floats in seconds
```

### 2.5 Metric Configuration and Data Structures

#### Metric Definition
```python
class Metric(NestedConfig):
    name: 
