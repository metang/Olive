# Olive (ONNX LIVE) Architecture Document

## 1. Project Overview

**Olive** is Microsoft's AI Model Optimization Toolkit for ONNX Runtime. It automates the optimization of machine learning models for efficient inference on various hardware targets (CPU, GPU, NPU). Given a model and target hardware, Olive composes the best optimization techniques to produce efficient ONNX models while respecting constraints like accuracy and latency.

**Package Name**: `olive-ai`
**Python**: >=3.10
**License**: MIT

## 2. High-Level Architecture

```
+------------------------------------------------------------------+
|                         CLI Layer                                 |
|  olive optimize | olive finetune | olive quantize | olive run    |
+------------------------------------------------------------------+
                                |
+------------------------------------------------------------------+
|                      Workflow Layer                               |
|              RunConfig -> Engine Orchestration                    |
+------------------------------------------------------------------+
                                |
+------------------------------------------------------------------+
|                       Engine Core                                 |
|  Engine -> Pass Execution -> Evaluation -> Search Strategy        |
+------------------------------------------------------------------+
                                |
        +-----------------------+-----------------------+
        |                       |                       |
+-------v-------+      +--------v------+      +--------v------+
|    Passes     |      |    Models     |      |   Systems     |
| (Transforms)  |      |  (Handlers)   |      |  (Targets)    |
+---------------+      +---------------+      +---------------+
        |                       |                       |
+-------v-------+      +--------v------+      +--------v------+
|     Data      |      |   Hardware    |      |    Cache      |
| (Containers)  |      | (Accelerators)|      |  (Results)    |
+---------------+      +---------------+      +---------------+
```

## 3. Directory Structure

```
olive/
|-- __init__.py           # Version info
|-- __main__.py           # Entry point
|-- olive_config.json     # Pass registry & dependencies
|
|-- cli/                  # Command-line interface
|   |-- launcher.py       # Main CLI entry point
|   |-- auto_opt.py       # Automatic optimization command
|   |-- optimize.py       # Manual optimization command
|   |-- finetune.py       # Fine-tuning command
|   |-- quantize.py       # Quantization command
|   |-- run.py            # Workflow run command
|   +-- ...               # Other CLI commands
|
|-- engine/               # Workflow orchestration
|   |-- engine.py         # Core Engine class
|   |-- config.py         # Engine configuration
|   |-- footprint.py      # Run tracking/history
|   |-- output.py         # Workflow output handling
|   +-- packaging/        # Model packaging utilities
|
|-- model/                # Model abstractions
|   |-- handler/          # Model type handlers
|   |   |-- base.py       # OliveModelHandler ABC
|   |   |-- hf.py         # HuggingFace models
|   |   |-- onnx.py       # ONNX models
|   |   |-- pytorch.py    # PyTorch models
|   |   +-- ...           # Other model types
|   |-- config/           # Model configuration
|   +-- utils/            # Model utilities
|
|-- passes/               # Optimization passes
|   |-- olive_pass.py     # Base Pass class
|   |-- pass_config.py    # Pass configuration
|   |-- onnx/             # ONNX-specific passes
|   |   |-- conversion.py
|   |   |-- quantization.py
|   |   |-- transformer_optimization.py
|   |   +-- ...
|   |-- pytorch/          # PyTorch passes
|   |   |-- lora.py       # LoRA/QLoRA/DoRA
|   |   |-- gptq.py       # GPTQ quantization
|   |   +-- ...
|   |-- openvino/         # OpenVINO passes
|   |-- qnn/              # Qualcomm QNN passes
|   +-- diffusers/        # Diffusion model passes
|
|-- evaluator/            # Model evaluation
|   |-- olive_evaluator.py
|   |-- metric.py         # Metric definitions
|   +-- metric_result.py  # Evaluation results
|
|-- search/               # Hyperparameter search
|   |-- search_strategy.py
|   |-- search_space.py
|   |-- search_sample.py
|   +-- samplers/         # Search algorithms
|
|-- systems/              # Execution environments
|   |-- olive_system.py   # Base OliveSystem
|   |-- local.py          # Local execution
|   |-- docker/           # Docker execution
|   +-- python_environment/
|
|-- data/                 # Data handling
|   |-- config.py         # Data configuration
|   |-- container/        # Data containers
|   +-- component/        # Data components
|
|-- hardware/             # Hardware abstraction
|   +-- accelerator.py    # AcceleratorSpec
|
|-- cache.py              # Caching system
|-- common/               # Shared utilities
|-- workflows/            # Workflow configs
+-- platform_sdk/         # Platform-specific SDKs
    +-- qualcomm/         # Qualcomm tools
```

## 4. Core Components

### 4.1 Engine (`olive/engine/engine.py`)

The **Engine** is the central orchestrator that:
- Registers and executes optimization passes
- Manages model caching and evaluation
- Implements search strategies for hyperparameter tuning
- Tracks run history via Footprint

```python
class Engine:
    def __init__(self, ...):
        self.cache: OliveCache           # Caching system
        self.search_strategy: SearchStrategy  # Optional search
        self.footprint: Footprint        # Run tracking

    def register(self, pass_type, config, name, ...):
        """Register a pass for execution"""

    def run(self, input_model_config, accelerator_spec, ...):
        """Execute all registered passes"""
```

**Key Methods**:
- `register()`: Register optimization passes
- `run()`: Execute the optimization workflow
- `_run_pass()`: Execute a single pass
- `_evaluate_model()`: Evaluate model against metrics

### 4.2 Pass System (`olive/passes/`)

All optimizations are implemented as **Passes** - modular transformations that take a model and produce an optimized model.

**Base Class** (`olive/passes/olive_pass.py:40`):
```python
class Pass(ABC):
    registry: ClassVar[dict[str, type["Pass"]]] = {}  # Auto-registration

    @abstractmethod
    def _default_config(cls, accelerator_spec) -> dict[str, PassConfigParam]:
        """Define pass parameters"""

    @abstractmethod
    def _run_for_config(self, model, config, output_path) -> OliveModelHandler:
        """Execute the pass"""
```

**Pass Categories** (60+ passes):

| Category | Examples | Purpose |
|----------|----------|---------|
| **ONNX Conversion** | `OnnxConversion`, `OptimumConversion` | Convert models to ONNX |
| **Quantization** | `OnnxStaticQuantization`, `Gptq`, `AutoAWQ` | Reduce precision |
| **Optimization** | `OrtTransformersOptimization`, `OnnxPeepholeOptimizer` | Graph optimization |
| **Fine-tuning** | `LoRA`, `QLoRA`, `DoRA` | Parameter-efficient training |
| **Mixed Precision** | `OrtMixedPrecision`, `OnnxFloatToFloat16` | FP16/BF16 conversion |
| **Hardware-specific** | `EPContextBinaryGenerator`, `QNNConversion` | NPU/QNN targets |

### 4.3 Model Handlers (`olive/model/handler/`)

Model handlers provide a unified interface for different model formats.

**Base Class** (`olive/model/handler/base.py:15`):
```python
class OliveModelHandler(ABC, ResourceMixin, IoConfigMixin, JsonMixin):
    @abstractmethod
    def load_model(self, rank=None) -> object:
        """Load model into memory"""

    @abstractmethod
    def prepare_session(self, inference_settings, device, ...) -> Any:
        """Create inference session"""

    @abstractmethod
    def run_session(self, session, inputs, ...) -> Any:
        """Run inference"""
```

**Supported Model Types**:
- `HfModelHandler` - HuggingFace Transformers models
- `ONNXModelHandler` - ONNX format models
- `PyTorchModelHandler` - PyTorch nn.Module
- `OpenVINOModelHandler` - OpenVINO IR models
- `QNNModelHandler` - Qualcomm QNN models
- `CompositeModelHandler` - Multi-component models
- `DistributedOnnxModelHandler` - Distributed models

### 4.4 Systems (`olive/systems/`)

Systems define where model optimization and evaluation occur.

**Base Class** (`olive/systems/olive_system.py`):
```python
class OliveSystem(ABC):
    @abstractmethod
    def run_pass(self, the_pass, model_config, output_path) -> ModelConfig:
        """Execute a pass"""

    @abstractmethod
    def evaluate_model(self, model_config, evaluator_config, accelerator) -> MetricResult:
        """Evaluate a model"""
```

**System Types**:
- `LocalSystem` - Execute on local machine (`olive/systems/local.py`)
- `DockerSystem` - Execute in Docker container
- `PythonEnvironmentSystem` - Execute in specific Python environment

### 4.5 Search Strategy (`olive/search/`)

For hyperparameter optimization, Olive implements search strategies:

```python
class SearchStrategy:
    """Iterates through search space, collecting feedback signals"""

class SearchStrategyConfig:
    execution_order: str      # "joint" or "pass-by-pass"
    sampler: str              # Search algorithm
    output_model_num: int     # Number of output models
    stop_when_goals_met: bool
    max_iter: int
```

**Samplers**: Random, Grid, TPE (Tree-Parzen Estimator), etc.

### 4.6 Evaluator (`olive/evaluator/`)

Evaluators measure model quality against defined metrics:

```python
class OliveEvaluator(ABC):
    @abstractmethod
    def evaluate(self, model, metrics, device, execution_providers) -> MetricResult:
        """Evaluate model against metrics"""
```

**Metric Types**:
- `Latency` - Inference latency
- `Throughput` - Tokens/samples per second
- `Accuracy` - Task-specific accuracy
- `SizeOnDisk` - Model file size

### 4.7 Data System (`olive/data/`)

Handles dataset loading and preprocessing:

```python
class DataConfig:
    name: str
    type: str                    # Container type
    load_dataset_config: dict    # Dataset loading params
    pre_process_data_config: dict
    post_process_data_config: dict
```

**Data Containers**:
- `HuggingfaceContainer` - HuggingFace datasets
- `DummyDataContainer` - Synthetic data for calibration
- `RawDataContainer` - Custom data
- `ImageDataContainer` - Image datasets

### 4.8 Cache System (`olive/cache.py`)

Caches intermediate results for faster reruns:

```python
class OliveCache:
    def cache_model(self, model_id, model_json): ...
    def load_model(self, model_id) -> dict: ...
    def cache_run(self, pass_type, config, input_id, output_id): ...
    def cache_evaluation(self, model_id, evaluation_json): ...
```

**Cache Structure**:
```
.olive-cache/
|-- runs/           # Pass execution results
|-- evaluations/    # Evaluation results
|-- resources/      # Downloaded resources
+-- mlflow/         # MLflow artifacts
```

## 5. CLI Architecture

Entry point: `olive/cli/launcher.py`

```python
def main():
    parser = get_cli_parser()
    # Register all commands
    WorkflowRunCommand.register_subcommand(commands_parser)
    OptimizeCommand.register_subcommand(commands_parser)
    FineTuneCommand.register_subcommand(commands_parser)
    # ... more commands
```

**Available Commands**:

| Command | Purpose |
|---------|---------|
| `olive run` | Run workflow from JSON/YAML config |
| `olive optimize` | High-level optimization command |
| `olive auto-opt` | Automatic optimization |
| `olive finetune` | Fine-tune models |
| `olive quantize` | Quantize models |
| `olive capture-onnx-graph` | Export to ONNX |
| `olive benchmark` | Benchmark models |
| `olive run-pass` | Run single pass |

## 6. Configuration System

### 6.1 Workflow Configuration (`olive/workflows/run/config.py`)

```python
class RunConfig:
    workflow_id: str
    input_model: ModelConfig      # Input model
    systems: dict[str, SystemConfig]
    data_configs: list[DataConfig]
    evaluators: dict[str, OliveEvaluatorConfig]
    engine: RunEngineConfig
    passes: dict[str, list[RunPassConfig]]
```

### 6.2 Pass Registry (`olive/olive_config.json`)

All 60+ passes are registered with metadata:
```json
{
    "passes": {
        "OnnxConversion": {
            "module_path": "olive.passes.onnx.conversion.OnnxConversion",
            "supported_providers": ["*"],
            "supported_precisions": ["*"],
            "supported_algorithms": []
        },
        "Gptq": {
            "module_path": "olive.passes.pytorch.gptq.Gptq",
            "supported_precisions": ["int4", "int8"],
            "supported_algorithms": ["gptq", "quarot", "spinquant"],
            "dataset": "dataset_optional"
        }
    },
    "extra_dependencies": {
        "gpu": ["onnxruntime-gpu"],
        "lora": ["accelerate", "peft", "scipy"]
    }
}
```

## 7. Data Flow

```
+----------------+
|  Input Model   | (HuggingFace, PyTorch, ONNX)
+-------+--------+
        |
        v
+----------------+     +-----------------+
|    Engine      |---->|  Pass Registry  |
+-------+--------+     +-----------------+
        |
        v
+----------------+
|   Pass 1       | (e.g., OnnxConversion)
|  ---------     |
|  Input Model   |---> Output Model 1
+-------+--------+
        |
        v
+----------------+
|   Pass 2       | (e.g., Gptq)
|  ---------     |
|  Output 1      |---> Output Model 2
+-------+--------+
        |
        v
+----------------+
|   Evaluator    |---> MetricResult
+-------+--------+
        |
        v
+----------------+
| Output Model   | (Optimized ONNX)
+----------------+
```

## 8. Extension Points

### 8.1 Adding a New Pass

1. Create a class extending `Pass`:
```python
class MyPass(Pass):
    @classmethod
    def _default_config(cls, accelerator_spec):
        return {"param1": PassConfigParam(type_=int, default_value=1)}

    def _run_for_config(self, model, config, output_path):
        # Implement optimization logic
        return output_model
```

2. Register in `olive_config.json`

### 8.2 Adding a New Model Handler

Extend `OliveModelHandler` and implement:
- `load_model()`
- `prepare_session()`
- `run_session()`

### 8.3 Adding a New System

Extend `OliveSystem` and implement:
- `run_pass()`
- `evaluate_model()`
- `get_supported_execution_providers()`

## 9. Key Design Patterns

1. **Registry Pattern**: Passes auto-register via `__init_subclass__`
2. **Strategy Pattern**: Search strategies are pluggable
3. **Template Method**: Pass base class defines workflow, subclasses implement specifics
4. **Mixin Pattern**: Model handlers use mixins for I/O, JSON, and resource handling
5. **Configuration-Driven**: All workflows defined via JSON/YAML configs

## 10. Testing Structure

```
test/
|-- cli/              # CLI command tests
|-- common/           # Utility tests
|-- data_container/   # Data handling tests
|-- engine/           # Engine tests
|-- evaluator/        # Evaluator tests
|-- model/            # Model handler tests
|-- passes/           # Pass-specific tests
|   |-- onnx/
|   |-- pytorch/
|   +-- ...
+-- conftest.py       # Pytest fixtures
```

## 11. Key Dependencies

- **onnxruntime**: ONNX model inference
- **torch**: PyTorch model support
- **transformers**: HuggingFace model loading
- **optimum**: Model export/optimization
- **peft**: Parameter-efficient fine-tuning
- **pydantic**: Configuration validation

---

This architecture enables Olive to be a flexible, extensible toolkit for ML model optimization across diverse hardware targets.

---

## 12. Deep Dive: `olive optimize` Command

The `olive optimize` command is the high-level CLI for comprehensive model optimization with intelligent pass scheduling based on target hardware and precision requirements.

### 12.1 Command Registration

**Entry Point**: `olive/cli/launcher.py:42`
```python
OptimizeCommand.register_subcommand(commands_parser)
```

**Command Class**: `olive/cli/optimize.py:25`
```python
class OptimizeCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "optimize",
            help="Optimize the input model with comprehensive pass scheduling",
        )
```

### 12.2 Command Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `-m, --model_name_or_path` | str | Required | Input model (HF name, local path, or AzureML registry) |
| `-o, --output_path` | str | `optimized-model` | Output directory for optimized model |
| `--provider` | choice | `CPUExecutionProvider` | Target execution provider |
| `--device` | choice | None | Target device: `cpu`, `gpu`, `npu` |
| `--precision` | choice | `fp32` | Target precision (int4, int8, fp16, fp32, etc.) |
| `--act_precision` | choice | None | Activation precision for quantization |
| `--exporter` | choice | `model_builder` | ONNX exporter to use |
| `--num_split` | int | None | Number of model splits |
| `--memory` | int | None | Available device memory (MB) |
| `--block_size` | int | None | Quantization block size (-1 for per-channel) |
| `--use_qdq_format` | flag | False | Use QDQ quantization format |
| `--surgeries` | list | None | Graph surgery operations |
| `--enable_aot` | flag | False | Enable Ahead-of-Time compilation |
| `--qnn_env_path` | str | None | QNN environment path (for AOT) |
| `--extra_mb_options` | str | None | Extra model builder options |
| `--dry_run` | flag | False | Generate config without running |
| `--save_config_file` | flag | False | Save generated config to file |

**Supported Execution Providers**:
- `CPUExecutionProvider`
- `CUDAExecutionProvider`
- `QNNExecutionProvider`
- `VitisAIExecutionProvider`
- `OpenVINOExecutionProvider`
- `WebGpuExecutionProvider`
- `NvTensorRTRTXExecutionProvider`

**Supported Precisions**:
- Integer: `int4`, `int8`, `int16`, `int32`, `uint4`, `uint8`, `uint16`, `uint32`
- Floating: `fp16`, `fp32`, `bf16`

### 12.3 Execution Flow

```
olive optimize --model_name_or_path <model> --precision int4 --provider CUDAExecutionProvider
        |
        v
+------------------+
| OptimizeCommand  |
|   __init__()     |  Initialize pass enable flags
+--------+---------+
         |
         v
+------------------+
|    run()         |
|  _run_workflow() |  (inherited from BaseOliveCLICommand)
+--------+---------+
         |
         v
+------------------+
| _get_run_config()|  Build configuration dict
+--------+---------+
         |
    +----+----+
    |         |
    v         v
+--------+ +------------------+
|Validate| |_build_passes_   |
|  Args  | |     config()    |
+--------+ +--------+---------+
                    |
                    v
           +------------------+
           | Conditional Pass |
           |    Selection     |
           +--------+---------+
                    |
                    v
           +------------------+
           | olive.workflows  |
           |    .run()        |
           +------------------+
```

### 12.4 Pass Selection Logic

The `_build_passes_config()` method (`olive/cli/optimize.py:305`) conditionally enables passes based on:
- Input model type (HuggingFace vs ONNX)
- Target execution provider
- Target precision
- User-specified options

**Pass Decision Matrix**:

| Pass | Condition |
|------|-----------|
| **QuaRot** | HF model + int4/uint4 + (QNN or VitisAI EP) |
| **Gptq** | HF model + int4/uint4 + not OpenVINO |
| **CaptureSplitInfo** | HF model + (num_split or memory specified) |
| **ModelBuilder** | HF model + not OpenVINO + exporter=model_builder |
| **OnnxConversion** | HF model + not OpenVINO + (dynamo or torchscript exporter) |
| **OpenVINOOptimumConversion** | HF model + OpenVINO EP |
| **DynamicToFixedShape** | (QNN or VitisAI or NPU) + dim_param + dim_value |
| **OnnxPeepholeOptimizer** | exporter != model_builder |
| **OrtTransformersOptimization** | torchscript or dynamo exporter + not NvTensorRT |
| **MatMulNBitsToQDQ** | HF model + GPTQ enabled + use_qdq_format |
| **GraphSurgeries** | surgeries argument provided |
| **OnnxBlockWiseRtnQuantization** | ONNX model (not HF) + int4 precision |
| **OnnxFloatToFloat16** | fp16 precision |
| **OnnxStaticQuantization** | int8/uint8/int16/uint16 precision (weight or activation) |
| **SplitModel** | HF model + (num_split or memory specified) |
| **StaticLLM** | text modality + (QNN or VitisAI EP) |
| **VitisAIAddMetaData** | VitisAI EP |
| **EPContextBinaryGenerator** | enable_aot + QNN EP |
| **ComposeOnnxModels** | HF model + enable_aot + split + QNN EP |
| **OpenVINOEncapsulation** | HF model + OpenVINO EP |
| **OnnxIODataTypeConverter** | WebGPU EP |

### 12.5 Pass Configuration Examples

**GPTQ Pass** (`olive/cli/optimize.py:426`):
```python
def _get_gptq_pass_config(self) -> dict[str, Any]:
    precision = Precision(self.args.precision)
    bits = precision_bits_from_precision(precision).value
    return {
        "type": "Gptq",
        "bits": bits,                    # 4 for int4
        "sym": precision == Precision.INT4,  # symmetric quantization
        "group_size": self.args.block_size or default
    }
```

**ModelBuilder Pass** (`olive/cli/optimize.py:464`):
```python
def _get_model_builder_pass_config(self) -> dict[str, Any]:
    config = {
        "type": "ModelBuilder",
        "precision": precision.value,
        "int4_block_size": 32,      # Valid: 16, 32, 64, 128, 256
        "int4_accuracy_level": 4,
        "int4_op_types_to_quantize": ["MatMul", "Gather"],
        "extra_options": {...}
    }
    return config
```

### 12.6 Input Model Resolution

The `get_input_model_config()` function (`olive/cli/base.py:203`) resolves model paths in this order:

1. **No model path + model_script** -> PyTorch model with custom loader
2. **Previous Olive output** -> Load `model_config.json` from directory
3. **model_script provided** -> PyTorch model
4. **AzureML Registry** -> `azureml://registries/<reg>/models/<name>/versions/<ver>`
5. **HuggingFace URL** -> `https://huggingface.co/<model>`
6. **HuggingFace ID** -> Validate via `hf_repo_exists()`
7. **Local ONNX file/folder** -> Check for `.onnx` extension
8. **Local HF model folder** -> Fallback for local directories

### 12.7 Generated Configuration Structure

```python
TEMPLATE = {
    "input_model": {"type": "HfModel"},
    "passes": OrderedDict(),  # Passes in execution order
    "systems": {},
    "no_artifacts": True,
}
```

**Example generated config for `olive optimize -m Qwen/Qwen2.5-0.5B --precision int4`**:
```json
{
    "input_model": {
        "type": "HfModel",
        "model_path": "Qwen/Qwen2.5-0.5B",
        "load_kwargs": {"attn_implementation": "eager"}
    },
    "passes": {
        "gptq": {"type": "Gptq", "bits": 4, "sym": true},
        "model_builder": {
            "type": "ModelBuilder",
            "precision": "int4",
            "int4_block_size": 32,
            "int4_accuracy_level": 4
        }
    },
    "output_dir": "optimized-model",
    "log_severity_level": 3,
    "no_artifacts": true
}
```

### 12.8 Validation Rules

**Argument Validation** (`olive/cli/optimize.py:253`):

| Rule | Error |
|------|-------|
| CPU EP + GPU/NPU device | Invalid combination |
| CUDA EP + CPU/NPU device | Invalid combination |
| NvTensorRT EP + CPU/NPU device | Invalid combination |
| enable_aot + not QNN EP | AOT only supports QNN |
| enable_aot + no qnn_env_path | QNN path required for AOT |
| use_qdq_format + OpenVINO EP | QDQ not supported with OpenVINO |

### 12.9 Data Configuration

For static quantization with text modality, WikiText-2 dataset is automatically configured:

```python
WIKITEXT2_DATA_CONFIG_TEMPLATE = [{
    "name": "wikitext2_train",
    "type": "HuggingfaceContainer",
    "load_dataset_config": {
        "data_name": "Salesforce/wikitext",
        "subset": "wikitext-2-raw-v1",
        "split": "train"
    },
    "pre_process_data_config": {
        "strategy": "line-by-line",
        "add_special_tokens": False,
        "max_samples": 128,
        "max_seq_len": 512,
    },
}]
```

### 12.10 Workflow Execution

After configuration is built, `_run_workflow()` (`olive/cli/base.py:31`) executes:

```python
def _run_workflow(self):
    with tempfile.TemporaryDirectory(...) as tempdir:
        run_config = self._get_run_config(tempdir)

        if self.args.save_config_file or self.args.dry_run:
            self._save_config_file(run_config)  # Save to output_dir/config.json

        if self.args.dry_run:
            return None  # Exit without running

        workflow_output = olive_run(run_config)  # Execute olive.workflows.run()
        return workflow_output
```

### 12.11 Common Usage Patterns

**Basic INT4 quantization**:
```bash
olive optimize -m microsoft/Phi-3-mini-4k-instruct --precision int4
```

**CUDA with FP16**:
```bash
olive optimize -m meta-llama/Llama-2-7b-hf \
    --provider CUDAExecutionProvider \
    --precision fp16
```

**QNN NPU with AOT compilation**:
```bash
olive optimize -m Qwen/Qwen2.5-0.5B \
    --provider QNNExecutionProvider \
    --precision int4 \
    --enable_aot \
    --qnn_env_path /path/to/qnn/env
```

**Model splitting for memory-constrained devices**:
```bash
olive optimize -m microsoft/phi-2 \
    --precision int4 \
    --memory 4096 \
    --num_split 2
```

**OpenVINO optimization**:
```bash
olive optimize -m meta-llama/Llama-2-7b-hf \
    --provider OpenVINOExecutionProvider \
    --precision int4 \
    --device cpu
```

**Dry run to generate config**:
```bash
olive optimize -m Qwen/Qwen2.5-0.5B \
    --precision int4 \
    --dry_run \
    --save_config_file
```

### 12.12 Class Hierarchy

```
BaseOliveCLICommand (olive/cli/base.py:21)
    |
    +-- OptimizeCommand (olive/cli/optimize.py:25)
    +-- FineTuneCommand
    +-- QuantizeCommand
    +-- AutoOptCommand
    +-- ...
```

**Key Methods**:
- `register_subcommand()`: Static method to register CLI args
- `__init__()`: Initialize pass enable flags
- `run()`: Entry point, calls `_run_workflow()`
- `_get_run_config()`: Build the Olive workflow configuration
- `_validate_arguments()`: Validate CLI argument combinations
- `_build_passes_config()`: Conditionally select and configure passes
- `_enable_<pass>_pass()`: Check if pass should be enabled
- `_get_<pass>_pass_config()`: Generate pass configuration dict

---

## 13. Model Type Support

### 13.1 Supported Model Types (10 Handlers)

Olive supports 10 model handler types, registered in `olive/model/config/registry.py`:

| CLI Name | Handler Class | Location | Description |
|----------|---------------|----------|-------------|
| `HfModel` | `HfModelHandler` | `olive/model/handler/hf.py:29` | HuggingFace Transformers models |
| `OnnxModel` | `ONNXModelHandler` | `olive/model/handler/onnx.py:29` | ONNX format models |
| `PyTorchModel` | `PyTorchModelHandler` | `olive/model/handler/pytorch.py:113` | Custom PyTorch nn.Module |
| `DiffusersModel` | `DiffusersModelHandler` | `olive/model/handler/diffusers.py:24` | Diffusers pipelines (SD, SDXL, SD3, Flux, Sana) |
| `OpenVINOModel` | `OpenVINOModelHandler` | `olive/model/handler/openvino.py:16` | OpenVINO IR models |
| `QNNModel` | `QNNModelHandler` | `olive/model/handler/qnn.py:22` | Qualcomm QNN models |
| `TensorFlowModel` | `TensorFlowModelHandler` | `olive/model/handler/tensorflow.py:15` | TensorFlow models |
| `CompositeModel` | `CompositeModelHandler` | `olive/model/handler/composite.py:21` | Multi-component models |
| `DistributedHfModel` | `DistributedHfModelHandler` | `olive/model/handler/hf.py:142` | Distributed HuggingFace models |
| `DistributedOnnxModel` | `DistributedOnnxModelHandler` | `olive/model/handler/onnx.py:204` | Distributed ONNX models |

### 13.2 `olive optimize` Command - Supported Input Types

The `olive optimize` command supports **3 input model types**:

| Input Type | How to Use | Example |
|------------|------------|---------|
| **HuggingFace** | Model ID, URL, or local checkpoint | `olive optimize -m Qwen/Qwen3-0.6B` |
| **ONNX** | `.onnx` file or folder with `.onnx` | `olive optimize -m ./model.onnx` |
| **PyTorch** | With `--model_script` argument | `olive optimize --model_script model.py` |

### 13.3 Model Type Auto-Detection Logic

```
Input Path (--model_name_or_path)
    |
    +-- No path + --model_script provided
    |       --> PyTorchModel (custom loader)
    |
    +-- Directory with model_config.json
    |       --> (Previous Olive output, type from config)
    |
    +-- --model_script provided
    |       --> PyTorchModel
    |
    +-- azureml://registries/<reg>/models/<name>/versions/<ver>
    |       --> HfModel (AzureML Registry)
    |
    +-- https://huggingface.co/<org>/<model>
    |       --> HfModel (HuggingFace URL)
    |
    +-- String matching HuggingFace repo (validated via API)
    |       --> HfModel (HuggingFace ID)
    |
    +-- Local .onnx file
    |       --> OnnxModel
    |
    +-- Local directory containing .onnx file(s)
    |       --> OnnxModel
    |
    +-- Local directory with config.json (HF checkpoint)
            --> HfModel
```

**Code Location**: `olive/cli/base.py:203` - `get_input_model_config()`

### 13.4 Supported Model Categories (by Domain/Modality)

#### Text/NLP Models

| Category | Tasks | Example Models |
|----------|-------|----------------|
| **LLM (Decoder-only)** | `text-generation`, `text-generation-with-past` | Llama, Llama2, Llama3, Phi, Phi-3, Qwen, Qwen2, Mistral, Mixtral, Falcon, GPT-2, GPT-Neo, GPT-J, OPT, Bloom, StableLM, Gemma, DeepSeek |
| **Text Encoder** | `feature-extraction`, `fill-mask` | BERT, RoBERTa, DistilBERT, ALBERT, XLM-RoBERTa, DeBERTa, Electra, CamemBERT |
| **Text Classification** | `text-classification` | BERT, RoBERTa for sentiment analysis |
| **Token Classification** | `token-classification` | BERT, DistilBERT for NER, POS tagging |
| **Question Answering** | `question-answering` | BERT, RoBERTa, DistilBERT |
| **Seq2Seq (Encoder-Decoder)** | `text2text-generation`, `text2text-generation-with-past` | T5, BART, mBART, Pegasus, MarianMT, M2M100, NLLB |
| **Multiple Choice** | `multiple-choice` | BERT, RoBERTa |

#### Vision Models

| Category | Tasks | Example Models |
|----------|-------|----------------|
| **Image Classification** | `image-classification` | ViT, Swin, DeiT, BEiT, ConvNeXt, ResNet, EfficientNet |
| **Object Detection** | `object-detection` | DETR, YOLOS, Conditional DETR |
| **Semantic Segmentation** | `semantic-segmentation` | SegFormer, Mask2Former |
| **Zero-shot Classification** | `zero-shot-image-classification` | CLIP, SigLIP |

#### Audio Models

| Category | Tasks | Example Models |
|----------|-------|----------------|
| **Audio Classification** | `audio-classification` | Wav2Vec2, HuBERT, SEW |
| **Speech-to-Text (ASR)** | `automatic-speech-recognition` | Whisper, Wav2Vec2, HuBERT |

#### Diffusion Models (Image Generation)

| Pipeline | Components | Example Models |
|----------|------------|----------------|
| **Stable Diffusion 1.x/2.x** | text_encoder, unet, vae | stable-diffusion-v1-5, stable-diffusion-2-1 |
| **SDXL** | text_encoder, text_encoder_2, unet, vae | stable-diffusion-xl-base-1.0 |
| **SD3** | text_encoder, text_encoder_2, text_encoder_3 (T5), transformer, vae | stable-diffusion-3-medium |
| **Flux** | text_encoder, text_encoder_2 (T5), transformer, vae | FLUX.1-dev, FLUX.1-schnell |
| **Sana** | gemma2_text_encoder, sana_transformer, dcae | Sana-1600M |

#### Multimodal Models

| Category | Tasks | Example Models |
|----------|-------|----------------|
| **Vision-Language** | `zero-shot-image-classification`, `visual-question-answering` | CLIP, BLIP, BLIP-2, LLaVA (limited) |
| **Document Understanding** | `document-question-answering` | LayoutLM, Donut |

### 13.5 Support Level by Modality

| Modality | `olive optimize` | `olive finetune` | Notes |
|----------|------------------|------------------|-------|
| **LLM/Text** | Full | Full | Default modality, best supported |
| **Text Encoder** | Full | Partial | BERT-style models |
| **Seq2Seq** | Full | Partial | T5, BART |
| **Image Classification** | Partial | No | Via HF models, no dedicated CLI |
| **Object Detection** | Partial | No | Via HF models |
| **Audio/ASR** | Partial | No | Whisper supported via HF |
| **Diffusion** | No | Yes (LoRA) | Use `olive finetune` for SD LoRA |
| **Video** | Not supported | Not supported | No video-specific tasks |
| **3D/Point Cloud** | Not supported | Not supported | Not implemented |

### 13.6 NOT Supported

The following model types are **not supported** by Olive:

- Video models (no video tasks defined)
- 3D models / Point cloud models
- Reinforcement learning models
- Graph neural networks (GNN)
- Time series models (limited, only `time-series-forecasting` task exists)
- Recommender system models

### 13.7 Task/Category Auto-Detection

**Olive does NOT fully auto-detect the task.** It uses a **default assumption** + **manual override** approach.

#### Detection Behavior

| Property | Auto-Detected | Source |
|----------|---------------|--------|
| `model_type` | Yes | HF `config.json` → `"model_type": "qwen2"` |
| `architectures` | Yes | HF `config.json` → `"architectures": ["Qwen2ForCausalLM"]` |
| `vocab_size` | Yes | HF `config.json` |
| `hidden_size` | Yes | HF `config.json` |
| **task** | **NO** | Must specify `--task` or defaults to `text-generation-with-past` |
| **category** | **NO** | Inferred from task |

#### Default Task

**Code Location**: `olive/common/constants.py:25`
```python
DEFAULT_HF_TASK = "text-generation-with-past"
```

All HuggingFace models default to `text-generation-with-past` (LLM task) unless `--task` is specified.

#### Model Type Detection

**Code Location**: `olive/model/handler/mixin/hf.py:128`
```python
def get_hf_model_type(self) -> str:
    """Get model type for the model."""
    return self.get_hf_model_config().model_type  # e.g., "llama", "qwen2", "bert"
```

Olive reads `model_type` from the HuggingFace config but does NOT automatically map it to a task.

#### Auto-Detection (OpenVINO Only)

Only OpenVINO passes have true auto-detection via Optimum's `TasksManager`:

**Code Location**: `olive/passes/openvino/optimum_intel.py:65`
```python
def infer_task(task, model_name_or_path, ...):
    from optimum.exporters import TasksManager

    task = TasksManager.map_from_synonym(task)
    if task == "auto":
        task = TasksManager._infer_task_from_model_name_or_path(
            model_name_or_path=model_name_or_path,
            ...
        )
    return task
```

#### Usage Examples

```bash
# LLM model - works with default task
olive optimize -m Qwen/Qwen3-0.6B --precision int4
# task defaults to "text-generation-with-past" - CORRECT

# BERT model - MUST specify task
olive optimize -m bert-base-uncased --precision int8 --task fill-mask
# Without --task, would incorrectly use "text-generation-with-past" - WRONG

# Whisper model - MUST specify task
olive optimize -m openai/whisper-base --precision int8 --task automatic-speech-recognition

# Image model - MUST specify task
olive optimize -m google/vit-base-patch16-224 --precision int8 --task image-classification

# OpenVINO with auto-detection
olive optimize -m bert-base-uncased --provider OpenVINOExecutionProvider --precision int8
# OpenVINO pass will auto-detect task via TasksManager
```

#### HuggingFace Config Examples

```json
// Qwen/Qwen3-0.6B config.json - LLM
{
  "architectures": ["Qwen2ForCausalLM"],
  "model_type": "qwen2",
  "vocab_size": 151936,
  "hidden_size": 896
}

// bert-base-uncased config.json - Encoder
{
  "architectures": ["BertForMaskedLM"],
  "model_type": "bert",
  "vocab_size": 30522,
  "hidden_size": 768
}

// openai/whisper-base config.json - ASR
{
  "architectures": ["WhisperForConditionalGeneration"],
  "model_type": "whisper",
  "vocab_size": 51865
}
```

#### Architecture to Task Mapping (NOT automatic in Olive)

For reference, here's the typical mapping (users must specify manually):

| Architecture Pattern | Typical Task |
|---------------------|--------------|
| `*ForCausalLM` | `text-generation` |
| `*ForMaskedLM` | `fill-mask` |
| `*ForSequenceClassification` | `text-classification` |
| `*ForTokenClassification` | `token-classification` |
| `*ForQuestionAnswering` | `question-answering` |
| `*ForConditionalGeneration` | `text2text-generation` or `automatic-speech-recognition` |
| `*ForImageClassification` | `image-classification` |
| `*ForObjectDetection` | `object-detection` |

### 13.8 Task Definitions

Tasks are defined in `olive/assets/io_configs/tasks.yaml`. Each task specifies:
- Input tensor names and shapes
- Output tensor names
- Dynamic axes for ONNX export
- Data types

**Full Task List:**
```
Text Tasks:
  - text-generation
  - text-generation-with-past
  - text-classification
  - token-classification
  - question-answering
  - fill-mask
  - feature-extraction
  - feature-extraction-with-past
  - text2text-generation
  - text2text-generation-with-past
  - multiple-choice
  - document-question-answering

Vision Tasks:
  - image-classification
  - object-detection
  - semantic-segmentation
  - zero-shot-image-classification
  - image-to-image
  - image-to-text
  - depth-estimation
  - mask-generation
  - keypoint-detection

Audio Tasks:
  - audio-classification
  - audio-frame-classification
  - audio-xvector
  - automatic-speech-recognition

Multimodal Tasks:
  - visual-question-answering
  - zero-shot-object-detection
```

### 13.2 HuggingFace Model Support

HuggingFace models are supported through the **Optimum library** for ONNX export. The supported architectures are listed at:
https://huggingface.co/docs/optimum/exporters/onnx/overview

**Common Supported Architectures**:

| Category | Model Families |
|----------|----------------|
| **Decoder-only LLMs** | Llama, Llama2, Llama3, Phi, Phi-3, Qwen, Qwen2, Mistral, Mixtral, Falcon, GPT-2, GPT-Neo, GPT-J, OPT, Bloom, StableLM, Gemma, DeepSeek |
| **Encoder-only** | BERT, RoBERTa, DistilBERT, ALBERT, XLM-RoBERTa, DeBERTa, Electra, CamemBERT |
| **Encoder-Decoder** | T5, BART, mBART, Pegasus, MarianMT, M2M100, NLLB |
| **Vision** | ViT, Swin, DeiT, BEiT, ConvNeXt, ResNet, EfficientNet |
| **Multimodal** | CLIP, LLaVA, Pix2Struct, BLIP, BLIP-2 |
| **Audio** | Whisper, Wav2Vec2, HuBERT, SEW |

### 13.3 Supported Tasks

Tasks are defined in `olive/assets/io_configs/tasks.yaml`:

| Task | Description | Example Models |
|------|-------------|----------------|
| `text-generation` | Autoregressive text generation | Llama, GPT, Phi, Qwen |
| `text-generation-with-past` | Text generation with KV cache | Same as above |
| `text-classification` | Sequence classification | BERT, RoBERTa |
| `token-classification` | Named entity recognition | BERT, DistilBERT |
| `question-answering` | Extractive QA | BERT, RoBERTa |
| `fill-mask` | Masked language modeling | BERT, RoBERTa |
| `feature-extraction` | Embeddings extraction | BERT, Sentence-BERT |
| `text2text-generation` | Seq2seq generation | T5, BART |
| `multiple-choice` | Multiple choice QA | BERT, RoBERTa |
| `image-classification` | Image classification | ViT, ResNet |
| `object-detection` | Object detection | DETR, YOLOS |
| `semantic-segmentation` | Semantic segmentation | SegFormer |
| `audio-classification` | Audio classification | Wav2Vec2 |
| `automatic-speech-recognition` | Speech-to-text | Whisper |
| `zero-shot-image-classification` | Zero-shot image classification | CLIP |

**Task Synonyms** (mapped automatically):
```
"default"          -> "feature-extraction"
"masked-lm"        -> "fill-mask"
"causal-lm"        -> "text-generation"
"causal-lm-with-past" -> "text-generation-with-past"
"seq2seq-lm"       -> "text2text-generation"
"sequence-classification" -> "text-classification"
```

### 13.4 Diffusers Pipeline Support

Supported in `olive/assets/io_configs/diffusers.yaml`:

| Pipeline | Components |
|----------|------------|
| **SD** (Stable Diffusion) | text_encoder, unet, vae_encoder, vae_decoder |
| **SDXL** | text_encoder, text_encoder_2, unet, vae_encoder, vae_decoder |
| **SD3** | text_encoder, text_encoder_2, text_encoder_3 (T5), transformer, vae |
| **Flux** | text_encoder, text_encoder_2 (T5), transformer, vae |
| **Sana** | gemma2_text_encoder, sana_transformer, dcae_encoder, dcae_decoder |

### 13.5 How to Check if a Model is Supported

#### Method 1: Check HuggingFace Optimum Support

```bash
# If the model exports with optimum, it's supported
optimum-cli export onnx --model <model_name> --task <task> output_dir/
```

#### Method 2: Try Olive Directly

```bash
# Olive will report if the model type is unsupported
olive optimize -m <model_name> --precision fp32 --dry_run
```

#### Method 3: Check Programmatically

```python
from olive.common.hf.io_config import is_task_supported
from olive.common.hf.utils import get_model_config

# Check if task is supported
print(is_task_supported("text-generation"))  # True

# Get model architecture type
config = get_model_config("microsoft/phi-2")
print(config.model_type)  # "phi"
```

#### Method 4: Check Model Config

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
print(f"Model type: {config.model_type}")       # "qwen2"
print(f"Architectures: {config.architectures}") # ["Qwen2ForCausalLM"]
```

### 13.6 Model Detection Flow

```
Input: model_name_or_path
        |
        v
+------------------+
| get_input_model_ |
|     config()     |  (olive/cli/base.py:203)
+--------+---------+
         |
    +----+----+----+----+
    |    |    |    |    |
    v    v    v    v    v
  AzureML  HF URL  HF ID  Local  Local
  Registry         string  ONNX   HF
    |         |      |      |      |
    +---------+------+      |      |
              |             |      |
              v             v      v
        HfModelHandler  ONNXModel  HfModel
```

### 13.7 Unsupported Model Handling

When a model architecture is not supported:

1. **IO Config Generation Fails**:
   ```
   ValueError: Unable to get dummy inputs for the model 'xxx'.
   The model type 'xxx' may not be supported.
   Please provide io_config manually.
   ```

2. **Workaround - Provide Manual io_config**:
   ```json
   {
     "input_model": {
       "type": "HfModel",
       "model_path": "your-model",
       "io_config": {
         "input_names": ["input_ids", "attention_mask"],
         "output_names": ["logits"],
         "dynamic_axes": {
           "input_ids": {"0": "batch", "1": "seq"},
           "attention_mask": {"0": "batch", "1": "seq"},
           "logits": {"0": "batch", "1": "seq"}
         }
       }
     }
   }
   ```

3. **Use PyTorchModelHandler with Custom Script**:
   ```bash
   olive optimize \
       --model_name_or_path /path/to/model \
       --model_script model_definition.py
   ```

   Where `model_definition.py` contains:
   ```python
   def _model_loader():
       # Load and return your model
       return model

   def _io_config(model):
       return {
           "input_names": [...],
           "output_names": [...],
           "dynamic_axes": {...}
       }

   def _dummy_inputs(model):
       return {"input_ids": torch.ones(1, 128, dtype=torch.long)}
   ```

### 13.8 Model Type Registration

New model handlers are registered using the decorator pattern:

```python
# olive/model/config/registry.py
REGISTRY = {}

def model_handler_registry(model_type):
    """Decorator to register model handlers."""
    model_type = model_type.lower()

    def decorator_model_class(cls):
        REGISTRY[model_type] = cls
        cls.model_type = model_type
        return cls

    return decorator_model_class

# Usage in handler files:
@model_handler_registry("HFModel")
class HfModelHandler(PyTorchModelHandlerBase, ...):
    ...
```

### 13.9 Quick Reference: Model Type by Source

| Source | Detected Type | Handler |
|--------|---------------|---------|
| `Qwen/Qwen2.5-0.5B` (HF ID) | HfModel | HfModelHandler |
| `https://huggingface.co/meta-llama/Llama-2-7b` | HfModel | HfModelHandler |
| `azureml://registries/.../models/.../versions/...` | HfModel | HfModelHandler |
| `/path/to/model.onnx` | OnnxModel | ONNXModelHandler |
| `/path/to/model_dir/*.onnx` | OnnxModel | ONNXModelHandler |
| `/path/to/hf_checkpoint/` (with config.json) | HfModel | HfModelHandler |
| `--model_script custom.py` | PyTorchModel | PyTorchModelHandler |
| Previous Olive output (with model_config.json) | (from config) | (from config) |

---

## 14. `olive optimize` - Complete Configuration Reference

### 14.1 Input Model Types Supported

| Input Type | Detection Method | Handler |
|------------|------------------|---------|
| **HuggingFace Model** | HF repo ID, HF URL, local HF checkpoint | `HfModel` |
| **ONNX Model** | `.onnx` file or directory with `.onnx` | `OnnxModel` |
| **PyTorch Model** | `--model_script` argument | `PyTorchModel` |
| **Previous Olive Output** | Directory with `model_config.json` | (from config) |

### 14.2 Configuration Matrix by Provider + Precision

#### 14.2.1 CPUExecutionProvider (Default)

**HuggingFace Model + CPU:**

| Precision | Passes Applied | Config |
|-----------|----------------|--------|
| **fp32** | `ModelBuilder` | `{"type": "ModelBuilder", "precision": "fp32"}` |
| **fp16** | `ModelBuilder` -> `OnnxFloatToFloat16` | FP16 conversion after ONNX export |
| **int4** | `Gptq` -> `ModelBuilder` | `{"type": "Gptq", "bits": 4, "sym": true}` + `{"type": "ModelBuilder", "precision": "int4", "int4_block_size": 32}` |
| **int8** | `ModelBuilder` -> `OnnxStaticQuantization` | Static quantization with wikitext2 calibration |

```bash
# Example: HF + CPU + INT4
olive optimize -m Qwen/Qwen3-0.6B --provider CPUExecutionProvider --precision int4
```

**ONNX Model + CPU:**

| Precision | Passes Applied | Config |
|-----------|----------------|--------|
| **fp32** | (none) | Pass-through |
| **fp16** | `OnnxFloatToFloat16` | `{"type": "OnnxFloatToFloat16"}` |
| **int4** | `OnnxBlockWiseRtnQuantization` | `{"type": "OnnxBlockWiseRtnQuantization"}` |
| **int8** | `OnnxStaticQuantization` | `{"type": "OnnxStaticQuantization", "precision": "int8"}` |

```bash
# Example: ONNX + CPU + INT4
olive optimize -m ./model.onnx --provider CPUExecutionProvider --precision int4
```

#### 14.2.2 CUDAExecutionProvider

**HuggingFace Model + CUDA:**

| Precision | Passes Applied | Config |
|-----------|----------------|--------|
| **fp32** | `ModelBuilder` | `{"type": "ModelBuilder", "precision": "fp32"}` |
| **fp16** | `ModelBuilder` -> `OnnxFloatToFloat16` | FP16 conversion |
| **int4** | `Gptq` -> `ModelBuilder` | GPTQ quantization + ModelBuilder |
| **int8** | `ModelBuilder` -> `OnnxStaticQuantization` | Static quantization |

```bash
# Example: HF + CUDA + INT4
olive optimize -m Qwen/Qwen3-0.6B --provider CUDAExecutionProvider --precision int4
```

#### 14.2.3 QNNExecutionProvider (Qualcomm NPU)

**HuggingFace Model + QNN:**

| Precision | Passes Applied | Config |
|-----------|----------------|--------|
| **fp32** | `ModelBuilder` -> `StaticLLM` | Static shapes for NPU |
| **fp16** | `ModelBuilder` -> `OnnxFloatToFloat16` -> `StaticLLM` | FP16 + static shapes |
| **int4** | `QuaRot` -> `Gptq` -> `ModelBuilder` -> `StaticLLM` | Full quantization pipeline |
| **int4 + int8 act** | `QuaRot` -> `Gptq` -> `ModelBuilder` -> `OnnxStaticQuantization` -> `StaticLLM` | Weight + activation quantization |

**Pass Configurations for QNN INT4:**
```json
{
  "quarot": {"type": "QuaRot"},
  "gptq": {"type": "Gptq", "bits": 4, "sym": true},
  "model_builder": {
    "type": "ModelBuilder",
    "precision": "int4",
    "int4_block_size": 32,
    "int4_accuracy_level": 4,
    "int4_op_types_to_quantize": ["MatMul", "Gather"]
  },
  "static_llm": {"type": "StaticLLM"}
}
```

**With AOT Compilation:**
```json
{
  "ep_context_binary_generator": {
    "type": "EPContextBinaryGenerator",
    "session_options": {"intra_op_num_threads": 2, "inter_op_num_threads": 1},
    "weight_sharing": true,
    "provider_options": {
      "htp_performance_mode": "burst",
      "htp_graph_finalization_optimization_mode": "3",
      "soc_model": "60"
    }
  }
}
```

```bash
# Example: HF + QNN + INT4
olive optimize -m Qwen/Qwen3-0.6B --provider QNNExecutionProvider --precision int4

# With activation quantization
olive optimize -m Qwen/Qwen3-0.6B --provider QNNExecutionProvider --precision int4 --act_precision int8

# With AOT
olive optimize -m Qwen/Qwen3-0.6B --provider QNNExecutionProvider --precision int4 --enable_aot --qnn_env_path /path/to/qnn
```

#### 14.2.4 OpenVINOExecutionProvider

**HuggingFace Model + OpenVINO:**

| Precision | Passes Applied | Config |
|-----------|----------------|--------|
| **fp32** | `OpenVINOOptimumConversion` -> `OpenVINOIoUpdate` -> `OpenVINOEncapsulation` | OpenVINO IR format |
| **fp16** | Same pipeline | `weight_format: "fp16"` |
| **int4** | Same pipeline | `weight_format: "int4"` |
| **int8** | Same pipeline | `weight_format: "int8"` |

**Pass Configurations:**
```json
{
  "optimum_openvino_conversion": {
    "type": "OpenVINOOptimumConversion",
    "extra_args": {"device": "cpu"},
    "ov_quant_config": {
      "task": "text-generation-with-past",
      "weight_format": "int4",
      "group_size": 128,
      "ratio": 1
    }
  },
  "openvino_io_update": {
    "type": "OpenVINOIoUpdate",
    "static": false,
    "reuse_cache": true
  },
  "openvino_encapsulation": {
    "type": "OpenVINOEncapsulation",
    "target_device": "cpu",
    "keep_ov_dynamic_shapes": true,
    "op_version": "2025.1",
    "reuse_cache": true
  }
}
```

```bash
# Example: HF + OpenVINO + INT4
olive optimize -m Qwen/Qwen3-0.6B --provider OpenVINOExecutionProvider --precision int4 --device cpu
```

#### 14.2.5 VitisAIExecutionProvider (AMD/Xilinx)

**HuggingFace Model + VitisAI:**

| Precision | Passes Applied | Config |
|-----------|----------------|--------|
| **int4** | `QuaRot` -> `Gptq` -> `ModelBuilder` -> `StaticLLM` -> `VitisAIAddMetaData` | Full pipeline with metadata |
| **int4 + int8 act** | Above + `OnnxStaticQuantization` | Weight + activation quantization |

**Pass Configurations:**
```json
{
  "static_llm": {
    "type": "StaticLLM",
    "batch_size": 1,
    "context_length": 64,
    "group_session_options": {
      "log_id": "onnxruntime-genai",
      "provider_options": [{"VitisAI": {}}],
      "graph_optimization_level": "ORT_ENABLE_ALL"
    }
  },
  "vitis_ai_add_metadata": {
    "type": "VitisAIAddMetaData",
    "config_meta_data_keys": ["architectures", "model_type"],
    "weight_type": "int4",
    "quant_type": "gptq"
  }
}
```

```bash
# Example: HF + VitisAI + INT4
olive optimize -m Qwen/Qwen3-0.6B --provider VitisAIExecutionProvider --precision int4
```

#### 14.2.6 WebGpuExecutionProvider

**HuggingFace Model + WebGPU:**

| Precision | Passes Applied | Config |
|-----------|----------------|--------|
| **fp32** | `ModelBuilder` -> `OnnxIODataTypeConverter` | Convert logits FP16->FP32 |
| **int4** | `Gptq` -> `ModelBuilder` -> `OnnxIODataTypeConverter` | Quantization + dtype fix |

**Pass Configurations:**
```json
{
  "onnx_io_datatype_converter": {
    "type": "OnnxIODataTypeConverter",
    "name_pattern": "logits",
    "source_dtype": 10,
    "target_dtype": 1
  }
}
```

```bash
# Example: HF + WebGPU + INT4
olive optimize -m Qwen/Qwen3-0.6B --provider WebGpuExecutionProvider --precision int4
```

#### 14.2.7 NvTensorRTRTXExecutionProvider

**HuggingFace Model + TensorRT:**

| Precision | Passes Applied | Config |
|-----------|----------------|--------|
| **fp32** | `ModelBuilder` | No OrtTransformersOptimization |
| **int4** | `Gptq` -> `ModelBuilder` | GPTQ + ModelBuilder only |

```bash
# Example: HF + TensorRT + INT4
olive optimize -m Qwen/Qwen3-0.6B --provider NvTensorRTRTXExecutionProvider --precision int4
```

### 14.3 Optional Features (Additional Passes)

#### Model Splitting (`--num_split` or `--memory`)

```bash
olive optimize -m Qwen/Qwen3-0.6B --precision int4 --num_split 2
```

**Additional Passes:**
```json
{
  "capture_split_info": {
    "type": "CaptureSplitInfo",
    "unique_embeds_lm_head_splits": true,
    "num_splits": 2
  },
  "split_model": {"type": "SplitModel"}
}
```

#### QDQ Format (`--use_qdq_format`)

```bash
olive optimize -m Qwen/Qwen3-0.6B --precision int4 --use_qdq_format
```

**Additional Pass:**
```json
{
  "matmul_nbits_to_qdq": {
    "type": "MatMulNBitsToQDQ",
    "add_zero_point": "true",
    "save_as_external_data": "true",
    "nodes_to_exclude": ["/lm_head/MatMul_Q4"],
    "use_int4": "true"
  }
}
```

#### Dynamic to Fixed Shape (`--dim_param` + `--dim_value`)

```bash
olive optimize -m Qwen/Qwen3-0.6B --precision int4 --provider QNNExecutionProvider \
    --dim_param "batch,seq_len" --dim_value "1,128"
```

**Additional Pass:**
```json
{
  "dynamic_to_fixed_shape": {
    "type": "DynamicToFixedShape",
    "dim_param": ["batch", "seq_len"],
    "dim_value": [1, 128]
  }
}
```

#### Graph Surgeries (`--surgeries`)

```bash
olive optimize -m Qwen/Qwen3-0.6B --precision int4 --surgeries "RemoveShapeOps,FuseReshapes"
```

**Additional Pass:**
```json
{
  "graph_surgeries": {
    "type": "GraphSurgeries",
    "surgeries": [{"surgeon": "RemoveShapeOps"}, {"surgeon": "FuseReshapes"}],
    "save_as_external_data": "true"
  }
}
```

### 14.4 Summary Table: Model Type x Provider x Precision

| Model | Provider | Precision | Key Passes |
|-------|----------|-----------|------------|
| HF | CPU | fp32 | ModelBuilder |
| HF | CPU | int4 | Gptq + ModelBuilder |
| HF | CUDA | int4 | Gptq + ModelBuilder |
| HF | QNN | int4 | QuaRot + Gptq + ModelBuilder + StaticLLM |
| HF | QNN+AOT | int4 | Above + EPContextBinaryGenerator |
| HF | OpenVINO | int4 | OpenVINOOptimumConversion + IoUpdate + Encapsulation |
| HF | VitisAI | int4 | QuaRot + Gptq + ModelBuilder + StaticLLM + VitisAIAddMetaData |
| HF | WebGPU | int4 | Gptq + ModelBuilder + OnnxIODataTypeConverter |
| HF | TensorRT | int4 | Gptq + ModelBuilder |
| ONNX | CPU | int4 | OnnxBlockWiseRtnQuantization |
| ONNX | CPU | int8 | OnnxStaticQuantization |
| ONNX | Any | fp16 | OnnxFloatToFloat16 |

### 14.5 Exporter Options

The `--exporter` argument controls how HuggingFace models are converted to ONNX:

| Exporter | Passes | Use Case |
|----------|--------|----------|
| `model_builder` (default) | `ModelBuilder` | Best for LLMs, uses onnxruntime-genai |
| `dynamo_exporter` | `OnnxConversion` + `OnnxPeepholeOptimizer` + `OrtTransformersOptimization` | PyTorch 2.0 dynamo export |
| `torchscript_exporter` | `OnnxConversion` + `OnnxPeepholeOptimizer` + `OrtTransformersOptimization` | Legacy TorchScript export |
| `optimum_exporter` | (via Optimum library) | HuggingFace Optimum integration |

```bash
# Using dynamo exporter
olive optimize -m Qwen/Qwen3-0.6B --precision fp32 --exporter dynamo_exporter

# Using torchscript exporter
olive optimize -m Qwen/Qwen3-0.6B --precision fp32 --exporter torchscript_exporter
```

### 14.6 Pass Execution Order

Passes are executed in the order they are added to the `OrderedDict` in `_build_passes_config()`:

1. `QuaRot` (rotation for better quantization)
2. `Gptq` (weight quantization)
3. `CaptureSplitInfo` (for model splitting)
4. `ModelBuilder` / `OnnxConversion` (ONNX export)
5. `OpenVINOOptimumConversion` (OpenVINO only)
6. `DynamicToFixedShape` (NPU targets)
7. `OnnxIODataTypeConverter` (WebGPU)
8. `OpenVINOIoUpdate` (OpenVINO)
9. `OnnxPeepholeOptimizer` (graph optimization)
10. `OrtTransformersOptimization` (transformer-specific optimizations)
11. `MatMulNBitsToQDQ` (QDQ format conversion)
12. `GraphSurgeries` (custom graph modifications)
13. `OnnxBlockWiseRtnQuantization` (ONNX model quantization)
14. `OnnxFloatToFloat16` (FP16 conversion)
15. `OnnxStaticQuantization` (activation quantization)
16. `VitisAIAddMetaData` (VitisAI metadata)
17. `SplitModel` (model splitting)
18. `StaticLLM` (static shapes for NPU)
19. `EPContextBinaryGenerator` (QNN AOT compilation)
20. `ComposeOnnxModels` (merge split models)
21. `OpenVINOEncapsulation` (OpenVINO packaging)
