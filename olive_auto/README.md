# olive-auto

Batch optimization pipeline generator for [Olive](https://github.com/microsoft/Olive).

Generate complete optimization pipelines for HuggingFace models across all supported targets and precisions with a single command.

## Installation

### Prerequisites

- Python 3.9 or later
- pip (Python package installer)
- Git (to clone the repository)

### Setup for New Users

1. **Clone the repository:**

   ```bash
   git clone https://github.com/microsoft/Olive.git
   cd Olive
   ```

2. **Install olive-auto:**

   ```bash
   pip install -e ./olive_auto
   ```

3. **Verify installation:**

   ```bash
   olive-auto --version
   ```

### Optional Dependencies

Install additional dependencies based on your target hardware:

```bash
# Full dependencies (includes Olive, Optimum, Transformers)
pip install -e "./olive_auto[full]"

# CPU inference only
pip install -e "./olive_auto[cpu]"

# NVIDIA GPU (CUDA)
pip install -e "./olive_auto[cuda]"

# Intel OpenVINO
pip install -e "./olive_auto[openvino]"

# Windows DirectML
pip install -e "./olive_auto[directml]"

# Development dependencies (pytest, ruff, mypy)
pip install -e "./olive_auto[dev]"
```

### Virtual Environment (Recommended)

It's recommended to use a virtual environment:

```bash
# Create and activate virtual environment
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat

# Then install olive-auto
pip install -e ./olive_auto
```

## Quick Start

```bash
# Generate pipeline for all compatible targets
olive-auto -m microsoft/phi-2 -o ./phi2_pipeline

# Generate for specific targets only
olive-auto -m openai/whisper-tiny --targets cpu cuda -o ./whisper_pipeline

# Generate and immediately execute
olive-auto -m bert-base-uncased --targets cpu --run -o ./bert_pipeline

# Preview without creating files
olive-auto -m microsoft/phi-2 --dry-run
```

## Generated Output

```
optimization_pipeline/
├── setup_environments.sh        # Setup all required environments
├── run_all_optimizations.sh     # Master orchestration script
├── cleanup.sh                   # Cleanup environments and temp files
├── manifest.json                # Pipeline metadata
├── environments/
│   ├── setup_cpu.sh
│   ├── setup_cuda.sh
│   └── ...
├── commands/
│   ├── cpu_fp32.sh
│   ├── cpu_int8.sh
│   ├── cuda_fp16.sh
│   └── ...
├── outputs/                     # Optimized models (after execution)
└── logs/                        # Execution logs
```

## Features

### Auto Task Detection
Automatically detects the model task from:
1. Optimum's TasksManager
2. Model architecture name patterns
3. HuggingFace model card metadata

### Supported Targets

| Target | Description | Platform |
|--------|-------------|----------|
| `cpu` | CPU inference via ONNX Runtime | All |
| `cuda` | NVIDIA GPU via CUDA | Linux/Windows |
| `qnn` | Qualcomm NPU | Windows/Android |
| `openvino` | Intel CPU/GPU/VPU | Linux/Windows |
| `tensorrt` | NVIDIA TensorRT | Linux/Windows |
| `directml` | Windows DirectML | Windows |
| `webgpu` | Browser WebGPU | Web |
| `vitisai` | AMD/Xilinx FPGA/NPU | Linux |
| `rocm` | AMD GPU via ROCm | Linux |

### Supported Precisions

| Precision | Description |
|-----------|-------------|
| `fp32` | 32-bit floating point |
| `fp16` | 16-bit floating point |
| `bf16` | Brain floating point 16 |
| `int8` | 8-bit integer quantization |
| `int4` | 4-bit integer quantization |
| `bnb4` | bitsandbytes 4-bit |

## CLI Reference

```
olive-auto [OPTIONS]

Options:
  -m, --model TEXT              HuggingFace model ID or local path [required]
  --task TEXT                   Model task (auto-detected if not specified)
  --trust-remote-code           Trust remote code from HuggingFace

  -t, --targets TEXT            Specific targets (can specify multiple)
  -p, --precisions TEXT         Specific precisions (can specify multiple)
  --exclude-targets TEXT        Targets to exclude

  -o, --output-dir PATH         Output directory [default: ./olive_auto_pipeline]
  --platform [linux|windows|auto]  Target platform [default: auto]

  --parallel INTEGER            Number of parallel optimizations [default: 1]
  --run                         Execute pipeline after generation
  --dry-run                     Show what would be generated

  -v, --verbose                 Enable verbose output
  --version                     Show version and exit
  --help                        Show this message and exit
```

## Usage Examples

### Basic Usage

```bash
# Generate pipeline with auto-detected settings
olive-auto -m microsoft/phi-2 -o ./pipeline
```

### Selective Targets

```bash
# Only CPU and CUDA
olive-auto -m microsoft/phi-2 -t cpu -t cuda -o ./pipeline

# Exclude specific targets
olive-auto -m microsoft/phi-2 --exclude-targets qnn vitisai -o ./pipeline
```

### Specific Precisions

```bash
# Only int8 quantization
olive-auto -m microsoft/phi-2 -p int8 -o ./pipeline

# Multiple precisions
olive-auto -m microsoft/phi-2 -p fp16 -p int8 -p int4 -o ./pipeline
```

### Platform-Specific

```bash
# Generate for Linux
olive-auto -m microsoft/phi-2 --platform linux -o ./pipeline

# Generate for Windows (PowerShell scripts)
olive-auto -m microsoft/phi-2 --platform windows -o ./pipeline
```

### Execute Pipeline

```bash
# Generate and run
olive-auto -m bert-base-uncased -t cpu --run -o ./pipeline

# Run with parallel execution
olive-auto -m bert-base-uncased -t cpu -t cuda --run --parallel 4 -o ./pipeline
```

## Running the Generated Pipeline

### Linux/macOS

```bash
cd ./pipeline

# First time: setup environments
./setup_environments.sh

# Run all optimizations
./run_all_optimizations.sh

# Or run individual commands
./commands/cpu_fp32.sh
./commands/cuda_fp16.sh

# Cleanup when done
./cleanup.sh
```

### Windows PowerShell

```powershell
cd .\pipeline

# First time: setup environments
.\setup_environments.ps1

# Run all optimizations
.\run_all_optimizations.ps1

# Cleanup when done
.\cleanup.ps1
```

## Development

### Running Tests

```bash
cd olive_auto
pip install -e ".[dev]"
pytest
```

### Code Style

```bash
ruff check .
ruff format .
```

## License

MIT License - see the [Olive repository](https://github.com/microsoft/Olive) for details.
