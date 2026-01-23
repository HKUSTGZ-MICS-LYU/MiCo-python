# AGENTS.md: Comprehensive Guide for LLM Agentic Development

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture and Core Concepts](#architecture-and-core-concepts)
3. [Repository Structure](#repository-structure)
4. [Setup and Prerequisites](#setup-and-prerequisites)
5. [Core Components](#core-components)
6. [Workflow Guide](#workflow-guide)
7. [Supported Models and Datasets](#supported-models-and-datasets)
8. [Code Examples and Use Cases](#code-examples-and-use-cases)
9. [Advanced Features](#advanced-features)
10. [Best Practices for LLM Agents](#best-practices-for-llm-agents)
11. [Troubleshooting Guide](#troubleshooting-guide)
12. [API Reference](#api-reference)

---

## Project Overview

**MiCo** (Mixed Precision Neural Network Co-Exploration Framework) is an end-to-end framework for training, exploring, and deploying mixed precision quantized models optimized for Edge AI applications.

### Key Capabilities
- **Mixed Precision Quantization (MPQ)**: Automatically search for optimal bitwidth configurations per layer
- **Hardware-Aware Search**: Proxy models for BitFusion and custom hardware latency prediction
- **C Code Generation**: Convert PyTorch models to optimized C code for deployment
- **Multiple Search Algorithms**: Support for Bayesian Optimization, HAQ, NLP, and custom MiCo searcher
- **End-to-End Pipeline**: From training to hardware deployment

### Publication
The framework is described in the paper: "MiCo: End-to-End Mixed Precision Neural Network Co-Exploration Framework for Edge AI" (ICCAD 2025)

---

## Architecture and Core Concepts

### Mixed Precision Quantization (MPQ)
- **Per-Layer Bitwidth Assignment**: Each layer can have different weight and activation bitwidths (e.g., 1-8 bits)
- **Search Space**: Explores combinations of bitwidths to find optimal accuracy-efficiency tradeoffs
- **Constraints**: Can optimize under BOPs (Bit Operations), MACs, or hardware latency constraints

### Three-Stage Pipeline
1. **Training Stage**: Train/load a full-precision or quantized model
2. **Search Stage**: Use MPQ search algorithms to find optimal bitwidth configurations
3. **Deployment Stage**: Generate C code and compile for target hardware

### Hardware-Aware Features
- **Proxy Models**: Predict hardware latency without running on actual hardware
- **Supported Targets**: CPUs (RISC-V VexiiRiscv, Rocket, Boom), Gemmini, BitFusion
- **Chipyard Integration**: Compatible with Chipyard ecosystem designs

---

## Repository Structure

```
MiCo-python/
├── Core Modules (Python files in root)
│   ├── MiCoModel.py          # Base model class with unified training/testing
│   ├── MiCoQLayers.py         # Quantized layer implementations (BitLinear, BitConv2d, etc.)
│   ├── MiCoUtils.py           # Utilities for layer replacement, model fusion, export
│   ├── MiCoEval.py            # Model evaluation (accuracy, BOPs, MACs, latency)
│   ├── MiCoSearch.py          # MPQ search coordination
│   ├── MiCoCodeGen.py         # C code generator using Torch FX
│   ├── MiCoGraphGen.py        # DNN Weaver graph generator
│   ├── MiCoLLaMaGen.py        # Specialized LLaMa C code generator
│   ├── MiCoProxy.py           # Hardware latency proxy models
│   ├── MiCoRegistry.py        # Registry pattern for custom operations
│   ├── MiCoAnalysis.py        # Model statistics and analysis
│   ├── SimUtils.py            # Hardware simulation utilities
│   ├── DimTransform.py        # Dimension transformation for search
│   └── datasets.py            # Dataset loaders
│
├── searchers/                 # MPQ search algorithms
│   ├── MiCoSearcher.py        # Main MiCo searcher (RF/XGB/Bayes)
│   ├── BayesSearcher.py       # Bayesian Optimization
│   ├── HAQSearcher.py         # HAQ (Hardware-Aware Quantization)
│   ├── NLPSearcher.py         # Natural Language Processing inspired
│   ├── RegressionSearcher.py  # Regression-based searcher
│   ├── QSearcher.py           # Base searcher class
│   └── SearchUtils.py         # Sampling utilities
│
├── models/                    # Model architectures
│   ├── MLP.py, LeNet.py, VGG.py, ResNet.py, MobileNetV2.py
│   ├── SqueezeNet.py, ShuffleNet.py, LLaMa.py, ViT.py
│   ├── DSCNN.py, M5.py, HARMLP.py
│   └── model_zoo.py           # Model registry
│
├── examples/                  # Example training and search scripts
│   ├── lenet_mnist.py         # Train LeNet on MNIST
│   ├── lenet_mnist_search.py  # MPQ search on LeNet
│   ├── mpq_search.py          # General MPQ search script
│   ├── mpq_train.py           # General MPQ training script
│   └── [other model examples]
│
├── deploy/                    # Hardware deployment examples
│   ├── lenet_on_mico.py       # Deploy LeNet on MiCo hardware
│   ├── vgg_on_bf.py           # Deploy VGG on BitFusion
│   └── [other deployment scripts]
│
├── doc/                       # Documentation
│   ├── REGISTRY_USAGE.md      # Guide for custom operations
│   ├── MEMORY_OPTIMIZATION.md # Memory pool optimization details
│   ├── CHIPYARD_INTEGRATION.md # Chipyard integration guide
│   └── icon_v1.jpg
│
├── tests/                     # Unit and integration tests
├── project/                   # C project templates (requires MiCo Library submodule)
├── hw/                        # Hardware-specific implementations
├── profiler/                  # Hardware profiling scripts
├── benchmark_results/         # Profiled hardware kernel datasets
├── output/                    # Generated outputs (ckpt, json, figs)
├── TinyStories/               # TinyStories dataset for LLaMa
├── requirements.txt           # Python dependencies
└── readme.md                  # Main README
```

---

## Setup and Prerequisites

### Environment Setup

```bash
# Create conda environment
conda create -n mico_env python=3.10
conda activate mico_env

# Install PyGMO (for evolutionary algorithms)
conda install pygmo

# Install Python packages
pip install -r requirements.txt
```

### Required Packages
- **Core**: torch, torchvision, numpy, matplotlib
- **Optimization**: botorch, gekko, pygmo
- **Audio** (optional): torchaudio, torchcodec
- **LLMs** (optional): requests, sentencepiece
- **Quantization**: torchao

### Path Setup
If you encounter `ModuleNotFoundError`:
```bash
export PYTHONPATH=$PYTHONPATH:.
```

### Submodules (for deployment)
```bash
git submodule update --init
```

---

## Core Components

### 1. MiCoModel (`MiCoModel.py`)
**Purpose**: Base class for MiCo models with unified training/testing interface.

**Key Features**:
- Unified `train_loop()` and `test()` methods
- Layer-wise bitwidth assignment via `set_qscheme()`
- Automatic conversion to quantized layers
- Support for both QAT (Quantization-Aware Training) and PTQ (Post-Training Quantization)

**Usage**:
```python
from models import LeNet
model = LeNet(in_channels=1)
model.set_qscheme([[8]*model.n_layers, [8]*model.n_layers])  # [weight_bits, act_bits]
```

### 2. MiCoQLayers (`MiCoQLayers.py`)
**Purpose**: Quantized layer implementations.

**Key Classes**:
- `BitLinear`: Quantized linear layer
- `BitConv2d`: Quantized 2D convolution
- `BitConv1d`: Quantized 1D convolution (for audio/time-series)
- `weight_quant()`: Weight quantization function

**Features**:
- Per-layer bitwidth control
- Scale factor calculation
- Support for symmetric/asymmetric quantization
- MAC (Multiply-Accumulate) and BOP (Bit Operation) counting

### 3. MiCoUtils (`MiCoUtils.py`)
**Purpose**: Utilities for model manipulation.

**Key Functions**:
- `list_quantize_layers(model)`: List quantizable layers
- `replace_quantize_layers(model, weight_q, act_q)`: Replace with quantized layers
- `fuse_model(model)`: Fuse Conv-BN layers
- `export_layer_weights(model)`: Export weights for C code generation
- `set_to_qforward(model)`: Switch to quantized forward pass

### 4. MiCoEval (`MiCoEval.py`)
**Purpose**: Comprehensive model evaluation.

**Evaluation Modes**:
- `ptq_acc`: Post-Training Quantization accuracy
- `qat_acc`: Quantization-Aware Training accuracy
- `bops`: Bit Operations count
- `size`: Model size (parameters)
- `latency_mico`: Hardware latency on MiCo
- `latency_bitfusion`: Hardware latency on BitFusion
- `latency_proxy`: Predicted latency using proxy models
- `latency_torchao`: TorchAO quantization latency

**Key Methods**:
```python
evaluator = MiCoEval(model, epochs, train_loader, test_loader, ckpt_path)
acc = evaluator.eval_f([8,8,8,8,8,8,8,8])  # Evaluate with bitwidth config
bops = evaluator.eval_bops([8,8,8,8,8,8,8,8])  # Calculate BOPs
latency = evaluator.eval_latency([8,8,8,8,8,8,8,8], target='mico')  # Hardware latency
```

### 5. MiCoCodeGen (`MiCoCodeGen.py`)
**Purpose**: Convert PyTorch models to C code.

**Features**:
- Uses PyTorch FX graph tracing
- Memory pool optimization (30-75% memory savings)
- Support for custom operations via registry pattern
- Generates `.h` model header and `.bin` weight binary

**Workflow**:
```python
from MiCoCodeGen import MiCoCodeGen
codegen = MiCoCodeGen(model, align_to=32)
codegen.forward(torch.randn(1, 1, 28, 28))  # Trace model
codegen.convert("output", "model_name")  # Generate C code
```

### 6. MiCo Searchers (`searchers/`)
**Purpose**: MPQ search algorithms.

**Available Searchers**:
- `MiCoSearcher`: Main searcher with RF/XGB/Bayes regressors
- `BayesSearcher`: Bayesian Optimization with Gaussian Process
- `HAQSearcher`: Hardware-Aware Quantization (CVPR'19)
- `NLPSearcher`: NLP-inspired search
- `RegressionSearcher`: Generic regression-based search

**Search Interface**:
```python
searcher = MiCoSearcher(evaluator, n_inits=10, qtypes=[4,5,6,7,8])
best_x, best_y = searcher.search(
    n_iterations=40, 
    objective='ptq_acc',        # Maximize accuracy
    constraint='bops',          # Constraint type (bops, size, latency_*)
    constraint_value=0.5        # 50% of INT8 BOPs
)
```

### 7. MiCoRegistry (`MiCoRegistry.py`)
**Purpose**: Registry pattern for extensible operation handlers.

**Key Decorators**:
- `@MiCoOpRegistry.register_function(torch.nn.functional.relu)`: Register function handler
- `@MiCoOpRegistry.register_module(torch.nn.Conv2d)`: Register module handler

**See**: [doc/REGISTRY_USAGE.md](doc/REGISTRY_USAGE.md) for detailed guide.

### 8. MiCoProxy (`MiCoProxy.py`)
**Purpose**: Hardware latency prediction without actual hardware.

**Proxy Models**:
- **MiCo Proxy**: CBOPs-based proxy for custom hardware
- **BitFusion Proxy**: Latency prediction for BitFusion accelerator

**Usage**:
```python
from MiCoProxy import get_mico_matmul_proxy, get_mico_conv2d_proxy
matmul_proxy = get_mico_matmul_proxy(mico_type="small")  # or "large"
conv2d_proxy = get_mico_conv2d_proxy(mico_type="small")
evaluator.set_proxy(matmul_proxy, conv2d_proxy)
```

---

## Workflow Guide

### Workflow 1: Train a Model

```bash
# Example: Train LeNet on MNIST
python examples/lenet_mnist.py
```

**Steps in code**:
1. Load model and dataset
2. Replace layers with quantized versions
3. Train with `model.train_loop()`
4. Save checkpoint
5. Export weights to JSON

**Output**:
- `output/ckpt/lenet_mnist.pth`: Model checkpoint
- `output/json/lenet_mnist.json`: Layer weights

### Workflow 2: Mixed Precision Search

```bash
# Example: MPQ search on trained LeNet
python examples/lenet_mnist_search.py
```

**Steps**:
1. Load trained model checkpoint
2. Create `MiCoEval` evaluator
3. Initialize searcher (Bayes, HAQ, MiCo, etc.)
4. Run search with constraints
5. Get optimal bitwidth configuration
6. Visualize search trajectory

**Output**:
- Best bitwidth configuration
- Accuracy vs. constraint tradeoff plot
- `output/json/lenet_mnist_search.json`: Search trace

### Workflow 3: General MPQ Search

```bash
# General search with model zoo
python examples/mpq_search.py lenet_mnist --init 16 -n 40 -c 0.5 -ctype bops -t 5 -m ptq_acc
```

**Arguments**:
- `model_name`: Model from model_zoo
- `--init`: Initial random samples
- `-n, --n-search`: Number of search iterations
- `-c, --constraint-factor`: Constraint ratio (e.g., 0.5 = 50% of INT8)
- `-ctype, --constraint`: Constraint type (bops, size, latency_mico, latency_bitfusion, latency_proxy)
- `-t, --trails`: Number of random seed trials
- `-m, --mode`: Objective mode (ptq_acc, qat_acc)

### Workflow 4: Code Generation

```python
from MiCoCodeGen import MiCoCodeGen
from models import LeNet
from MiCoUtils import fuse_model
import torch

# Load and prepare model
model = LeNet(1)
model.set_qscheme([[8,6,6,4,4], [8,8,8,8,8]])  # Optimal MPQ config
model.load_state_dict(torch.load('output/ckpt/lenet_mnist.pth'))
model = fuse_model(model)  # Fuse Conv-BN
model.eval()

# Generate C code
codegen = MiCoCodeGen(model, align_to=32)
codegen.forward(torch.randn(1, 1, 28, 28))
codegen.convert("output", "lenet_mnist")
```

**Output**:
- `output/lenet_mnist.h`: Model header with C code
- `output/lenet_mnist.bin`: Binary weights

### Workflow 5: Compile and Deploy

```bash
# Navigate to project directory
cd project

# Clean previous builds
make clean

# Compile for target hardware
make MAIN=main TARGET=host OPT=unroll      # Local host
make MAIN=main TARGET=vexii OPT=simd       # VexiiRiscv
make MAIN=main TARGET=rocket               # Rocket CPU (Chipyard)
make MAIN=main TARGET=spike                # Spike simulator
```

**Run on host**:
```bash
make run-host
```

**Run on VexiiRiscv**: Load ELF to simulator (see [VexiiRiscv docs](https://spinalhdl.github.io/VexiiRiscv-RTD/master/VexiiRiscv/HowToUse/index.html#run-a-simulation))

### Workflow 6: Hardware-Aware Search

```python
# Example: deploy/lenet_on_mico.py
from MiCoProxy import get_mico_matmul_proxy, get_mico_conv2d_proxy

# Setup proxy models
matmul_proxy = get_mico_matmul_proxy(mico_type="small")
conv2d_proxy = get_mico_conv2d_proxy(mico_type="small")

# Create evaluator with hardware latency constraint
evaluator = MiCoEval(
    model, epochs, train_loader, test_loader, ckpt_path,
    objective='ptq_acc',
    constraint='latency_proxy'  # Use proxy-predicted latency
)

# Set proxy in evaluator
evaluator.set_proxy(matmul_proxy, conv2d_proxy)
evaluator.set_mico_target("small")

# Search with hardware awareness
searcher = MiCoSearcher(evaluator, n_inits=10, qtypes=[4,5,6,7,8])
best_config, best_acc = searcher.search(40, 'ptq_acc', 'latency_proxy', max_latency*0.5)
```

---

## Supported Models and Datasets

### Supported Models

| Model | Layers | MPQ Search | C Deploy | DNNWeaver Deploy |
|-------|--------|-----------|---------|-----------------|
| MLP | Linear | ✅ | ✅ | ✅ |
| HARMLP | Linear | ✅ | ✅ | ✅ |
| LeNet | Linear, Conv2D | ✅ | ✅ | ✅ |
| CNN | Linear, Conv2D | ✅ | ✅ | ✅ |
| VGG | Linear, Conv2D | ✅ | ✅ | ✅ |
| ResNet | Linear, BottleNeck | ✅ | ✅ | ✅ |
| MobileNetV2 | Linear, BottleNeck | ✅ | ✅ | ✅ |
| SqueezeNet | Linear, Conv2D | ✅ | ✅ | ✅ |
| ShuffleNet | Linear, Conv2D | ✅ | ✅ | ⏳ Not Yet |
| LLaMa | Transformers | ✅ | ✅ | ⏳ Not Yet |
| ViT | Transformers | ✅ | ⏳ Not Yet | ⏳ Not Yet |
| M5 | Linear, Conv1D | ✅ | ✅ | ⏳ Not Yet |
| DS CNN | Linear, Conv2D | ✅ | ✅ | ✅ |

### Supported Datasets

- **Vision**: MNIST, Fashion MNIST, CIFAR-10, CIFAR-100
- **Text**: TinyStories (for LLaMa)
- **Sensor**: UCI HAR (wearable sensors)
- **Audio**: SpeechCommands (keyword spotting, requires additional packages)

**Load Dataset Example**:
```python
from datasets import mnist, cifar10, cifar100

train_loader, test_loader = mnist(batch_size=64, resize=28)
train_loader, test_loader = cifar10(batch_size=128)
```

---

## Code Examples and Use Cases

### Example 1: Train and Search LeNet

```python
import torch
from models import LeNet
from datasets import mnist
from MiCoEval import MiCoEval
from searchers import MiCoSearcher

# Train model
model = LeNet(in_channels=1)
model.set_qscheme([[8]*model.n_layers, [8]*model.n_layers])
train_loader, test_loader = mnist(batch_size=64)
model.train_loop(epochs=10, train_loader, test_loader)
torch.save(model.state_dict(), 'output/ckpt/lenet_mnist.pth')

# MPQ Search
evaluator = MiCoEval(model, 1, train_loader, test_loader, 
                     'output/ckpt/lenet_mnist.pth')
searcher = MiCoSearcher(evaluator, n_inits=10, qtypes=[4,5,6,7,8])

max_bops = evaluator.eval_bops([8]*model.n_layers*2)
best_config, best_acc = searcher.search(
    40, 'ptq_acc', 'bops', max_bops*0.5
)
print(f"Best Config: {best_config}, Accuracy: {best_acc}")
```

### Example 2: Custom Model with Registry

```python
from MiCoRegistry import MiCoOpRegistry
import torch
import torch.nn as nn

# Define custom layer
class MyCustomActivation(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return x * self.alpha + torch.sigmoid(x)

# Register handler
@MiCoOpRegistry.register_module(MyCustomActivation)
def handle_my_custom_activation(codegen, n, out, module, input_names):
    codegen.add_uninitialized_tensor(n.name, out)
    codegen.add_forward_call(
        "MiCo_custom_activation{dim}d_{dtype}",
        out, n.name, input_names, [module.alpha]
    )

# Use in model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, 3)
        self.custom = MyCustomActivation(alpha=2.0)
    
    def forward(self, x):
        return self.custom(self.conv(x))

# Code generation will automatically use registered handler
model = MyModel()
codegen = MiCoCodeGen(model)
codegen.forward(torch.randn(1, 1, 28, 28))
codegen.convert("output", "my_model")
```

### Example 3: LLaMa Model Quantization

```python
from models import LLaMa
from datasets import tinystories
from MiCoLLaMaGen import MiCoLLaMaGen

# Load LLaMa model
model = LLaMa(dim=288, n_layers=6, n_heads=6, vocab_size=512)
model.set_qscheme([[8]*model.n_layers, [8]*model.n_layers])

# Train on TinyStories
train_loader, test_loader = tinystories(batch_size=32)
model.train_loop(10, train_loader, test_loader)

# Generate C code with specialized LLaMa generator
codegen = MiCoLLaMaGen(model)
codegen.convert("output", "tinyllama")
```

### Example 4: Memory Optimization

```python
from MiCoCodeGen import MiCoCodeGen
from models import ResNet8

# ResNet with skip connections
model = ResNet8(num_classes=10)
model.set_qscheme([[8]*model.n_layers, [8]*model.n_layers])

# Code generation with automatic memory pool optimization
codegen = MiCoCodeGen(model, align_to=32)
codegen.forward(torch.randn(1, 3, 32, 32))

# Before optimization: ~3150 KB
# After optimization: ~768 KB (75.6% savings!)
codegen.convert("output", "resnet8")
```

---

## Advanced Features

### 1. Memory Pool Optimization
**Automatic memory reuse for tensors with non-overlapping lifetimes.**

- Reduces memory by 30-75% depending on model architecture
- Handles complex models with skip connections (ResNet)
- No code changes needed - automatic in `MiCoCodeGen`

**See**: [doc/MEMORY_OPTIMIZATION.md](doc/MEMORY_OPTIMIZATION.md)

### 2. Registry Pattern for Custom Operations
**Extensible architecture for adding custom PyTorch operations.**

- Register custom functions and modules
- No need to modify core `MiCoCodeGen.py`
- Easy testing and maintenance

**See**: [doc/REGISTRY_USAGE.md](doc/REGISTRY_USAGE.md)

### 3. Dimension Transformation
**Reduce search space dimensionality for large models.**

```python
from DimTransform import DimTransform

# For 50-layer ResNet (100 dimensions)
dim_transform = DimTransform(
    in_dim=20,      # Search in 20D space
    out_dim=100,    # Map to 100D model
    method='expand'
)

searcher = MiCoSearcher(
    evaluator, 
    n_inits=10, 
    qtypes=[4,5,6,7,8],
    dim_trans=dim_transform  # Use dimension transformation
)
```

### 4. Hardware Proxy Models
**Predict hardware latency without simulation.**

```python
from MiCoProxy import get_mico_matmul_proxy, get_mico_conv2d_proxy
from MiCoProxy import get_bitfusion_matmul_proxy, get_bitfusion_conv2d_proxy

# MiCo hardware proxies
mico_matmul_proxy = get_mico_matmul_proxy(mico_type="small")  # or "large", "high"
mico_conv2d_proxy = get_mico_conv2d_proxy(mico_type="small")

# BitFusion accelerator proxies
bf_matmul_proxy = get_bitfusion_matmul_proxy()
bf_conv2d_proxy = get_bitfusion_conv2d_proxy()
```

### 5. Chipyard Integration
**Deploy to Chipyard ecosystem designs.**

Supported:
- Rocket CPU
- Boom CPU
- Gemmini (INT8)
- Spike Simulator

**See**: [doc/CHIPYARD_INTEGRATION.md](doc/CHIPYARD_INTEGRATION.md)

```bash
# Activate Chipyard environment
conda activate chipyard-env

# Build for Gemmini
cd project
make MAIN=main TARGET=rocket CHIPYARD_DIR=/path/to/chipyard
```

---

## Best Practices for LLM Agents

### 1. Understanding the Codebase
**When starting with MiCo, LLM agents should:**

✅ **DO:**
- Read `readme.md` first for high-level overview
- Check `examples/` for concrete usage patterns
- Review model files in `models/` to understand architectures
- Look at test files in `tests/` for expected behaviors
- Read specialized docs in `doc/` for advanced features

❌ **DON'T:**
- Modify core files (`MiCoCodeGen.py`, `MiCoQLayers.py`) directly for simple tasks
- Skip the registry pattern when adding custom operations
- Ignore existing utility functions in `MiCoUtils.py`

### 2. Adding New Features

**For Custom Operations:**
```python
# ✅ Good: Use registry pattern
from MiCoRegistry import MiCoOpRegistry

@MiCoOpRegistry.register_function(torch.my_op)
def handle_my_op(codegen, n, out, input_names, input_args):
    # Implementation
    pass

# ❌ Bad: Modify MiCoCodeGen.py directly
# elif function == torch.my_op:  # Don't do this!
```

**For New Models:**
- Inherit from `MiCoModel` or use `from_torch()` for torchvision models
- Implement quantizable layers (Linear, Conv2d, Conv1d)
- Set `n_layers` attribute for search compatibility
- Test with small search (5-10 iterations) before full runs

### 3. Running Experiments

**Search Strategy:**
```python
# ✅ Good: Start with small search for debugging
searcher = MiCoSearcher(evaluator, n_inits=5, qtypes=[6,7,8])
best_x, best_y = searcher.search(10, 'ptq_acc', 'bops', constraint)

# Then scale up for final results
searcher = MiCoSearcher(evaluator, n_inits=20, qtypes=[4,5,6,7,8])
best_x, best_y = searcher.search(100, 'ptq_acc', 'bops', constraint)
```

**Reproducibility:**
```python
# Always set random seeds for reproducible results
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

### 4. File Organization

**Output Structure:**
```
output/
├── ckpt/         # Model checkpoints (.pth)
├── json/         # Layer weights and search traces
├── figs/         # Plots and visualizations
└── txt/          # Text results

# Use consistent naming
# Good: lenet_mnist.pth, lenet_mnist_search.json
# Bad: model1.pth, results.json
```

### 5. Code Generation Workflow

**Complete Pipeline:**
```python
# 1. Load trained model
model = LeNet(1)
model.load_state_dict(torch.load('output/ckpt/lenet_mnist.pth'))

# 2. Apply optimal MPQ configuration
model.set_qscheme([[8,6,6,4,4], [8,8,8,8,8]])

# 3. Fuse layers for deployment
model = fuse_model(model)
model.eval()

# 4. Generate C code
codegen = MiCoCodeGen(model, align_to=32)
codegen.forward(torch.randn(1, 1, 28, 28))
codegen.convert("output", "lenet_mnist")

# 5. Verify generated files exist
# output/lenet_mnist.h and output/lenet_mnist.bin
```

### 6. Debugging Tips

**Common Issues:**

1. **ModuleNotFoundError**: 
   ```bash
   export PYTHONPATH=$PYTHONPATH:.
   ```

2. **CUDA Out of Memory**: Reduce batch size or use CPU
   ```python
   device = torch.device("cpu")  # Force CPU
   ```

3. **Search Not Finding Good Solutions**: 
   - Increase `n_inits` (initial random samples)
   - Try different search algorithms (MiCo, HAQ, Bayes)
   - Relax constraint (increase constraint_factor)

4. **Code Generation Fails**:
   - Check if model is in eval mode: `model.eval()`
   - Verify all layers are supported (check `MiCoRegistry.py`)
   - Use registry pattern to add custom operations

### 7. Performance Optimization

**Search Performance:**
- Use hardware proxy models instead of simulation when possible
- Start with coarse bitwidth options [4,8] before [4,5,6,7,8]
- Use dimension transformation for models with >50 layers
- Parallel evaluation: Set `num_workers` in DataLoader

**Code Generation:**
- Memory alignment (align_to=32) improves performance
- Fuse Conv-BN before code generation
- Use appropriate target (host, vexii, rocket) for compilation

### 8. Testing and Validation

**Always validate changes:**
```python
# 1. Test accuracy preservation
fp_acc = model.test(test_loader)  # FP accuracy
int8_acc = model.test_quantized(test_loader)  # INT8 accuracy
assert int8_acc > fp_acc * 0.95, "Too much accuracy drop"

# 2. Test code generation
codegen = MiCoCodeGen(model)
codegen.forward(test_input)
codegen.convert("output", "test_model")

# 3. Compile and run
# cd project && make clean && make run-host
```

---

## Troubleshooting Guide

### Installation Issues

**Problem**: `conda install pygmo` fails
```bash
# Solution: Try mamba
conda install -c conda-forge mamba
mamba install pygmo
```

**Problem**: CUDA version mismatch
```bash
# Check CUDA version
nvcc --version
# Install matching PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Training Issues

**Problem**: Model accuracy very low
- Check if data normalization is applied
- Verify model architecture matches dataset (input channels, output classes)
- Try longer training (increase epochs)
- Check learning rate (try 0.001, 0.0001)

**Problem**: Training too slow
```python
# Use DataLoader num_workers
train_loader, test_loader = mnist(batch_size=64, num_workers=4)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### Search Issues

**Problem**: Search finds no valid solutions
- Constraint too tight: Increase `constraint_factor` (e.g., 0.3 → 0.5)
- Bad initialization: Increase `n_inits` (e.g., 10 → 20)
- Wrong objective: Verify `objective` and `constraint` types match

**Problem**: Search is very slow
- Use proxy models instead of hardware simulation
- Reduce `n_search` for initial experiments
- Use dimension transformation for large models
- Check evaluator is not retraining model each iteration

### Code Generation Issues

**Problem**: Unsupported operation error
```
RuntimeError: Function 'torch.some_op' is not registered
```
**Solution**: Add custom handler using registry pattern (see [doc/REGISTRY_USAGE.md](doc/REGISTRY_USAGE.md))

**Problem**: Memory optimization not working
- Check if model is properly traced (no dynamic control flow)
- Verify all tensors are properly tracked
- See [doc/MEMORY_OPTIMIZATION.md](doc/MEMORY_OPTIMIZATION.md) for details

**Problem**: Generated C code doesn't compile
- Check submodule is initialized: `git submodule update --init`
- Verify Makefile TARGET is correct
- Check model.h includes correct headers

### Deployment Issues

**Problem**: `make` fails with missing library
```bash
# Update submodules
git submodule update --init --recursive
```

**Problem**: Compiled model gives wrong results
- Verify quantization scheme matches trained model
- Check input data preprocessing (normalization, resizing)
- Compare with PyTorch model output

**Problem**: Chipyard integration fails
- Activate correct conda environment
- Set `CHIPYARD_DIR` environment variable
- Check Chipyard installation is complete

---

## API Reference

### MiCoModel API

```python
class MiCoModel(nn.Module):
    def train_loop(self, epochs, train_loader, test_loader, 
                   lr=0.001, verbose=True)
    def test(self, test_loader)
    def set_qscheme(self, qscheme)  # [[w_bits], [a_bits]]
    def get_qlayers(self)
    n_layers: int  # Number of quantizable layers
```

### MiCoEval API

```python
class MiCoEval:
    def __init__(self, model, epochs, train_loader, test_loader, 
                 pretrained_model, lr=0.0001, objective='ptq_acc',
                 constraint='bops', output_json='...')
    
    def eval_f(self, qscheme: list) -> float  # Evaluate objective
    def eval_bops(self, qscheme: list) -> float  # Calculate BOPs
    def eval_size(self, qscheme: list) -> float  # Calculate model size
    def eval_latency(self, qscheme: list, target: str) -> float  # Hardware latency
    # target can be: 'mico', 'bitfusion', 'proxy', 'host', 'torchao'
    
    def set_proxy(self, matmul_proxy, conv2d_proxy)
    def set_misc_proxy(self, misc_proxy)
    def set_mico_target(self, mico_type: str)
```

### MiCoCodeGen API

```python
class MiCoCodeGen(torch.fx.Interpreter):
    def __init__(self, model, align_to=1)
    
    def forward(self, *args)  # Trace model
    def convert(self, output_dir, model_name)  # Generate C code
    
    # Helper methods for custom handlers
    def add_uninitialized_tensor(self, name, tensor)
    def add_initialized_tensor(self, name, tensor, quant=0, scale=0.0)
    def add_connect_tensor(self, name, tensor)
    def add_forward_call(self, func_name, out, layer_name, 
                        input_names, parameters=None)
```

### Searcher API

```python
class QSearcher:  # Base class
    def __init__(self, evaluator, n_inits, qtypes)
    def search(self, n_iterations, objective, constraint_type, 
               constraint_value) -> Tuple[list, float]
    
# Specific searchers: MiCoSearcher, BayesSearcher, HAQSearcher, NLPSearcher
```

### MiCoUtils API

```python
def list_quantize_layers(model) -> List[nn.Module]
def replace_quantize_layers(model, weight_q, act_q, quant_aware=False, 
                           device=None, use_bias=True)
def fuse_model(model) -> nn.Module
def fuse_model_seq(model) -> nn.Module
def set_to_qforward(model)
def export_layer_weights(model) -> dict
def get_model_macs(model, input_size) -> int
```

### Dataset API

```python
from datasets import mnist, fashion_mnist, cifar10, cifar100
from datasets import tinystories, ucihar, speechcommands

train_loader, test_loader = mnist(
    batch_size=64, 
    num_workers=0,  # DataLoader workers
    resize=28,      # Image size
    shuffle=True
)
```

---

## Quick Reference Cheatsheet

### Common Command Patterns

```bash
# Training
python examples/lenet_mnist.py
python examples/mpq_train.py -h  # See all options

# Search
python examples/lenet_mnist_search.py
python examples/mpq_search.py model_name -n 40 -c 0.5

# Code Generation (in Python)
codegen = MiCoCodeGen(model, align_to=32)
codegen.forward(test_input)
codegen.convert("output", "model_name")

# Compilation
cd project
make clean
make MAIN=main TARGET=host OPT=unroll
make run-host
```

### Typical MPQ Workflow

```python
# 1. Train
model.train_loop(epochs=10, train_loader, test_loader)
torch.save(model.state_dict(), 'ckpt.pth')

# 2. Search
evaluator = MiCoEval(model, 1, train_loader, test_loader, 'ckpt.pth')
searcher = MiCoSearcher(evaluator, n_inits=10, qtypes=[4,5,6,7,8])
best_config, best_acc = searcher.search(40, 'ptq_acc', 'bops', constraint)

# 3. Generate Code
model.set_qscheme([best_config[:n], best_config[n:]])
model = fuse_model(model)
codegen = MiCoCodeGen(model)
codegen.forward(test_input)
codegen.convert("output", "model_name")

# 4. Compile & Deploy
# cd project && make && make run-host
```

---

## Additional Resources

### Documentation
- [README.md](readme.md) - Main project overview
- [doc/REGISTRY_USAGE.md](doc/REGISTRY_USAGE.md) - Custom operation registry
- [doc/MEMORY_OPTIMIZATION.md](doc/MEMORY_OPTIMIZATION.md) - Memory pool optimization
- [doc/CHIPYARD_INTEGRATION.md](doc/CHIPYARD_INTEGRATION.md) - Chipyard integration

### Related Projects
- [ucb-bar/Baremetal-NN](https://github.com/ucb-bar/Baremetal-NN) - Inspiration for code generation
- [mit-han-lab/haq](https://github.com/mit-han-lab/haq) - HAQ searcher implementation
- [karpathy/llama2.c](https://github.com/karpathy/llama2.c) - LLaMa2 C implementation

### Publication
```bibtex
@inproceedings{jiang2025mico,
  title={MiCo: End-to-End Mixed Precision Neural Network Co-Exploration Framework for Edge AI},
  author={Jiang, Zijun and Lyu, Yangdi},
  booktitle={ICCAD},
  year={2025}
}
```

### Roadmap
Check the [Roadmap](../../issues/1) for planned features and improvements.

---

## Contributing Guidelines for LLM Agents

When modifying or extending MiCo:

1. **Use Registry Pattern**: For new operations, use `@MiCoOpRegistry.register_*()` decorators
2. **Follow Naming Conventions**: Use `handle_<op_name>` for handlers, `<model>_<dataset>` for scripts
3. **Add Tests**: Create tests in `tests/` for new features
4. **Document Changes**: Update relevant `.md` files
5. **Maintain Compatibility**: Don't break existing APIs unless necessary
6. **Optimize Memory**: Use memory pools for code generation
7. **Support Multiple Targets**: Test on different hardware targets when possible

---

**Last Updated**: 2026-01-23  
**Version**: 1.0  
**Maintainers**: HKUSTGZ-MICS-LYU

For questions or issues, please open an issue on GitHub or refer to the roadmap.
