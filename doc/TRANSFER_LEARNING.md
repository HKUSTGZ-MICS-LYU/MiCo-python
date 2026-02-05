# MiCoProxy Transfer Learning Guide

This document describes the transfer learning capabilities added to MiCoProxy for efficiently adapting proxy models across different hardware targets.

## Overview

Transfer learning allows training a proxy model on one hardware target (source domain) and adapting it to another target (target domain) with minimal additional profiling data. This is particularly useful when:

- Profiling a new hardware configuration is expensive
- You want to quickly estimate latency on a new target
- You have limited data from the target hardware

## Key Features

### 1. Pretrain and Finetune Workflow

The `MiCoProxy` class now supports transfer learning through two new methods:

```python
from MiCoProxy import MiCoProxy, LogRandomForestRegressor, load_proxy_data

# Create proxy
model = LogRandomForestRegressor(random_state=42)
proxy = MiCoProxy(model, preprocess='cbops+')

# Load source and target data
X_source, y_source = load_proxy_data('benchmark_results/mico_small_matmul_zoo.csv', 'matmul')
X_target, y_target = load_proxy_data('benchmark_results/mico_high_matmul_zoo.csv', 'matmul')

# Pretrain on source domain
proxy.pretrain(X_source, y_source)

# Fine-tune on target domain with only 10% of target data
result = proxy.finetune(X_target, y_target, finetune_ratio=0.1, strategy='combined')

# Use for prediction
predictions = proxy.predict(X_target)
```

### 2. Fine-tuning Strategies

Three fine-tuning strategies are available:

- **`combined`** (default): Combines source and target data for retraining. Best for maintaining source domain knowledge while adapting.
- **`target_only`**: Retrains only on target data. Useful as a baseline comparison.
- **`weighted`**: Weights target samples higher by oversampling. Useful when target domain differs significantly.

### 3. Model Types

Three model types are available for transfer learning:

- **`random_forest`** (default): LogRandomForestRegressor - Tree-based ensemble. Generally performs better for this task due to small dataset sizes and well-engineered features.
- **`mlp`**: LogMLPRegressor - sklearn MLP with warm_start for weight transfer. Good for simple transfer scenarios.
- **`torch_mlp`**: TorchMLPRegressor - PyTorch MLP with advanced transfer learning techniques:
  - **Layer freezing**: Freeze feature extractor layers, fine-tune only the head
  - **Discriminative learning rates**: Use lower LR for pretrained layers
  - **Gradual unfreezing**: Progressively unfreeze layers during training

```python
# Use PyTorch MLP with layer freezing for transfer learning
proxy, results = get_transfer_proxy(
    source_type='mico_small',
    target_type='mico_high',
    kernel_type='matmul',
    finetune_ratio=0.1,
    model_type='torch_mlp',
    freeze_strategy='freeze_extractor'  # or 'discriminative_lr', 'none'
)
```

### 4. Freeze Strategies (for torch_mlp)

When using `model_type='torch_mlp'`, you can control how transfer learning is performed:

- **`freeze_extractor`** (default): Freeze feature extraction layers, train only the head (final layers). Best for domain shift scenarios.
- **`discriminative_lr`**: Use lower learning rate for pretrained layers, higher for head. Allows gradual adaptation.
- **`none`**: No freezing, fine-tune all layers with lower learning rate.

### 5. Utility Functions

#### `get_transfer_proxy()`
Create a proxy with transfer learning in one call:

```python
from MiCoProxy import get_transfer_proxy

proxy, results = get_transfer_proxy(
    source_type='mico_small',
    target_type='mico_high',
    kernel_type='matmul',
    finetune_ratio=0.1,
    strategy='combined',
    model_type='torch_mlp',  # 'random_forest', 'mlp', or 'torch_mlp'
    freeze_strategy='freeze_extractor',  # for torch_mlp only
    verbose=True
)

print(f"MAPE improvement: {results['mape_improvement']*100:.1f}%")
```

#### `compare_model_types_for_transfer()`
Compare all model types for transfer learning:

```python
from MiCoProxy import compare_model_types_for_transfer

results = compare_model_types_for_transfer(
    source_type='mico_small',
    target_type='mico_high',
    kernel_type='matmul',
    finetune_ratios=[0.05, 0.1, 0.2, 0.5],
    n_trials=3
)
# Returns comparison metrics for both model types
```

#### `compare_transfer_directions()`
Compare transfer learning in both directions:

```python
from MiCoProxy import compare_transfer_directions

results = compare_transfer_directions(
    kernel_type='matmul',
    finetune_ratio=0.1
)
# Returns results for both small->high and high->small
```

#### `evaluate_transfer_learning()`
Systematically evaluate across multiple ratios and strategies:

```python
from MiCoProxy import evaluate_transfer_learning

results = evaluate_transfer_learning(
    source_type='mico_small',
    target_type='mico_high',
    kernel_type='matmul',
    ratios=[0.05, 0.1, 0.2, 0.5],
    strategies=['combined', 'target_only'],
    n_trials=5
)
```

#### `explore_cross_target_transfer()`
Explore transfer from one accelerator to another:

```python
from MiCoProxy import explore_cross_target_transfer

results = explore_cross_target_transfer(
    source_type='bitfusion',
    target_types=['mico_small', 'mico_high'],
    kernel_type='matmul',
    finetune_ratios=[0.1, 0.2, 0.5]
)
```

## Validation Results

### MatMul Kernels

Transfer learning between `mico_small` and `mico_high` for matmul kernels shows:

| Target Data Ratio | Transfer (small→high) | From Scratch | Benefit |
|-------------------|----------------------|--------------|---------|
| 5%                | 45.2% MAPE           | 645% MAPE    | **93% better** |
| 10%               | 43.5% MAPE           | 72.6% MAPE   | 40% better |
| 20%               | 39.9% MAPE           | 19.1% MAPE   | Scratch better |
| 100%              | 21.2% MAPE           | 3.2% MAPE    | Full data works best |

**Key Finding**: With very limited target data (5-10%), transfer learning dramatically outperforms training from scratch.

### Conv2D Kernels

| Target Data Ratio | Transfer (small→high) | From Scratch | Notes |
|-------------------|----------------------|--------------|-------|
| 5%                | 48.9% MAPE           | 37.8% MAPE   | Scratch better |
| 10%               | 47.0% MAPE           | 23.8% MAPE   | Scratch better |
| 100%              | 22.7% MAPE           | 2.5% MAPE    | Full data works best |

**Key Finding**: For conv2d with more data available, training from scratch is generally better, but transfer provides stability with scarce data.

### Cross-Target Transfer (BitFusion → MiCo)

Transfer learning from BitFusion to VexiiRiscv/MiCo targets shows:

- Improvements of 5-50% depending on target data ratio
- More target data leads to better adaptation
- Cross-architecture transfer is feasible but requires more target data

### Model Comparison: Random Forest vs MLP

We evaluated tree-based (Random Forest), sklearn MLP, and PyTorch MLP with layer freezing for transfer learning:

**MatMul Kernels (small → high):**

| Data Ratio | Random Forest | sklearn MLP | PyTorch MLP (freeze) | Best |
|------------|---------------|-------------|---------------------|------|
| 10% | 43.3% MAPE | 40.7% MAPE | 49.2% MAPE | sklearn MLP |
| 20% | 39.9% MAPE | 39.1% MAPE | 43.2% MAPE | sklearn MLP |
| 50% | 31.9% MAPE | 40.4% MAPE | 41.8% MAPE | RF |

**Key Findings**: 
- sklearn MLP (`mlp`) performs best at low data ratios (10-20%) due to warm_start enabling true weight transfer
- Random Forest (`random_forest`) is most stable and performs best with more data (50%+)
- PyTorch MLP with layer freezing (`torch_mlp`) provides advanced transfer learning techniques but may need hyperparameter tuning for small datasets
- The small dataset sizes (99-736 samples) limit the benefit of complex neural architectures

**PyTorch MLP Freeze Strategies:**

| Strategy | Description | Best For |
|----------|-------------|----------|
| `freeze_extractor` | Freeze feature layers, train only head | Domain adaptation with limited data |
| `discriminative_lr` | Lower LR for pretrained layers | Gradual adaptation |
| `none` | Full fine-tuning with lower LR | When domains are similar |

**Recommendations**:
- For small datasets: Use `random_forest` (default) for stability
- For transfer with 10-20% target data: Try `mlp` for potentially better results
- For experimentation: Use `torch_mlp` with different freeze strategies to find optimal configuration

## Usage Recommendations

1. **Use transfer learning when:**
   - Target profiling data is very limited (<20% of full dataset)
   - Quick prototyping is needed for new hardware
   - Source and target have similar characteristics

2. **Use training from scratch when:**
   - Full target profiling data is available
   - Maximum accuracy is required
   - Source and target differ significantly

3. **Model selection:**
   - Use `model_type='random_forest'` (default) for stability
   - Try `model_type='mlp'` for best transfer at 10-20% data ratio
   - Use `model_type='torch_mlp'` for advanced techniques with layer freezing

4. **Best practices:**
   - Start with 10-20% target data for transfer learning
   - Use `combined` strategy as default
   - Validate on held-out target data when possible

## Running the Validation Script

```bash
# Quick validation
python examples/proxy_transfer_learning.py --quick

# Comprehensive analysis
python examples/proxy_transfer_learning.py --comprehensive

# Specific kernel type
python examples/proxy_transfer_learning.py --kernel matmul
```

## API Reference

### `MiCoProxy.pretrain(X_source, y_source)`
Pretrain the model on source domain data.

### `MiCoProxy.finetune(X_target, y_target, finetune_ratio=1.0, strategy='combined')`
Fine-tune a pretrained model on target domain data.

### `MiCoProxy.is_pretrained()`
Check if model has been pretrained.

### `load_proxy_data(profile_dataset, kernel_type)`
Load and preprocess proxy training data from a CSV file.

### `get_transfer_proxy(...)`
Create a proxy model using transfer learning.

### `evaluate_transfer_learning(...)`
Systematically evaluate transfer learning across configurations.

### `compare_transfer_directions(...)`
Compare transfer learning in both directions.

### `explore_cross_target_transfer(...)`
Explore transfer from one hardware target to another.
