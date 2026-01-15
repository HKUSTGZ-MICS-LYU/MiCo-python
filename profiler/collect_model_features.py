#!/usr/bin/env python3
"""
Model Feature Collector Script

This script loads models from the model_zoo and extracts layer feature
configurations (parameter sets) for hardware profiling. It collects:
- MatMul/Linear layer features: (N, M, K)
- Conv2D layer features: (HW, C, K, KS, S)
- Pooling layer features: (C, HW, K, S)

Usage:
    python profiler/collect_model_features.py --output profile_list.csv
    python profiler/collect_model_features.py --model cmsiscnn_cifar10 --output features.csv
"""

import argparse
import csv
import sys
import os
from typing import Dict, List, Tuple, Optional, Any

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


def get_layer_features(module: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, List[Tuple]]:
    """
    Extract layer features from a model by tracing.
    
    Args:
        module: PyTorch model
        input_shape: Expected input shape (batch, ...)
        
    Returns:
        Dictionary with 'matmul', 'conv2d', 'pooling' feature lists
    """
    features = {
        'matmul': [],
        'conv2d': [],
        'pooling': []
    }
    
    # Hook to capture layer info
    hooks = []
    
    def linear_hook(module, input, output):
        if len(input) > 0 and hasattr(input[0], 'shape'):
            in_features = module.in_features
            out_features = module.out_features
            batch_size = input[0].shape[0] if len(input[0].shape) > 1 else 1
            # N = batch, M = out_features, K = in_features
            features['matmul'].append((batch_size, out_features, in_features))
    
    def conv2d_hook(module, input, output):
        if len(input) > 0 and hasattr(input[0], 'shape'):
            in_channels = module.in_channels
            out_channels = module.out_channels
            kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
            stride = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            # Get input HW
            if len(input[0].shape) >= 4:
                H, W = input[0].shape[2], input[0].shape[3]
                # HW, C, K, KS, S
                features['conv2d'].append((H, in_channels, out_channels, kernel_size, stride))
    
    def pool_hook(module, input, output):
        if len(input) > 0 and hasattr(input[0], 'shape'):
            kernel_size = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
            stride = module.stride if isinstance(module.stride, int) else module.stride[0]
            if len(input[0].shape) >= 4:
                C, H, W = input[0].shape[1], input[0].shape[2], input[0].shape[3]
                # C, HW, K, S
                features['pooling'].append((C, H, kernel_size, stride))
    
    # Register hooks
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv2d_hook))
        elif isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
            hooks.append(m.register_forward_hook(pool_hook))
    
    # Run forward pass
    try:
        device = next(module.parameters()).device
        dummy_input = torch.randn(input_shape).to(device)
        with torch.no_grad():
            module(dummy_input)
    except Exception as e:
        print(f"Warning: Forward pass failed: {e}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return features


def get_model_input_shape(model_name: str) -> Tuple[int, ...]:
    """Get the expected input shape for a model."""
    # CIFAR models: 32x32x3
    if 'cifar' in model_name.lower():
        return (1, 3, 32, 32)
    # MNIST models: 28x28x1
    elif 'mnist' in model_name.lower() or 'lenet' in model_name.lower():
        return (1, 1, 28, 28)
    # ImageNet models: 224x224x3
    elif 'imagenet' in model_name.lower() or 'mobilenet' in model_name.lower() or 'vgg' in model_name.lower():
        return (1, 3, 224, 224)
    # MLP
    elif 'mlp' in model_name.lower() and 'har' not in model_name.lower():
        return (1, 256)  # Flattened input
    # HAR MLP
    elif 'har' in model_name.lower():
        return (1, 9, 128)  # UCI HAR format
    # KWS models
    elif 'kws' in model_name.lower() or 'm5' in model_name.lower():
        return (1, 1, 16000)  # Audio waveform
    # Default
    else:
        return (1, 3, 32, 32)


def collect_features_from_zoo(model_names: Optional[List[str]] = None) -> Dict[str, List[Tuple]]:
    """
    Collect features from models in the zoo.
    
    Args:
        model_names: List of model names to process, or None for all
        
    Returns:
        Dictionary of aggregated features
    """
    from models.model_zoo import from_zoo
    
    all_features = {
        'matmul': [],
        'conv2d': [],
        'pooling': []
    }
    
    # Default models to collect from
    if model_names is None:
        model_names = [
            'cmsiscnn_cifar10',
            'vgg_cifar10',
            'resnet8_cifar10',
            'resnet18_cifar10',
            'mobilenetv2_cifar100',
            'lenet_mnist',
        ]
    
    for name in model_names:
        print(f"Processing {name}...")
        try:
            model, _, _ = from_zoo(name, shuffle=False, batch_size=1)
            if model is None:
                print(f"  Skipping {name}: model not loaded")
                continue
            
            input_shape = get_model_input_shape(name)
            features = get_layer_features(model, input_shape)
            
            print(f"  Found {len(features['matmul'])} linear layers")
            print(f"  Found {len(features['conv2d'])} conv2d layers")
            print(f"  Found {len(features['pooling'])} pooling layers")
            
            all_features['matmul'].extend(features['matmul'])
            all_features['conv2d'].extend(features['conv2d'])
            all_features['pooling'].extend(features['pooling'])
            
        except Exception as e:
            print(f"  Error processing {name}: {e}")
    
    # Remove duplicates
    for key in all_features:
        all_features[key] = list(set(all_features[key]))
    
    return all_features


def save_features_to_csv(features: Dict[str, List[Tuple]], output_prefix: str):
    """
    Save collected features to CSV files.
    
    Args:
        features: Dictionary of features
        output_prefix: Output file prefix (without extension)
    """
    # Save MatMul features
    if features['matmul']:
        matmul_file = f"{output_prefix}_matmul.csv"
        with open(matmul_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['N', 'M', 'K'])
            for row in sorted(features['matmul']):
                writer.writerow(row)
        print(f"Saved {len(features['matmul'])} matmul configs to {matmul_file}")
    
    # Save Conv2D features
    if features['conv2d']:
        conv2d_file = f"{output_prefix}_conv2d.csv"
        with open(conv2d_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['HW', 'C', 'K', 'KS', 'S'])
            for row in sorted(features['conv2d']):
                writer.writerow(row)
        print(f"Saved {len(features['conv2d'])} conv2d configs to {conv2d_file}")
    
    # Save Pooling features
    if features['pooling']:
        pooling_file = f"{output_prefix}_pooling.csv"
        with open(pooling_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['C', 'HW', 'K', 'S'])
            for row in sorted(features['pooling']):
                writer.writerow(row)
        print(f"Saved {len(features['pooling'])} pooling configs to {pooling_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Collect layer features from model zoo for profiling'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='profile_features',
        help='Output file prefix (default: profile_features)'
    )
    parser.add_argument(
        '--model', '-m', type=str, nargs='*', default=None,
        help='Specific model(s) to process (default: all supported models)'
    )
    args = parser.parse_args()
    
    print("Collecting layer features from model zoo...")
    features = collect_features_from_zoo(args.model)
    
    print(f"\nTotal unique features collected:")
    print(f"  MatMul: {len(features['matmul'])}")
    print(f"  Conv2D: {len(features['conv2d'])}")
    print(f"  Pooling: {len(features['pooling'])}")
    
    save_features_to_csv(features, args.output)
    print("\nDone!")


if __name__ == '__main__':
    main()
