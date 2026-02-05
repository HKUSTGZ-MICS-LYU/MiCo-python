"""
Test script demonstrating two-stage proxy integration with MiCo workflows.

This script shows how to use the two-stage predictor in:
1. Hardware latency evaluation
2. Mixed precision quantization (MPQ) search (simulated)
"""

import sys
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, r2_score

sys.path.append('.')
from MiCoProxy import (
    get_mico_matmul_two_stage_proxy,
    get_mico_conv2d_two_stage_proxy,
    get_mico_matmul_proxy,
    get_mico_conv2d_proxy
)


def suppress_output(func):
    """Decorator to suppress stdout during function execution."""
    def wrapper(*args, **kwargs):
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        result = func(*args, **kwargs)
        sys.stdout = old_stdout
        return result
    return wrapper


@suppress_output
def get_proxies(use_two_stage=True, mico_type='small'):
    """Get proxy models (two-stage or original)."""
    if use_two_stage:
        matmul_proxy = get_mico_matmul_two_stage_proxy(mico_type=mico_type)
        conv2d_proxy = get_mico_conv2d_two_stage_proxy(mico_type=mico_type)
    else:
        matmul_proxy = get_mico_matmul_proxy(mico_type=mico_type)
        conv2d_proxy = get_mico_conv2d_proxy(mico_type=mico_type)
    return matmul_proxy, conv2d_proxy


def simulate_layer_latency(proxy, layer_config, kernel_type='matmul'):
    """
    Simulate latency prediction for a neural network layer.
    
    Args:
        proxy: Trained proxy model
        layer_config: Dict with layer parameters
        kernel_type: 'matmul' or 'conv2d'
    """
    if kernel_type == 'matmul':
        # Extract MatMul parameters
        N = layer_config.get('N', 1)
        M = layer_config['M']
        K = layer_config['K']
        QA = layer_config['QA']
        QW = layer_config['QW']
        
        MACS = N * M * K
        X = np.array([[MACS, M, K, QA, QW]])
        
    elif kernel_type == 'conv2d':
        # Extract Conv2D parameters
        H = layer_config['H']
        W = layer_config['W']
        C = layer_config['C']
        K = layer_config['K']
        Ks = layer_config['Ks']
        S = layer_config.get('S', 1)
        QA = layer_config['QA']
        QW = layer_config['QW']
        
        H_out = (H - Ks) / S + 1
        W_out = (W - Ks) / S + 1
        MACS = H_out * W_out * C * K * Ks * Ks
        X = np.array([[MACS, H, W, C, K, Ks, S, QA, QW]])
    
    return proxy.predict(X)[0]


def test_single_layer_prediction():
    """Test 1: Single layer latency prediction."""
    print("\n" + "="*70)
    print("TEST 1: Single Layer Latency Prediction")
    print("="*70)
    
    # Get proxies
    print("\nLoading proxies...")
    matmul_original, _ = get_proxies(use_two_stage=False, mico_type='small')
    matmul_two_stage, _ = get_proxies(use_two_stage=True, mico_type='small')
    print("✓ Proxies loaded")
    
    # Define test layer (e.g., a linear layer)
    layer = {
        'M': 128,   # Output features
        'K': 512,   # Input features
        'QA': 4,    # Activation quantization
        'QW': 4     # Weight quantization
    }
    
    print(f"\nTest layer configuration:")
    print(f"  Linear layer: {layer['K']} → {layer['M']}")
    print(f"  Precision: QA={layer['QA']}, QW={layer['QW']}")
    
    # Predict latency
    latency_original = simulate_layer_latency(matmul_original, layer, 'matmul')
    latency_two_stage = simulate_layer_latency(matmul_two_stage, layer, 'matmul')
    
    print(f"\nPredicted latency:")
    print(f"  Original proxy:   {latency_original:>10.2f} μs")
    print(f"  Two-stage proxy:  {latency_two_stage:>10.2f} μs")
    print(f"  Difference:       {abs(latency_two_stage - latency_original):>10.2f} μs")
    
    return True


def test_mixed_precision_search():
    """Test 2: Simulate MPQ search with two-stage proxy."""
    print("\n" + "="*70)
    print("TEST 2: Mixed Precision Quantization Search Simulation")
    print("="*70)
    
    # Get proxies
    print("\nLoading proxies...")
    matmul_original, conv_original = get_proxies(use_two_stage=False, mico_type='small')
    matmul_two_stage, conv_two_stage = get_proxies(use_two_stage=True, mico_type='small')
    print("✓ Proxies loaded")
    
    # Define a simple 3-layer network
    network_layers = [
        {'type': 'conv2d', 'H': 28, 'W': 28, 'C': 1, 'K': 16, 'Ks': 3, 'S': 1},
        {'type': 'conv2d', 'H': 26, 'W': 26, 'C': 16, 'K': 32, 'Ks': 3, 'S': 1},
        {'type': 'matmul', 'M': 10, 'K': 12800}  # Flattened
    ]
    
    print(f"\nSimulated network: {len(network_layers)} layers")
    print(f"  Layer 1: Conv2D (28x28x1 → 26x26x16)")
    print(f"  Layer 2: Conv2D (26x26x16 → 24x24x32)")
    print(f"  Layer 3: Linear (12800 → 10)")
    
    # Test different bitwidth configurations
    configs = [
        ([8, 8, 8], [8, 8, 8]),  # INT8 baseline
        ([4, 4, 4], [4, 4, 4]),  # INT4
        ([8, 6, 4], [8, 6, 4]),  # Mixed precision
    ]
    
    print(f"\nTesting {len(configs)} quantization configurations...")
    print(f"\n{'Config':<20} {'Original (μs)':<15} {'Two-Stage (μs)':<15} {'Speedup':<10}")
    print("-"*70)
    
    for weight_bits, act_bits in configs:
        # Compute total latency with original proxy
        total_original = 0
        for i, layer in enumerate(network_layers):
            layer_config = layer.copy()
            layer_config['QW'] = weight_bits[i]
            layer_config['QA'] = act_bits[i]
            
            if layer['type'] == 'matmul':
                latency = simulate_layer_latency(matmul_original, layer_config, 'matmul')
            else:
                latency = simulate_layer_latency(conv_original, layer_config, 'conv2d')
            
            total_original += latency
        
        # Compute total latency with two-stage proxy
        total_two_stage = 0
        for i, layer in enumerate(network_layers):
            layer_config = layer.copy()
            layer_config['QW'] = weight_bits[i]
            layer_config['QA'] = act_bits[i]
            
            if layer['type'] == 'matmul':
                latency = simulate_layer_latency(matmul_two_stage, layer_config, 'matmul')
            else:
                latency = simulate_layer_latency(conv_two_stage, layer_config, 'conv2d')
            
            total_two_stage += latency
        
        config_str = f"W{weight_bits} A{act_bits}"
        speedup = total_original / total_two_stage if total_two_stage > 0 else 1.0
        
        print(f"{config_str:<20} {total_original:>12.2f}    {total_two_stage:>12.2f}    {speedup:>8.2f}x")
    
    return True


def test_precision_sweep():
    """Test 3: Sweep across different precisions."""
    print("\n" + "="*70)
    print("TEST 3: Precision Sweep Analysis")
    print("="*70)
    
    # Get proxies
    print("\nLoading proxies...")
    matmul_original, _ = get_proxies(use_two_stage=False, mico_type='small')
    matmul_two_stage, _ = get_proxies(use_two_stage=True, mico_type='small')
    print("✓ Proxies loaded")
    
    # Fixed layer configuration
    base_layer = {'M': 256, 'K': 256}
    
    print(f"\nTest layer: Linear({base_layer['K']} → {base_layer['M']})")
    print(f"\nSweeping precisions...")
    print(f"\n{'Precision':<15} {'Original (μs)':<15} {'Two-Stage (μs)':<15} {'Δ%':<10}")
    print("-"*70)
    
    precisions = [(8, 8), (6, 6), (4, 4), (2, 2), (8, 4), (4, 8)]
    
    for qa, qw in precisions:
        layer = base_layer.copy()
        layer['QA'] = qa
        layer['QW'] = qw
        
        lat_orig = simulate_layer_latency(matmul_original, layer, 'matmul')
        lat_two = simulate_layer_latency(matmul_two_stage, layer, 'matmul')
        
        diff_pct = (lat_two - lat_orig) / lat_orig * 100 if lat_orig > 0 else 0
        
        print(f"QA={qa}, QW={qw:<5} {lat_orig:>12.2f}    {lat_two:>12.2f}    {diff_pct:>8.2f}%")
    
    return True


def main():
    """Run all integration tests."""
    print("\n" + "="*70)
    print(" Two-Stage Proxy Integration Tests")
    print("="*70)
    
    tests = [
        ("Single Layer Prediction", test_single_layer_prediction),
        ("MPQ Search Simulation", test_mixed_precision_search),
        ("Precision Sweep", test_precision_sweep),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                status = "✓ PASSED"
            else:
                failed += 1
                status = "✗ FAILED"
        except Exception as e:
            failed += 1
            status = f"✗ ERROR: {e}"
        
        print(f"\n{test_name}: {status}")
    
    print("\n" + "="*70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed == 0:
        print("\n🎉 All integration tests passed successfully!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
