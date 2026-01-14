from SimUtils import benchmark_bitfusion_matmul, benchmark_bitfusion_conv2d
import argparse
from tqdm import tqdm
import os

# Import the shared sampler classes
from profiler.adaptive import AdaptiveConv2DProfiler, AdaptiveMatMulProfiler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark BitFusion with adaptive random sampling'
    )
    parser.add_argument(
        '--matmul_samples', type=int, default=50, 
        help='Number of MatMul samples'
    )
    parser.add_argument(
        '--conv2d_samples', type=int, default=50, 
        help='Number of Conv2D samples'
    )
    parser.add_argument(
        '--strategy', type=str, default='adaptive',
        choices=['random', 'corner', 'prior', 'lhs', 'adaptive'],
        help='Sampling strategy (default: adaptive)'
    )
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs('benchmark_results', exist_ok=True)

    # MatMul Configuration
    matmul_ranges = {
        'N': [16], 
        'M': (10, 4096),
        'K': (10, 4096)
    }

    
    # matmul_profiler = AdaptiveMatMulProfiler(
    #     ranges=matmul_ranges,
    #     benchmark_fn=benchmark_bitfusion_matmul,
    #     seed = 0
    # )

    # matmul_samples = matmul_profiler.run(init_samples=20, iterations=3, samples_per_iteration=10)

    # with open('benchmark_results/bitfusion_matmul_samples.csv', 'w') as f:
    #     f.write('N,M,K,QA,QW,Time\n')
    #     for row in matmul_samples:
    #         f.write(','.join(map(str, row)) + '\n')
            
    # Conv2D Configuration
    conv2d_ranges = {
        'HW': (4, 64),
        'C': (3, 1024),
        'K': (16, 2048),
        'KS': [1, 3, 5, 7],
        'S' : [1, 2] 
    }

    conv2d_profiler = AdaptiveConv2DProfiler(
        ranges=conv2d_ranges,
        benchmark_fn=benchmark_bitfusion_conv2d,
        seed = 0
    )

    conv2d_samples = conv2d_profiler.run(
        init_samples=20, iterations=1, samples_per_iteration=0)

    with open('benchmark_results/bitfusion_conv2d_samples.csv', 'w') as f:
        f.write('H,W,C,K,Ks,QA,QW,Time\n')
        for row in conv2d_samples:
            f.write(','.join(map(str, row)) + '\n')
