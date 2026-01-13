from SimUtils import benchmark_bitfusion_matmul, benchmark_bitfusion_conv2d
import argparse
from tqdm import tqdm
import os

# Import the shared sampler classes
from profile.sampler import MatMulSampler, Conv2DSampler


def generate_matmul_samples(num_samples, ranges, strategy='adaptive'):
    """
    Generate MatMul samples using the shared sampler.
    
    Args:
        num_samples: Number of samples to generate
        ranges: Dictionary of parameter ranges
        strategy: Sampling strategy ('random', 'corner', 'prior', 'lhs', 'adaptive')
        
    Returns:
        List of (N, M, K) tuples
    """
    sampler = MatMulSampler(ranges=ranges, strategy=strategy)
    return sampler.generate(num_samples=num_samples)


def generate_conv2d_samples(num_samples, ranges, strategy='adaptive'):
    """
    Generate Conv2D samples using the shared sampler.
    
    Args:
        num_samples: Number of samples to generate
        ranges: Dictionary of parameter ranges
        strategy: Sampling strategy ('random', 'corner', 'prior', 'lhs', 'adaptive')
        
    Returns:
        List of (HW, C, K, KS) tuples
    """
    sampler = Conv2DSampler(ranges=ranges, strategy=strategy)
    return sampler.generate(num_samples=num_samples)

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
        'M': (16, 4096),
        'K': (16, 4096)
    }

    print(f"Generating {args.matmul_samples} MatMul samples "
          f"using '{args.strategy}' strategy...")
    matmul_samples = generate_matmul_samples(
        args.matmul_samples, matmul_ranges, strategy=args.strategy
    )
    
    matmul_dataset = []
    for N, M, K in tqdm(matmul_samples, desc="Running MatMul Benchmark"):
        try:
            res = benchmark_bitfusion_matmul(N, M, K)
            matmul_dataset += res
        except Exception as e:
            print(f"Failed for N={N}, M={M}, K={K}: {e}")

    with open('benchmark_results/bitfusion_matmul_samples.csv', 'w') as f:
        f.write('N,M,K,QA,QW,Time\n')
        for row in matmul_dataset:
            f.write(','.join(map(str, row)) + '\n')
            
    # Conv2D Configuration
    conv2d_ranges = {
        'HW': (4, 64),
        'C': (3, 1024),
        'K': (16, 2048),
        'KS': [1, 3, 5, 7]
    }

    print(f"Generating {args.conv2d_samples} Conv2D samples "
          f"using '{args.strategy}' strategy...")
    conv2d_samples = generate_conv2d_samples(
        args.conv2d_samples, conv2d_ranges, strategy=args.strategy
    )
    
    conv2d_dataset = []
    for HW, C, K, KS in tqdm(conv2d_samples, desc="Running Conv2D Benchmark"):
        try:
            res = benchmark_bitfusion_conv2d(HW, HW, C, K, KS)
            conv2d_dataset += res
        except Exception as e:
            print(f"Failed for HW={HW}, C={C}, K={K}, KS={KS}: {e}")

    with open('benchmark_results/bitfusion_conv2d_samples.csv', 'w') as f:
        f.write('H,W,C,K,Ks,QA,QW,Time\n')
        for row in conv2d_dataset:
            f.write(','.join(map(str, row)) + '\n')
