from SimUtils import benchmark_bitfusion_matmul, benchmark_bitfusion_conv2d
import argparse
import random
import math
import csv
from tqdm import tqdm
import os

def get_random_val(r, log_scale=False):
    if isinstance(r, list):
        return random.choice(r)
    elif isinstance(r, tuple):
        if log_scale and r[0] > 0:
            val = math.exp(random.uniform(math.log(r[0]), math.log(r[1])))
        else:
            val = random.uniform(r[0], r[1])
        return int(round(val))
    else:
        return r

def generate_matmul_samples(num_samples, ranges):
    samples = set()
    attempts = 0
    # Include corners for representativeness
    if isinstance(ranges['M'], tuple) and isinstance(ranges['K'], tuple):
        min_m, max_m = ranges['M']
        min_k, max_k = ranges['K']
        n_val = ranges['N'][0] if isinstance(ranges['N'], list) else ranges['N']
        samples.add((n_val, min_m, min_k))
        samples.add((n_val, max_m, max_k))
        samples.add((n_val, min_m, max_k))
        samples.add((n_val, max_m, min_k))

    with tqdm(total=num_samples, desc="Generating MatMul Samples") as pbar:
        pbar.update(len(samples))
        while len(samples) < num_samples and attempts < num_samples * 10:
            N = get_random_val(ranges['N'])
            M = get_random_val(ranges['M'], log_scale=True)
            K = get_random_val(ranges['K'], log_scale=True)
            if (N, M, K) not in samples:
                samples.add((N, M, K))
                pbar.update(1)
            attempts += 1
    return list(samples)

def generate_conv2d_samples(num_samples, ranges):
    samples = set()
    attempts = 0
    
    # Include corners
    if isinstance(ranges['HW'], tuple) and isinstance(ranges['C'], tuple) and isinstance(ranges['K'], tuple):
        min_hw, max_hw = ranges['HW']
        min_c, max_c = ranges['C']
        min_k, max_k = ranges['K']
        ks_val = ranges['KS'][0] if isinstance(ranges['KS'], list) else 3
        
        samples.add((min_hw, min_c, min_k, ks_val))
        samples.add((max_hw, max_c, max_k, ks_val))

    with tqdm(total=num_samples, desc="Generating Conv2D Samples") as pbar:
        pbar.update(len(samples))
        while len(samples) < num_samples and attempts < num_samples * 10:
            HW = get_random_val(ranges['HW'], log_scale=False)
            C = get_random_val(ranges['C'], log_scale=True)
            K = get_random_val(ranges['K'], log_scale=True)
            KS = get_random_val(ranges['KS']) # Usually small set [1, 3, 5, 7] or range
            
            # Kernel size shouldn't be larger than image
            if KS > HW:
                continue
                
            if (HW, C, K, KS) not in samples:
                samples.add((HW, C, K, KS))
                pbar.update(1)
            attempts += 1
    return list(samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matmul_samples', type=int, default=50, help='Number of MatMul samples')
    parser.add_argument('--conv2d_samples', type=int, default=50, help='Number of Conv2D samples')
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs('benchmark_results', exist_ok=True)

    # MatMul Configuration
    matmul_ranges = {
        'N': [16], 
        'M': (16, 4096),
        'K': (16, 4096)
    }

    print(f"Generating {args.matmul_samples} MatMul samples...")
    matmul_samples = generate_matmul_samples(args.matmul_samples, matmul_ranges)
    
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

    print(f"Generating {args.conv2d_samples} Conv2D samples...")
    conv2d_samples = generate_conv2d_samples(args.conv2d_samples, conv2d_ranges)
    
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
