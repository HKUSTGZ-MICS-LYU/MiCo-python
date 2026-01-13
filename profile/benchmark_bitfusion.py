from SimUtils import benchmark_bitfusion_matmul, benchmark_bitfusion_conv2d, gen_sim_bitfusion

from itertools import product
from tqdm import tqdm

# Import shared samplers for adaptive profiling
from profile.sampler import MatMulSampler, Conv2DSampler

if __name__ == '__main__':

    # MatMul Benchmark
    # Ns = [16]
    # # Ms = [64, 128, 256, 512]
    # # Ks = [64, 128, 256, 512]

    # Ms = [1024, 2048, 4096]
    # Ks = [1024, 2048, 4096]

    # dataset = []
    # sweep = tqdm(total=len(Ns) * len(Ms) * len(Ks))
    # for N, M, K in product(Ns, Ms, Ks):
    #     sweep.set_description(f"N={N}, M={M}, K={K}")
    #     res = benchmark_bitfusion_matmul(N, M, K)
    #     dataset += res
    #     sweep.update()

    # with open('benchmark_results/bitfusion_matmul_large.csv', 'w') as f:
    #     f.write('N,M,K,QA,QW,Time\n')
    #     for row in dataset:
    #         f.write(','.join(map(str, row)) + '\n')

    # --- Example: Adaptive MatMul Benchmark using sampler ---
    # sampler = MatMulSampler(
    #     ranges={'N': [16], 'M': (64, 4096), 'K': (64, 4096)},
    #     strategy='adaptive'  # 'random', 'corner', 'prior', 'lhs', 'adaptive'
    # )
    # matmul_samples = sampler.generate(num_samples=100)
    # for N, M, K in tqdm(matmul_samples, desc="Adaptive MatMul"):
    #     res = benchmark_bitfusion_matmul(N, M, K)
    #     ...

    # Conv2D Benchmark

    HWs = [8, 32, 64]
    Cs = [1, 3, 16, 32, 256]
    Ks = [16, 32, 64, 256]
    KSs = [1, 3, 5]

    dataset = []
    sweep = tqdm(total=len(HWs) * len(Cs) * len(Ks) * len(KSs))
    for HW, C, K, KS in product(HWs, Cs, Ks, KSs):
        sweep.set_description(f"H={HW}, W={HW}, C={C}, K={K}, KS={KS}")
        res = benchmark_bitfusion_conv2d(HW, HW, C, K, KS)
        dataset += res
        sweep.update()

    with open('benchmark_results/bitfusion_conv2d.csv', 'w') as f:
        f.write('H,W,C,K,Ks,QA,QW,Time\n')
        for row in dataset:
            f.write(','.join(map(str, row)) + '\n')

    # --- Example: Adaptive Conv2D Benchmark using sampler ---
    # sampler = Conv2DSampler(
    #     ranges={'HW': (8, 64), 'C': (1, 256), 'K': (16, 256), 'KS': [1, 3, 5]},
    #     strategy='adaptive'
    # )
    # conv2d_samples = sampler.generate(num_samples=100)
    # for HW, C, K, KS in tqdm(conv2d_samples, desc="Adaptive Conv2D"):
    #     res = benchmark_bitfusion_conv2d(HW, HW, C, K, KS)
    #     ...