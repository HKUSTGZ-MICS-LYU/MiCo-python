from SimUtils import benchmark_bitfusion_matmul, benchmark_bitfusion_conv2d, gen_sim_bitfusion

from itertools import product
from tqdm import tqdm

# Import shared samplers for adaptive profiling
from profiler.sampler import MatMulSampler, Conv2DSampler

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

    # Conv2D Benchmark

    HWs = [8, 32, 64]
    Cs = [1, 3, 16, 32, 256]
    Ks = [16, 32, 64, 256]
    KSs = [1, 3, 5]
    Ss = [1, 2] # strides

    suite = [
        [32, 32, 3, 64, 3, 1],
        [32, 32, 64, 64, 3, 1],
        [32, 32, 64, 128, 3, 2],
        [16, 16, 128, 128, 3, 1],
        [32, 32, 64, 128, 1, 2],
        [16, 16, 128, 256, 3, 2],
        [8, 8, 256, 256, 3, 1],
        [16, 16, 128, 256, 1, 2],
        [8, 8, 256, 512, 3, 2],
        [4, 4, 512, 512, 3, 1],
        [8, 8, 256, 512, 1, 2],
    ]


    dataset = []
    # sweep = tqdm(total=len(HWs) * len(Cs) * len(Ks) * len(KSs))
    # for HW, C, K, KS in product(HWs, Cs, Ks, KSs):
    #     sweep.set_description(f"H={HW}, W={HW}, C={C}, K={K}, KS={KS}")
    #     res = benchmark_bitfusion_conv2d(HW, C, K, KS)
    #     dataset += res
    #     sweep.update()

    sweep = tqdm(total=len(suite))
    for H, W, C, K, KS, S in suite:
        sweep.set_description(f"H={H}, W={W}, C={C}, K={K}, KS={KS} S={S}")
        res = benchmark_bitfusion_conv2d(H, C, K, KS, S)
        dataset += res
        sweep.update()

    with open('benchmark_results/bitfusion_conv2d_resnet8.csv', 'w') as f:
        f.write('H,W,C,K,Ks,S,QA,QW,Time\n')
        for row in dataset:
            f.write(','.join(map(str, row)) + '\n')