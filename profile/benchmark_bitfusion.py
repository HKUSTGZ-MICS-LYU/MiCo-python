from SimUtils import benchmark_bitfusion_matmul, benchmark_bitfusion_conv2d

from itertools import product
from tqdm import tqdm
if __name__ == '__main__':

    # MatMul Benchmark
    # Ns = [16]
    # Ms = [64, 128, 256, 512]
    # Ks = [64, 128, 256, 512]

    # dataset = []
    # sweep = tqdm(total=len(Ns) * len(Ms) * len(Ks))
    # for N, M, K in product(Ns, Ms, Ks):
    #     sweep.set_description(f"N={N}, M={M}, K={K}")
    #     res = benchmark_bitfusion_matmul(N, M, K)
    #     dataset += res
    #     sweep.update()

    # with open('benchmark_results/bitfusion_matmul.csv', 'w') as f:
    #     f.write('N,M,K,QA,QW,Time\n')
    #     for row in dataset:
    #         f.write(','.join(map(str, row)) + '\n')

    # Conv2D Benchmark

    HWs = [8, 32, 64]
    Cs = [1, 3, 16, 32, 256]
    Ks = [16, 32, 64, 256]
    KSs = [3, 5]

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