from SimUtils import benchmark_host_linear, benchmark_host_conv2d

from itertools import product
from tqdm import tqdm
if __name__ == '__main__':


    # BitLinear Benchmark
    # Ns = [1, 16, 32]
    # Ms = [32, 64, 1024, 2048, 4096]
    # Ks = [32, 64, 1024, 2048, 4096]

    # test_name = "bitlinear_test"
    # opt = "lut"
    # dataset_file = f'benchmark_results/host_{opt}_{test_name}.csv'

    # dataset = []
    # sweep = tqdm(total=len(Ns) * len(Ms) * len(Ks))
    # with open(dataset_file, 'w') as f:
    #     f.write('N,M,K,QA,QW,Time\n')
    # for N, M, K in product(Ns, Ms, Ks):
    #     sweep.set_description(f"N={N}, M={M}, K={K}")
    #     res = benchmark_host_linear(N, M, K, test_name, opt)
    #     dataset += res
    #     with open(dataset_file, 'a') as f:
    #         for row in res:
    #             f.write(','.join(map(str, row)) + '\n')
    #     sweep.update()

    
    # BitConv2D Benchmark
    HWs = [16, 32, 64, 128, 256]
    Cs = [1, 3, 8, 16, 32, 64, 128, 256]
    Ks = [4, 8, 16, 32, 64, 128, 256]
    KSs = [1, 3, 5]

    test_name = "bitconv2d_test"
    opt = "lut"

    dataset = []
    dataset_file = f'benchmark_results/host_{opt}_{test_name}.csv'

    sweep = tqdm(total=len(HWs) * len(Cs) * len(Ks) * len(KSs))
    with open(dataset_file, 'w') as f:
        f.write('H,W,C,K,Ks,QA,QW,Time\n')

    for HW, C, K, KS in product(HWs, Cs, Ks, KSs):
        sweep.set_description(f"H={HW}, W={HW}, C={C}, K={K}, KS={KS}")
        res = benchmark_host_conv2d(HW, HW, C, K, KS, test_name, opt)
        dataset += res
        with open(dataset_file, 'a') as f:
            for row in res:
                f.write(','.join(map(str, row)) + '\n')
        sweep.update()