from SimUtils import benchmark_mico_matmul, benchmark_mico_conv2d

from itertools import product
from tqdm import tqdm
if __name__ == '__main__':


    # MatMul Benchmark
    # Ns = [1, 32, 64, 128]
    # Ms = [32, 64, 128]
    # Ks = [32, 64, 128]

    # mico_config = "small"
    # test_name = "bitlinear_test"
    # dataset_file = f'benchmark_results/mico_{mico_config}_{test_name}.csv'

    # dataset = []
    # sweep = tqdm(total=len(Ns) * len(Ms) * len(Ks))
    # with open(dataset_file, 'w') as f:
    #     f.write('N,M,K,QA,QW,Time\n')
    # for N, M, K in product(Ns, Ms, Ks):
    #     sweep.set_description(f"N={N}, M={M}, K={K}")
    #     res = benchmark_mico_matmul(N, M, K, f"sim_{mico_config}_mico.sh", test_name)
    #     dataset += res
    #     with open(dataset_file, 'a') as f:
    #         for row in res:
    #             f.write(','.join(map(str, row)) + '\n')
    #     sweep.update()

    # Conv2D Benchmark

    HWs = [16, 32]
    Cs = [1, 3, 8, 16]
    Ks = [4, 8, 16]
    KSs = [3, 5]

    mico_config = "cacheless"
    test_name = "bitconv2d_test"

    dataset = []
    dataset_file = f'benchmark_results/mico_{mico_config}_{test_name}.csv'

    sweep = tqdm(total=len(HWs) * len(Cs) * len(Ks) * len(KSs))
    with open(dataset_file, 'w') as f:
        f.write('H,W,C,K,Ks,QA,QW,Time\n')

    for HW, C, K, KS in product(HWs, Cs, Ks, KSs):
        sweep.set_description(f"H={HW}, W={HW}, C={C}, K={K}, KS={KS}")
        res = benchmark_mico_conv2d(HW, HW, C, K, KS, 
                                    f"sim_{mico_config}_mico.sh", test_name)
        dataset += res
        with open(dataset_file, 'a') as f:
            for row in res:
                f.write(','.join(map(str, row)) + '\n')
        sweep.update()
