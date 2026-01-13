from SimUtils import benchmark_mico_matmul, benchmark_mico_conv2d, benchmark_mico_pooling

from itertools import product
from tqdm import tqdm

# Import shared samplers for adaptive profiling
from profile.sampler import MatMulSampler, Conv2DSampler, PoolingSampler

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

    # --- Example: Adaptive MatMul Benchmark using sampler ---
    # sampler = MatMulSampler(
    #     ranges={'N': [16, 32, 64], 'M': (32, 512), 'K': (32, 512)},
    #     strategy='adaptive'
    # )
    # matmul_samples = sampler.generate(num_samples=50)
    # for N, M, K in tqdm(matmul_samples, desc="Adaptive MatMul"):
    #     res = benchmark_mico_matmul(N, M, K, f"sim_{mico_config}_mico.sh", test_name)
    #     ...

    # Conv2D Benchmark

    HWs = [16, 32]
    Cs = [1, 3, 8, 16]
    Ks = [4, 8, 16]
    KSs = [3, 5]

    mico_config = "high"
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

    # --- Example: Adaptive Conv2D Benchmark using sampler ---
    # sampler = Conv2DSampler(
    #     ranges={'HW': (8, 64), 'C': (1, 64), 'K': (4, 64), 'KS': [3, 5]},
    #     strategy='adaptive'
    # )
    # conv2d_samples = sampler.generate(num_samples=50)
    # for HW, C, K, KS in tqdm(conv2d_samples, desc="Adaptive Conv2D"):
    #     res = benchmark_mico_conv2d(HW, HW, C, K, KS, ...)
    #     ...

    # Cs = [1, 3, 8, 16]
    # HWs = [8, 16, 32]
    # Ks = [2, 3, 4]
    # Ss = [1, 2]

    # mico_configs = ["high", "small", "cacheless"]
    # test_names = ["maxpool4d_f32_test", "avgpool4d_f32_test"]

    # for mico_config, test_name in product(mico_configs, test_names):
    #     print(f"Running benchmark for {mico_config} {test_name}")
    #     dataset = []
    #     dataset_file = f'benchmark_results/mico_{mico_config}_{test_name}.csv'

    #     sweep = tqdm(total=len(HWs) * len(Cs) * len(Ks) * len(Ss))
    #     with open(dataset_file, 'w') as f:
    #         f.write('C,H,W,K,S,Time\n')

    #     for C, HW, K, S in product(Cs, HWs, Ks, Ss):
    #         sweep.set_description(f"C={C}, H={HW}, W={HW}, K={K}, S={S}")
    #         res = benchmark_mico_pooling(C, HW, HW, K, S,
    #                                      f"sim_{mico_config}_mico.sh", test_name)
    #         dataset += res
    #         with open(dataset_file, 'a') as f:
    #             for row in res:
    #                 f.write(','.join(map(str, row)) + '\n')
    #         sweep.update()

    # --- Example: Adaptive Pooling Benchmark using sampler ---
    # sampler = PoolingSampler(
    #     ranges={'C': (1, 64), 'HW': (8, 64), 'K': [2, 3, 4], 'S': [1, 2]},
    #     strategy='adaptive'
    # )
    # pooling_samples = sampler.generate(num_samples=30)
    # for C, HW, K, S in tqdm(pooling_samples, desc="Adaptive Pooling"):
    #     res = benchmark_mico_pooling(C, HW, HW, K, S, ...)
    #     ...