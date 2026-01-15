from SimUtils import benchmark_host_linear, benchmark_host_conv2d

from itertools import product
from tqdm import tqdm


if __name__ == '__main__':

    # MatMul Benchmarking

    profile_list = []
    with open("profile_features_matmul.csv", "r") as f:
        for line in f.readlines()[1:]:
            # N,M,K
            features = line.strip().split(",")
            profile_list.append([int(x) for x in features])

    # Overide N
    for i in range(len(profile_list)):
        profile_list[i][0] = 16

    sweep = tqdm(profile_list)
    dataset = []
    for N, M, K in sweep:
        sweep.set_description(f"N={N}, M={M}, K={K}")
        res = benchmark_host_linear(N, M, K)
        dataset += res
        sweep.update()

    with open('benchmark_results/host_matmul_zoo.csv', 'w') as f:
        f.write('N,M,K,QA,QW,Time\n')
        for row in dataset:
            f.write(','.join(map(str, row)) + '\n')


    # Conv2d Benchmarking
    profile_list = []
    with open("profile_features_conv2d.csv", "r") as f:
        for line in f.readlines()[1:]:
            # HW,C,K,KS,S
            features = line.strip().split(",")
            profile_list.append([int(x) for x in features])

    sweep = tqdm(profile_list)
    dataset = []
    for HW, C, K, KS, S in sweep:
        sweep.set_description(f"H={HW}, W={HW}, C={C}, K={K}, KS={KS}, S={S}")
        res = benchmark_host_conv2d(HW, C, K, KS, S)
        dataset += res
        sweep.update()

    with open('benchmark_results/host_conv2d_zoo.csv', 'w') as f:
        f.write('H,W,C,K,KS,S,QA,QW,Time\n')
        for row in dataset:
            f.write(','.join(map(str, row)) + '\n')