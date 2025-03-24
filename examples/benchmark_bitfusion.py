from SimUtils import benchmark_bitfusion

from itertools import product
from tqdm import tqdm
if __name__ == '__main__':

    Ns = [128, 256, 512]
    Ms = [128, 256, 512]
    Ks = [128, 256, 512]

    dataset = []
    sweep = tqdm(total=len(Ns) * len(Ms) * len(Ks))
    for N, M, K in product(Ns, Ms, Ks):
        sweep.set_description(f"N={N}, M={M}, K={K}")
        res = benchmark_bitfusion(N, M, K)
        dataset += res
        sweep.update()

    with open('benchmark_results/bitfusion.csv', 'w') as f:
        f.write('N,M,K,QA,QW,Time\n')
        for row in dataset:
            f.write(','.join(map(str, row)) + '\n')
