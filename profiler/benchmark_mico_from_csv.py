from SimUtils import benchmark_mico_matmul, benchmark_mico_conv2d

from datetime import datetime
from itertools import product
from time import perf_counter
from tqdm import tqdm

mico_target = "high"
mico_script = f"sim_{mico_target}_mico.sh"


def _timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _print_stage_start(stage, n_profiles=None):
    if n_profiles is None:
        print(f"[{_timestamp()}] Start {stage} benchmark")
    else:
        print(f"[{_timestamp()}] Start {stage} benchmark: {n_profiles} profiles")
    return perf_counter()


def _print_stage_end(stage, start_time, n_profiles, n_rows, profile_times):
    elapsed = perf_counter() - start_time
    print(f"[{_timestamp()}] End {stage} benchmark")
    print(f"[Timing] {stage}: total={elapsed:.3f}s, profiles={n_profiles}, rows={n_rows}")
    if n_profiles > 0:
        print(f"[Timing] {stage}: avg/profile={elapsed / n_profiles:.3f}s")
    if profile_times:
        print(
            f"[Timing] {stage}: per-call avg={sum(profile_times) / len(profile_times):.3f}s, "
            f"min={min(profile_times):.3f}s, max={max(profile_times):.3f}s"
        )
    return elapsed


if __name__ == '__main__':
    total_start = _print_stage_start("MiCo CSV")

    # MatMul Benchmarking

    profile_list = []
    with open("profile_features_matmul.csv", "r") as f:
        for line in f.readlines()[1:]:
            # N,M,K
            features = line.strip().split(",")
            profile_list.append([int(x) for x in features])

    matmul_start = _print_stage_start("MatMul", len(profile_list))
    sweep = tqdm(profile_list)
    dataset = []
    matmul_profile_times = []
    for N, M, K in sweep:
        sweep.set_description(f"N={N}, M={M}, K={K}")
        profile_start = perf_counter()
        res = benchmark_mico_matmul(N, M, K, mico_script, "bitlinear_test")
        profile_elapsed = perf_counter() - profile_start
        matmul_profile_times.append(profile_elapsed)
        sweep.set_postfix(time=f"{profile_elapsed:.2f}s")
        assert len(res) > 0, f"No results for N={N}, M={M}, K={K}"
        dataset += res
        sweep.update()
    matmul_elapsed = _print_stage_end(
        "MatMul", matmul_start, len(profile_list), len(dataset), matmul_profile_times)

    # with open(f'benchmark_results/mico_{mico_target}_matmul_zoo.csv', 'w') as f:
    #     f.write('N,M,K,QA,QW,Time\n')
    #     for row in dataset:
    #         f.write(','.join(map(str, row)) + '\n')


    # Conv2d Benchmarking
    profile_list = []
    with open("profile_features_conv2d.csv", "r") as f:
        for line in f.readlines()[1:]:
            # HW,C,K,KS,S
            features = line.strip().split(",")
            profile_list.append([int(x) for x in features])

    conv2d_start = _print_stage_start("Conv2d", len(profile_list))
    sweep = tqdm(profile_list)
    dataset = []
    conv2d_profile_times = []
    for HW, C, K, KS, S in sweep:
        sweep.set_description(f"H={HW}, W={HW}, C={C}, K={K}, KS={KS}, S={S}")
        profile_start = perf_counter()
        res = benchmark_mico_conv2d(HW, HW, C, K, KS, S, mico_script)
        profile_elapsed = perf_counter() - profile_start
        conv2d_profile_times.append(profile_elapsed)
        sweep.set_postfix(time=f"{profile_elapsed:.2f}s")
        dataset += res
        sweep.update()
    conv2d_elapsed = _print_stage_end(
        "Conv2d", conv2d_start, len(profile_list), len(dataset), conv2d_profile_times)

    total_elapsed = _print_stage_end(
        "MiCo CSV", total_start, len(matmul_profile_times) + len(conv2d_profile_times),
        0, matmul_profile_times + conv2d_profile_times)
    print(
        f"[Timing] Breakdown: matmul={matmul_elapsed:.3f}s, "
        f"conv2d={conv2d_elapsed:.3f}s, total={total_elapsed:.3f}s"
    )

    # with open(f'benchmark_results/mico_{mico_target}_conv2d_zoo.csv', 'w') as f:
    #     f.write('H,W,C,K,KS,S,QA,QW,Time\n')
    #     for row in dataset:
    #         f.write(','.join(map(str, row)) + '\n')
