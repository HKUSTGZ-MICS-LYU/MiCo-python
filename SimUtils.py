import os
import re
import sys
import json
import torch
import subprocess

from MiCoModel import MiCoModel

PWD = os.getcwd()

def gen_sim_bitfusion(model: MiCoModel, batch_size = 1, example_input = None):

    import sys
    import logging

    sys.path.append("hw/bitfusion")
    from MiCoGraphGen import MiCoGraphGen
    from dnnweaver2.graph import Graph

    if example_input is None:
        if model.default_dataset.startswith("CIFAR"):
            example_input = torch.randn(1, 3, 32, 32)
        elif model.default_dataset == "IMAGENET":
            example_input = torch.randn(1, 3, 224, 224)
    
    graph = Graph("Model", "Dataset", logging.INFO)
    with graph.as_default():
        m_graph = MiCoGraphGen(model, graph)
        m_graph.batch_size = batch_size
        m_graph(example_input)
        res = m_graph.sim()
    return res['Cycles']

def run_bitfusion_sim(json_name, config):
    with open(f'{PWD}/hw/bitfusion/configs/{json_name}', 'w') as f:
        json.dump(config, f)

    cmd = f'cd {PWD}/hw/bitfusion' + ' && ' + f'python sim.py configs/{json_name}'
    # Run command and get the output
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    proc.wait()
    output = proc.stdout.readlines()
    result = {}
    # Decode the output
    for line in output:
        line = str(line.decode())
        if line.startswith('Cycles: '):
            result['cycles'] = int(line.split(' ')[1])
        elif line.startswith('Energy: '):
            result['energy'] = float(line.split(' ')[1])
    assert 'cycles' in result, "Bitfusion simulation failed!"
    return result['cycles']

# NOTE: Deprecated function
def sim_bitfusion(model_name: str, wq: list, aq: list):

    # Check if bitfusion is installed
    if not os.path.exists(f'{PWD}/hw/bitfusion'):
        error = "Bitfusion not installed. Please install bitfusion first."
        raise FileNotFoundError(error)

    json_name = f'{model_name}.json'

    config = {
        "model_name": model_name,
        "wq": wq,
        "aq": aq
    }
    return run_bitfusion_sim(json_name, config)

def sim_mico(mico_type = "small"):

    mico_script = f"sim_{mico_type}_mico.sh"
    # Run the benchmark
    cmd = f'cd {PWD}/hw/VexiiMico' + ' && ' + \
        f'sh {mico_script} ../../project/main.elf'
    print("Running MiCo simulation...")
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    proc.wait()
    output = proc.stdout.readlines()
    if proc.stderr:
        print("Error in simulating the benchmark:")
        print(proc.stderr.readlines())
        return None
    for line in output:
        line_str = str(line.decode())
        if 'Execution Time:' in line_str:
            pat = r'Execution Time: (\d+)'
            match = re.search(pat, line_str)
            cycles = int(match.group(1))
            return cycles
    return None

def run_host(main="main", opt="opt"):

    # Run the benchmark
    cmd = f'{PWD}/project/{main}.elf'
    print("Running Host simulation...")
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    proc.wait()
    output = proc.stdout.readlines()
    if proc.stderr:
        print("Error in simulating the benchmark:")
        print(proc.stderr.readlines())
        return None
    for line in output:
        line_str = str(line.decode())
        if 'Execution Time:' in line_str:
            pat = r'Execution Time: (\d+)'
            match = re.search(pat, line_str)
            cycles = int(match.group(1))
            return cycles
    return None

def benchmark_bitfusion_matmul(N:int, M:int, K:int):

    # Check if bitfusion is installed
    if not os.path.exists(f'{PWD}/hw/bitfusion'):
        error = "Bitfusion not installed. Please install bitfusion first."
        raise FileNotFoundError(error)

    json_name = 'benchmark.json'

    config = {
        "model_name": "matmul",
        "n": N,
        "m": M,
        "k": K
    }
    qa = [2, 4, 8]
    qb = [2, 4, 8]
    res = []
    for i in range(len(qa)):
        for j in range(len(qb)):
            config['aq'] = [qa[i]]
            config['wq'] = [qb[j]]
            cycles = run_bitfusion_sim(json_name, config)
            print(f"QAxQB: {qa[i]}x{qb[j]}, Cycles: {cycles}")
            res.append((N, M, K, qa[i], qb[j], cycles))
    return res

def benchmark_bitfusion_conv2d(HW, C, K, KS, S):

    # Check if bitfusion is installed
    if not os.path.exists(f'{PWD}/hw/bitfusion'):
        error = "Bitfusion not installed. Please install bitfusion first."
        raise FileNotFoundError(error)

    json_name = 'benchmark.json'

    config = {
        "model_name": "conv2d",
        "h": HW,
        "w": HW,
        "c": C,
        "k": K,
        "ks": KS,
        "s": S
    }
    qa = [2, 4, 8]
    qb = [2, 4, 8]
    res = []
    for i in range(len(qa)):
        for j in range(len(qb)):
            config['aq'] = [qa[i]]
            config['wq'] = [qb[j]]
            cycles = run_bitfusion_sim(json_name, config)
            print(f"QAxQB: {qa[i]}x{qb[j]}, Cycles: {cycles}")
            res.append((HW, HW, C, K, KS, S, qa[i], qb[j], cycles))
    return res

def benchmark_mico_matmul(N: int, M: int, K: int, 
                   mico_script="sim_small_mico.sh",
                   mico_main="matmul_test"):
    
    # Check if mico is installed
    if not os.path.exists(f'{PWD}/hw/VexiiMico'):
        error = "VexiiMico not installed. Please install VexiiMico first."
        raise FileNotFoundError(error)
    
    res = []

    # Set Matmul Size
    with open(f"{PWD}/project/MiCo-Lib/test/{mico_main}.h", "w") as f:
        f.write(f"#define N {N}\n")
        f.write(f"#define M {M}\n")
        f.write(f"#define K {K}\n")

    # Compile the benchmark
    make_cmd = f'make recompile MAIN=tests/{mico_main} TARGET=vexii MARCH=rv32imfc OPT=simd'
    cmd = f'cd {PWD}/project' + ' && ' + make_cmd
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    proc.wait()
    if proc.stderr:
        print("Error in compiling the benchmark:")
        print(proc.stderr.readlines())
        return None
    
    # print("Benchmark compiled successfully!")
    
    # Run the benchmark
    cmd = f'cd {PWD}/hw/VexiiMico' + ' && ' + \
        f'sh {mico_script} ../../project/tests/{mico_main}.elf'

    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    proc.wait()
    output = proc.stdout.readlines()
    if proc.stderr:
        print("Error in simulating the benchmark:")
        print(proc.stderr.readlines())
        return None
    
    for line in output:
        line = str(line.decode())
        # Format: [info] MiCo QAxQB Time: xxxxxx
        if line.startswith('[info] MiCo'):
            qa = int(line[line.find('x')-1])
            qb = int(line[line.find('x')+1])
            cycles = int(line.split(': ')[1])
            res.append((N, M, K, qa, qb, cycles))
    return res

def benchmark_mico_conv2d(H, W, C, K, KS,
                   mico_script="sim_small_mico.sh",
                   mico_main="bitconv2d_test"):
    
    # Check if mico is installed
    if not os.path.exists(f'{PWD}/hw/VexiiMico'):
        error = "VexiiMico not installed. Please install VexiiMico first."
        raise FileNotFoundError(error)
    
    res = []

    # Set Matmul Size
    with open(f"{PWD}/project/MiCo-Lib/test/{mico_main}.h", "w") as f:
        f.write(f"#define N 1\n")
        f.write(f"#define INC {C}\n")
        f.write(f"#define INH {H}\n")
        f.write(f"#define INW {W}\n")
        f.write(f"#define K {KS}\n")
        f.write(f"#define M {K}\n")

    # Compile the benchmark
    make_cmd = f'make recompile MAIN=tests/{mico_main} TARGET=vexii MARCH=rv32imfc OPT=simd'
    cmd = f'cd {PWD}/project' + ' && ' + make_cmd
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    proc.wait()
    if proc.stderr:
        print("Error in compiling the benchmark:")
        print(proc.stderr.readlines())
        return None
    
    # print("Benchmark compiled successfully!")
    
    # Run the benchmark
    cmd = f'cd {PWD}/hw/VexiiMico' + ' && ' + \
        f'sh {mico_script} ../../project/tests/{mico_main}.elf'

    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    proc.wait()
    output = proc.stdout.readlines()
    if proc.stderr:
        print("Error in simulating the benchmark:")
        print(proc.stderr.readlines())
        return None
    
    for line in output:
        line = str(line.decode())
        # Format: [info] MiCo QAxQB Time: xxxxxx
        if line.startswith('[info] MiCo'):
            qa = int(line[line.find('x')-1])
            qb = int(line[line.find('x')+1])
            cycles = int(line.split(': ')[1])
            res.append((H, W, C, K, KS, qa, qb, cycles))
    return res

def benchmark_mico_pooling(C, H, W, K, S,
                   mico_script="sim_small_mico.sh",
                   mico_main="bitconv2d_test"):
    
    # Check if mico is installed
    if not os.path.exists(f'{PWD}/hw/VexiiMico'):
        error = "VexiiMico not installed. Please install VexiiMico first."
        raise FileNotFoundError(error)
    
    res = []

    # Set Matmul Size
    with open(f"{PWD}/project/MiCo-Lib/test/{mico_main}.h", "w") as f:
        f.write(f"#define N 1\n")
        f.write(f"#define INC {C}\n")
        f.write(f"#define INH {H}\n")
        f.write(f"#define INW {W}\n")
        f.write(f"#define K {K}\n")
        f.write(f"#define S {S}\n")

    # Compile the benchmark
    make_cmd = f'make recompile MAIN=tests/{mico_main} TARGET=vexii MARCH=rv32imfc OPT=simd'
    cmd = f'cd {PWD}/project' + ' && ' + make_cmd
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    proc.wait()
    if proc.stderr:
        print("Error in compiling the benchmark:")
        print(proc.stderr.readlines())
        return None
    
    # print("Benchmark compiled successfully!")
    
    # Run the benchmark
    cmd = f'cd {PWD}/hw/VexiiMico' + ' && ' + \
        f'sh {mico_script} ../../project/tests/{mico_main}.elf'

    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    proc.wait()
    output = proc.stdout.readlines()
    if proc.stderr:
        print("Error in simulating the benchmark:")
        print(proc.stderr.readlines())
        return None
    
    for line in output:
        line = str(line.decode())
        # Format: [info] MiCo QAxQB Time: xxxxxx
        if line.startswith('[info] MiCo Time'):
            cycles = int(line.split(': ')[1])
            res.append((C, H, W, K, S, cycles))
    return res


def benchmark_host_linear(N: int, M: int, K: int, 
                   main="bitlinear_test",
                   opt=""):
    
    res = []

    # Set Matmul Size
    with open(f"{PWD}/project/MiCo-Lib/test/{main}.h", "w") as f:
        f.write(f"#define N {N}\n")
        f.write(f"#define M {M}\n")
        f.write(f"#define K {K}\n")

    # Compile the benchmark
    make_cmd = f'make recompile MAIN=tests/{main} OPT=\"{opt}\"'
    cmd = f'cd {PWD}/project' + ' && ' + make_cmd
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    proc.wait()
    if proc.stderr:
        print("Error in compiling the benchmark:")
        print(proc.stderr.readlines())
        return None
    
    # print("Benchmark compiled successfully!")
    
    # Run the benchmark
    cmd = f'{PWD}/project/tests/{main}.elf'

    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    proc.wait()
    output = proc.stdout.readlines()
    if proc.stderr:
        print("Error in simulating the benchmark:")
        print(proc.stderr.readlines())
        return None
    
    for line in output:
        line = str(line.decode())
        # Format: MiCo QAxQB Time: xxxxxx
        if line.startswith('MiCo'):
            qa = int(line[line.find('x')-1])
            qb = int(line[line.find('x')+1])
            cycles = int(line.split(': ')[1])
            res.append((N, M, K, qa, qb, cycles))
    return res

def benchmark_host_conv2d(H, W, C, K, KS,
                   main="bitconv2d_test",
                   opt=""):
        
    res = []

    # Set Matmul Size
    with open(f"{PWD}/project/MiCo-Lib/test/{main}.h", "w") as f:
        f.write(f"#define N 1\n")
        f.write(f"#define INC {C}\n")
        f.write(f"#define INH {H}\n")
        f.write(f"#define INW {W}\n")
        f.write(f"#define K {KS}\n")
        f.write(f"#define M {K}\n")

    # Compile the benchmark
    make_cmd = f'make recompile MAIN=tests/{main} OPT=\"{opt}\"'
    cmd = f'cd {PWD}/project' + ' && ' + make_cmd
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    proc.wait()
    if proc.stderr:
        print("Error in compiling the benchmark:")
        print(proc.stderr.readlines())
        return None
    
    # print("Benchmark compiled successfully!")
    
    # Run the benchmark
    cmd = f'{PWD}/project/tests/{main}.elf'

    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    proc.wait()
    output = proc.stdout.readlines()
    if proc.stderr:
        print("Error in simulating the benchmark:")
        print(proc.stderr.readlines())
        return None
    
    for line in output:
        line = str(line.decode())
        # Format: MiCo QAxQB Time: xxxxxx
        if line.startswith('MiCo'):
            qa = int(line[line.find('x')-1])
            qb = int(line[line.find('x')+1])
            cycles = int(line.split(': ')[1])
            res.append((H, W, C, K, KS, qa, qb, cycles))
    return res

# Test
if __name__ == "__main__":
    res = benchmark_host_linear(128, 256, 512, opt="")
    print(res)
    res = benchmark_host_conv2d(28, 28, 64, 128, 3, opt="")
    print(res)