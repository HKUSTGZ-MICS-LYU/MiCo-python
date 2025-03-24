import os
import sys
import json
import subprocess

PWD = os.getcwd()

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
    # TODO: We could add another arg to select the output key
    return result['cycles']

def sim_mico(wq:list, aq:list):
    pass


def benchmark_mico(N: int, M: int, K: int, mico_script="sim_small_mico.sh"):
    
    # Check if mico is installed
    if not os.path.exists(f'{PWD}/hw/VexiiMico'):
        error = "VexiiMico not installed. Please install VexiiMico first."
        raise FileNotFoundError(error)
    
    res = []

    # Set Matmul Size
    with open(f"{PWD}/project/MiCo-Lib/test/matmul_test.h", "w") as f:
        f.write(f"#define N {N}\n")
        f.write(f"#define M {M}\n")
        f.write(f"#define K {K}\n")

    # Compile the benchmark
    make_cmd = 'make recompile MAIN=matmul_test TARGET=vexii MARCH=rv32imc OPT=simd'
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
        f'sh {mico_script} ../../project/matmul_test.elf'

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

# Test
if __name__ == "__main__":
    res = benchmark_mico(32, 32, 32)
    print(res)