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