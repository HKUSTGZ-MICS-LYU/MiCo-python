import os
import sys
import shutil
import subprocess

all_archs = [
    "rv32i",
    "rv32im",
    "rv32imc",
    "rv32imf",
    "rv32imfc",
    "rv64imfd",
    "rv64imfdc",
]

default_flags = [
    "TARGET=vexii_soc",
    "RAM_SIZE=8192K",
    "HEAP_SIZE=4096K",
]

main = "matmul_regression"

def make_arch(arch):
    full_arch = arch if "f" not in arch else arch.replace("f", "af")
    full_arch += "_zicsr"
    flags = default_flags + [f"MARCH={full_arch}"] + [f"MAIN={main}"] + [f"OPT=\"ref\""]
    cmd = ["make"] + flags + ["recompile"]
    print(f"Building architecture: {arch}")
    cmd = ' '.join(cmd)
    print(f"Command: {cmd}")
    result = subprocess.run([cmd], shell=True)
    if result.returncode != 0:
        print(f"Build failed for architecture: {arch}")
        sys.exit(1)
    print(f"Build succeeded for architecture: {arch}\n")
    os.mkdir("elfs/"+ arch) if not os.path.exists("elfs/"+ arch) else None
    shutil.move(f"{main}.elf", f"elfs/{arch}/{main}.elf")

if __name__ == "__main__":
    for arch in all_archs:
        make_arch(arch)