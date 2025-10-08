# Template MiCo Project

If you are trying to compile RISC-V binary, please note that the default Makefile are mostly tested on the pre-built GNU toolchain from sifive/freedom-tools with GCC 10.1 (https://github.com/sifive/freedom-tools/releases).

You can download it (Ubuntu version on X86) directly by:
```shell
wget https://static.dev.sifive.com/dev-tools/freedom-tools/v2020.08/riscv64-unknown-elf-gcc-10.1.0-2020.08.2-x86_64-linux-ubuntu14.tar.gz
```
And install it to your PATH.

**If you are using another RISC-V GNU toolchain, please tune the compile flags on your own GCC.** For example, if you are using the latest GNU toolchain built with multi-lib, you need to include `a` (e.g. `rv32imafc`) in `MARCH`.