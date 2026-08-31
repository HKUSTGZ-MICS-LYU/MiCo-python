---
name: mico-lib-operators
description: Develop, review, test, optimize, and integrate C operators in MiCo-Lib under project/MiCo-Lib, including quantized and mixed-precision kernels, tensor layouts, target-specific implementations, generated C calls, and project/Makefile builds. Use when a request mentions MiCo-Lib, project/Makefile, MAIN/TARGET/OPT, qtypes, packed weights, RVV/Vexii/Gemmini/Chipyard, or adding a deployable MiCo operator.
---

# MiCo-Lib Operators and `project/` Makefile

Use this skill for the C inference side of MiCo. The existing
`mico-mpq-workflow` skill covers Python models, quantization search, and the
high-level deployment workflow. This skill covers the ABI below
`project/MiCo-Lib`, implementation and optimization variants, tests, generated
C calls, and the build commands that select them.

## First Pass

Before editing or running a target build:

1. Read `project/Makefile` and the target file under
   `project/MiCo-Lib/targets/<TARGET>.mk`.
2. Read the public header that owns the interface:
   `project/MiCo-Lib/include/nn.h`, `mico_nn.h`, `mico_qnn.h`, or
   `mico_quant.h`.
3. Read the nearest baseline implementation under `project/MiCo-Lib/src/` and
   the nearest test under `project/tests/`.
4. If the operator is emitted by Python CodeGen, read the matching handler in
   `MiCoRegistry.py`, `MiCoCodeGen.py`, and `doc/REGISTRY_USAGE.md`.
5. Run `git status --short` and inspect changes in every file that the task
   touches. Preserve user changes and do not reset or clean unrelated files.
6. Decide whether the task is a baseline operator, quantization wrapper, low-bit
   kernel, target override, CodeGen integration, or a combination. Keep each
   layer of the change independently testable.

Treat the current source and headers as the ABI source of truth. The prose in
`project/MiCo-Lib/doc/` is useful context, but it may lag behind the Makefile
and target-specific code.

## Repository Map

- `project/MiCo-Lib/include/nn.h`: FP32 tensor structs, FP32 operator API,
  layout macros, allocation, assertions, and shared helpers.
- `project/MiCo-Lib/include/mico_nn.h`: FP32-to-quantized operator API and
  `Tensor*_Q8` declarations.
- `project/MiCo-Lib/include/mico_qnn.h`: packed integer MatMul ABI and the
  low-bit extraction macros.
- `project/MiCo-Lib/include/mico_quant.h`: FP32 activation quantization API,
  packing helpers, and quantization primitives.
- `project/MiCo-Lib/include/mico_runtime.h`: runtime MatMul function-pointer
  table and `qlog` selection.
- `project/MiCo-Lib/src/*.c`: baseline FP32 operators and shared utilities.
- `project/MiCo-Lib/src/mico/*.c`: reference mixed-precision wrappers,
  quantization, packed MatMul, and runtime dispatch.
- `project/MiCo-Lib/src/mico_unrolled/`: software unrolled variants.
- `project/MiCo-Lib/src/optimized/`: generic RISC-V software-optimized
  MatMul variants.
- `project/MiCo-Lib/src/mico_lut/`: LUT-based MatMul variants.
- `project/MiCo-Lib/src/im2col_conv2d/`: alternate FP32 Conv2D path selected by
  `OPT=im2col`.
- `project/MiCo-Lib/targets/`: platform makefiles, runtimes, assembly, and
  target-specific overrides.
- `project/tests/`: executable C tests. The top-level Makefile builds one test
  by setting `MAIN=tests/<name>`; there is no separate test Makefile.
- `project/MiCo-Lib/test/`: test headers, fixtures, and generated input data
  included by files under `project/tests/`.

## Operator Decision Tree

### 1. Define the contract first

Write down the equivalent PyTorch or mathematical operation before coding:

- input and output rank, logical shapes, and contiguous storage order;
- default and alternate layouts;
- parameter order and semantics, including stride, padding, dilation, groups,
  axes, and whether negative axes are accepted;
- dtype and scale behavior;
- whether `y` may alias an input or only share its data pointer as a view;
- behavior for empty bias, zero-size dimensions, tails, padding, and invalid
  arguments;
- target-specific requirements such as alignment, ISA, scratchpad, or memory.

Do not silently accept an argument that the implementation does not honor. In
particular, the current Conv1D and Conv2D baselines accept a `dilation` argument
but do not implement dilation in their indexing or output-size formula. A new
operator must either implement dilation correctly or assert/document that only
`dilation == 1` is supported, and the Python handler must have the same
restriction.

### 2. Choose the implementation layer

#### New FP32 operator

For an operator that consumes and produces FP32 tensors:

1. Add the public declaration to `include/nn.h`.
2. Add the baseline implementation to `src/<operation>.c`.
3. Use the existing `Tensor1D_F32` through `Tensor4D_F32` structs.
4. Add a focused executable test under `project/tests/`.

The top-level Makefile automatically includes `$(MICO_DIR)/src/*.c`, so a new
source file normally does not require a Makefile edit.

Existing examples include `src/add.c`, `src/pooling.c`, `src/relu.c`,
`src/mean.c`, `src/norm.c`, and `src/vit_ops.c`.

#### New mixed-precision wrapper

For FP32 input, packed quantized weights, and FP32 output:

1. Add the public declaration to `include/mico_nn.h`.
2. Implement the baseline under `src/mico/`.
3. Use `MiCo_*_quant` or `MiCo_2D_quant`/`MiCo_4D_quant` for activation
   quantization instead of duplicating packing logic.
4. Use `MiCo_runtime.matmul_matrix[...]` for packed MatMul when the operation
   reduces to MatMul.
5. Apply the output scale as `weight->scale * activation_scale` and add bias
   with the same semantics as the existing bitlinear and bitconv wrappers.
6. Check aligned dimensions, quantization-buffer capacity, output shapes, and
   groups before entering the kernel.

Read `src/mico/bitlinear.c`, `bitconv1d.c`, and `bitconv2d.c` before designing a
new wrapper. They show the expected quantize, accumulate, dequantize, and
profiling flow.

#### New packed MatMul kernel

For a new combination or implementation of integer packed MatMul:

1. Declare the public function in `include/mico_qnn.h`.
2. Add the default implementation in `src/mico/qmatmul.c`.
3. If reference comparison is needed, add a corresponding `_Ref` function to
   `src/mico/qmatmul_ref.c`; it is exposed only when `REF` is defined.
4. Add the same ABI to every required implementation directory, such as
   `src/optimized/`, `src/mico_unrolled/`, `src/mico_lut/`, or a target
   directory.
5. Update `mico_runtime.c` if the kernel is selected through the runtime
   function-pointer matrix. Do not add a qtype without checking the matrix
   dimensions, `MAX_QTYPE_LOG2`, `qlog`, all valid bit-width combinations, and
   every target that indexes the table.
6. Compare the new implementation against a scalar reference for every output
   element, including non-aligned reduction lengths.

The existing table supports bit widths 1, 2, 4, and 8. The names encode the
activation/weight combination in the existing runtime conventions, so preserve
that convention and verify which argument is the activation for each caller.

#### Target-specific or optimized override

Use a target-specific implementation when the algorithm depends on an ISA,
vector width, assembly ABI, or accelerator. Keep the public symbol and all
semantics identical to the baseline:

- exact C signature and argument order;
- same logical layout and packed-byte order;
- same output initialization, accumulation, scale, and tail behavior;
- same supported parameter range and failure behavior.

Many baseline functions are marked `__attribute__((weak))`; optimized or target
implementations provide a strong definition with the same name. This is an
override mechanism, not permission to create a second ABI. Check the final
symbol table when adding an override, and never combine source directories that
provide conflicting strong definitions without verifying the link result.

### 3. Keep Python CodeGen integration separate

If the new C function must be called by generated models, the C implementation
alone is insufficient:

1. Add the declaration to the header included by generated model headers.
2. Register a PyTorch function or module in `MiCoRegistry.py` using
   `@MiCoOpRegistry.register_function(...)` or
   `@MiCoOpRegistry.register_module(...)`.
3. In the handler, register output storage with
   `add_uninitialized_tensor`, `add_initialized_tensor`, or
   `add_connect_tensor` as appropriate.
4. Emit a call with `add_forward_call`. The generated argument order is output
   first, then tensor inputs, then scalar parameters; it passes each tensor as
   `&model-><name>`.
5. Use the existing name-template convention when useful, for example
   `MiCo_myop{dim}d_{dtype}`. The formatter turns `F32` into `f32`.
6. Trace a tiny model, inspect the generated call, compile it, and run it
   before optimizing the C kernel.

Prefer the registry extension point over adding a one-off `elif` to
`MiCoCodeGen.py`. If a handler needs a parameter that FX cannot resolve, fail
with an explicit error rather than emitting a guessed C call.

A function handler has the form:

```python
@MiCoOpRegistry.register_function(torch.my_op)
def handle_my_op(codegen, n, out, input_names, input_args):
    codegen.add_uninitialized_tensor(n.name, out)
    codegen.add_forward_call(
        "MiCo_my_op{dim}d_{dtype}", out, n.name, input_names, [some_param]
    )
```

A module handler has the form:

```python
@MiCoOpRegistry.register_module(MyModule)
def handle_my_module(codegen, n, out, module, input_names):
    codegen.add_uninitialized_tensor(n.name, out)
    codegen.add_forward_call(
        "MiCo_my_module_{dtype}", out, n.name, input_names, [module.param]
    )
```

Use `doc/REGISTRY_USAGE.md` for the full handler signatures and helper
semantics.

## MiCo-Lib ABI and Data Layout

### Tensor containers

`nn.h` defines `Tensor{0,1,2,3,4}D_F32` with a `float *data` pointer. Ranked
quantized containers in `mico_nn.h` have:

```c
qbyte *data;
float scale;
qtype wq;
```

`qbyte` is `int8_t` and `qtype` is `uint8_t`. `Tensor*_Q8` is a generic
container name; it can hold packed Q8, Q4, Q2, or Q1 values. Its `shape` records
logical dimensions, while the byte storage may be packed and/or padded for an
aligned logical dimension.

The containers have no stride metadata. Treat them as contiguous and make all
shape and layout assumptions explicit in the function contract.

### Layouts and weights

The default build is NCHW for 4D activations and NCL for 1D activations. Define
`USE_ALT_LAYOUT` to select the alternate paths that actually support it:

| Operation | Default layout | `USE_ALT_LAYOUT` |
|---|---|---|
| 4D activation | `[N, C, H, W]` | `[N, H, W, C]` |
| 2D conv weight | `[O, I, KH, KW]` (OIHW) | `[KH, KW, I, O]` (HWIO) |
| 2D MatMul weight | `[M, K]` | `[K, M]` |
| 1D activation/weight | activation `[N, C, L]`, weight `[O, I, K]` | no general alternate path |

`OFFSET_4D` in `nn.h` changes with `USE_ALT_LAYOUT`, but not every existing
operator has a complete alternate implementation. Do not assume that adding
the macro is enough. Test alternate layout with data and weight packing built
for that layout.

For generated Gemmini models, `MiCoCodeGen` can explicitly transpose or
permute weights and sets the appropriate layout macro through the target. Keep
that transformation aligned with the C ABI; do not transpose twice.

### Low-bit packing

Use the helpers in `mico_qnn.h` and the quantizers in `mico_quant.h`. The
existing byte layout is:

| Quantization | Values per byte | Value order |
|---|---:|---|
| Q8 | 1 | one signed byte per value |
| Q4 | 2 | low nibble is the first value, high nibble the second |
| Q2 | 4 | 2-bit fields at shifts 0, 2, 4, 6 |
| Q1 | 8 | bits 0 through 7 in logical order |

The Q2 code is not ordinary signed two's-complement interpretation. The
existing `TWO_BIT_TO_INT8` mapping is `0 -> 1`, `1 -> 0`, `2 -> -2`, `3 -> -1`.
Q1 uses `BIT_TO_INT8`, where bit 0 maps to `+1` and bit 1 maps to `-1`. Q4
values require sign extension with `SIGN_EXTEND_TO_INT8`.

Use `EXTRACT_BIT`, `EXTRACT_2BIT`, and `EXTRACT_4BIT` rather than duplicating
field arithmetic. Test the first, last, and cross-byte values. A logical length
that is not divisible by 2, 4, or 8 must have explicit zero padding in the
packed storage and must not be read as an additional real value.

### Quantization, bias, and alignment

The existing activation quantizers are batch-wise in several paths, not
per-row or per-channel. The usual scale behavior is:

- Q8: `127 / absmax` quantization and reciprocal scale;
- Q4: `7 / absmax` quantization and reciprocal scale;
- Q2: `1 / absmax` quantization and reciprocal scale;
- Q1: absolute-mean scale with sign bits.

Check the zero-input case before dividing by `absmax`; the current primitives
need special care when `absmax == 0`.

For wrappers using aligned dimensions:

- `align` is a logical feature/reduction alignment used to pad the quantized
  input or im2col K dimension;
- `MICO_ALIGN` is a byte/address alignment macro used by allocation, embedded
  sections, and included model weights;
- zero-fill every logical padding element before accumulation;
- `MiCo_QBuffer` is global and sized by `QUANTIZE_BUFFER_SIZE` (32 KiB by
  default, 1 MiB for the `host` target); it is not a per-thread workspace;
- follow the current capacity check and investigate the exact-boundary case
  rather than assuming the buffer is unlimited;
- use `MiCo_alloc`/`MiCo_free` for aligned dynamic memory when appropriate, and
  check allocation failures before dereferencing.

A bias with `bias->shape[0] == 0` means no bias in the existing wrappers. A
new operator must preserve that convention or define a different one clearly.

### Convolution and indexing pitfalls

For a 2D convolution, validate `in_c % groups == 0` and
`out_c % groups == 0`, calculate the output shape from the actual supported
parameters, and keep the weight offset consistent with groups and layout. For
padding, compute input coordinates in a signed type before checking bounds.
Do not write code such as `size_t ih = ...; if (ih >= 0)`: the lower-bound check
is ineffective after unsigned underflow.

When an operation writes a view or aliases storage, document the ownership and
lifetime. Existing flatten/view helpers intentionally share `data` pointers;
most numerical operators expect separate output storage.

## Implementing a New Operator

Follow this sequence unless the task explicitly requires a different scope:

1. **Find the nearest sibling.** Read a same-rank operator and its test. Reuse
   local indexing, assertion, profiler, and allocation patterns.
2. **Specify shapes and layouts.** Add comments only for non-obvious indexing;
   validate all parameters needed to avoid out-of-bounds access.
3. **Write the baseline.** Keep the first implementation scalar and easy to
   compare. Do not start with target intrinsics or an unverified packed fast
   path.
4. **Expose one public declaration.** Put it in the owning header and use the
   project naming convention, normally `MiCo_<operation>_<rank>d_f32` for FP32
   APIs.
5. **Add a deterministic test.** Use tiny static or heap-allocated inputs,
   hand-computed/reference outputs, and test tails and boundary conditions.
6. **Build the exact test with `MAIN`.** Use an isolated `BUILD` directory and
   the target/`OPT` combination under test.
7. **Add CodeGen registration if required.** Trace a tiny PyTorch module and
   compile the generated C artifact.
8. **Add optimized or target variants.** Preserve the baseline ABI and compare
   against the reference before measuring performance.
9. **Inspect the diff and symbols.** Confirm no accidental source-selection,
   layout, or generated-artifact changes.

### Reference test design

A test for a normal FP32 operator should cover:

- shape and output initialization;
- a nontrivial deterministic input, including negative values where relevant;
- stride/padding/groups/axis behavior;
- odd or non-aligned dimensions;
- boundary positions and invalid-parameter assertions where practical;
- default layout and `USE_ALT_LAYOUT` when supported.

A quantized wrapper test should additionally cover:

- packed Q8/Q4/Q2/Q1 storage and the exact tail;
- weight scale and activation scale;
- bias present and `bias->shape[0] == 0`;
- comparison with an integer accumulator plus explicit dequantization;
- quantization-buffer size and aligned dimensions.

A low-level kernel test should compile with `OPT=ref`, call the `_Ref` symbol,
and compare it element by element with the selected implementation. The RVV
operator tests are good examples of using a deliberately non-aligned input
length such as 17.

## `project/Makefile` Reference

Run from the repository root with `make -C project ...`, or `cd project` and
run `make ...`. `MAIN` is a path without the `.c` suffix.

### Core variables

| Variable | Current default | Meaning |
|---|---|---|
| `MICO_DIR` | `MiCo-Lib` | MiCo-Lib root relative to `project/` |
| `CC` | `gcc` | overwritten by some target files |
| `OBJDUMP` | `objdump` | disassembler, overwritten by some targets |
| `BUILD` | `build` | object and map output directory |
| `MAIN` | `main` | source stem; `MAIN=tests/foo` uses `tests/foo.c` |
| `TARGET` | `host` | selects `MiCo-Lib/targets/$(TARGET).mk` |
| `OPT` | empty | space-separated source/compile feature tokens |
| `TEST_NUM` | `1` | compile-time test selector used by target code |
| `MARCH`, `MABI` | target-specific | RISC-V compiler ISA and ABI |
| `CFLAGS` | target plus `-O3` | extra compile flags are appended |
| `EXTRA_CFLAGS` | empty | additional flags appended to `CFLAGS` |
| `C_DEFINES` | empty | passed through `targets/common.mk` |
| `LDFLAGS` | `-lm` plus target flags | linker flags |

The source list is assembled automatically:

- `src/*.c` and `src/mico/*.c` are always included;
- `OPT` adds variant source directories;
- the target makefile adds platform C/assembly sources;
- `MAIN.c` is linked directly into `MAIN.elf`.

Use `make -n` first when debugging source selection or flags. Use `make
-VARIABLES`-style assumptions only after reading the target file; this Makefile
has no universal `help` target.

### Core targets

```bash
# Preview the exact commands.
make -C project -n MAIN=tests/my_op_test TARGET=host BUILD=build/my-op compile

# Build MAIN.elf and MAIN.asm.
make -C project MAIN=tests/my_op_test TARGET=host BUILD=build/my-op compile

# Build and run a host executable.
make -C project MAIN=tests/my_op_test TARGET=host BUILD=build/my-op run-host

# Build and run in one invocation.
make -C project MAIN=tests/my_op_test TARGET=host BUILD=build/my-op compile run-host

# Clean the selected BUILD directory plus the Makefile's ELF/assembly outputs,
# then rebuild. The clean recipe also removes every project/tests/*.elf and
# project/tests/*.asm, so inspect existing test artifacts first.
make -C project MAIN=tests/my_op_test TARGET=host BUILD=build/my-op recompile

# Explicit cleanup. The recipe removes $(BUILD), root *.elf/*.asm, and all
# tests/*.elf/*.asm under project/.
make -C project BUILD=build/my-op clean
```

`compile` depends on `MAIN.asm`, which depends on `MAIN.elf`. `recompile` runs
`clean` followed by `compile`; `all` is an alias for `compile`; `debug` adds
`-DDEBUG` and recompiles. `run-host` runs `./$(MAIN).elf`, so it works with
`MAIN=tests/foo` when invoked from `project/` or through `make -C project`.

Use a distinct `BUILD` value for each target, layout, and optimization
combination. Do not clean a shared build directory while another experiment is
using it. `clean` does not remove arbitrary output directories outside its
selected `$(BUILD)` path.

### `OPT` tokens

`OPT` is parsed as whitespace-separated tokens, so quote combinations:

| Token | Effect |
|---|---|
| `unroll` | adds `src/mico_unrolled/*.c` |
| `gcc-unroll` | adds `-funroll-all-loops` |
| `im2col` | adds `src/im2col_conv2d/*.c` |
| `opt` | adds `src/optimized/*.c` |
| `lut` | adds `src/mico_lut/*.c` |
| `ref` | adds `-DREF`, exposing `_Ref` functions |
| `alt-layout` | adds `-DUSE_ALT_LAYOUT` |

Target files add further tokens:

- `vexii`: `simd` selects `targets/vexii/mico32` or `mico64`; `legacy` only
  changes that SIMD path to its `legacy` subdirectory.
- `vexii_soc`: `cfu`, `simd`, `bnrv`, and `bncfu` select target implementations;
  `legacy` modifies the `cfu`/`simd` path.
- Other target-specific flags are read from their `.mk` file, not inferred
  from the token name.

Useful combinations look like:

```bash
OPT='ref'                 # expose scalar references
OPT='ref alt-layout'      # alternate layout plus reference symbols
OPT='opt ref'             # optimized source plus reference symbols
OPT='simd'                # target-specific SIMD, e.g. TARGET=vexii
```

Do not combine `TARGET=rvv` with `OPT=opt`, `OPT=unroll`, or `OPT=lut`: those
modes can add another strong `MiCo_Q8_MatMul` definition. Also do not assume
`ref` replaces the selected kernel; it only compiles the `_Ref` functions, and
tests must call them explicitly.

`alt-layout` changes data interpretation, not just indexing. Build the test
with weights and input buffers initialized in the alternate format, and use a
separate `BUILD` directory from the default-layout build.

### Target matrix

Read the target makefile before choosing one. The current target names are:

| `TARGET` | Use | Run/build notes |
|---|---|---|
| `host` | portable host GCC baseline | `run-host`; uses a 1 MiB quantization buffer |
| `x86` | host plus AVX/AVX2 sources | needs an AVX2-capable compiler/CPU; `run-host` |
| `openmp` | experimental OpenMP source variant | current `.mk` does not add `-fopenmp` or `-lgomp`; do not assume OpenMP is enabled |
| `cuda` | CUDA `.cu` target source | requires `nvcc`; normal C is also compiled through the target's compiler rule |
| `rvv` | Chipyard RVV backend | `run-rvv`; requires `riscv64-unknown-elf-gcc`, Spike, matching ISA, and Chipyard setup |
| `spike` | Chipyard bare-metal Spike profile | `run-spike`; requires the Chipyard RISC-V toolchain and Spike |
| `rocket` | Rocket/Chipyard ELF | compile/disassemble; external Chipyard flow runs it |
| `boom` | Boom/Chipyard ELF | compile/disassemble; external Chipyard flow runs it |
| `gemmini` | Gemmini/Chipyard build | requires `CHIPYARD_DIR`; enables alternate layout and Gemmini headers |
| `vexii` | VexiiRiscv target | compile ELF; use target simulator/loading flow externally; `OPT=simd` selects SIMD sources |
| `vexii_bitnet` | Vexii BitNet target | compile ELF; configure `BITNET_QUANT` and `USE_SIMD`; external run flow |
| `vexii_soc` | Vexii MicroSoC target | compile ELF; configure linker memory and optional `cfu`/`simd`/`bnrv`/`bncfu` |
| `coralnpu` | CoralNPU RVV target | `coralnpu-check`, `coralnpu-sim`, and `run-coralnpu`; requires CoralNPU Bazel workspace/toolchain |

Examples:

```bash
# Host baseline.
make -C project TARGET=host MAIN=tests/my_op_test BUILD=build/my-op-host compile run-host

# x86 AVX2 variant.
make -C project TARGET=x86 MAIN=tests/my_op_test BUILD=build/my-op-x86 compile run-host

# RVV reference build and Spike run.
make -C project TARGET=rvv MAIN=tests/rvv_q8_test \
  BUILD=build/rvv-q8 OPT=ref compile run-rvv

# Alternate-layout RVV build; use a separate BUILD path.
make -C project TARGET=rvv MAIN=tests/rvv_q8_test \
  BUILD=build/rvv-q8-alt OPT='ref alt-layout' compile run-rvv

# Vexii SIMD ELF.
make -C project TARGET=vexii MAIN=main BUILD=build/vexii OPT=simd compile

# Vexii SoC CFU ELF with explicit memory settings.
make -C project TARGET=vexii_soc MAIN=main BUILD=build/vexii-soc \
  RAM_SIZE=8192K HEAP_SIZE=4096K OPT=cfu compile

# Chipyard Gemmini build.
make -C project TARGET=gemmini CHIPYARD_DIR=/path/to/chipyard \
  MAIN=main BUILD=build/gemmini compile

# CoralNPU checks and simulation.
make -C project TARGET=coralnpu MAIN=main BUILD=build/coralnpu coralnpu-check
make -C project TARGET=coralnpu MAIN=main BUILD=build/coralnpu run-coralnpu
```

For RVV, `MARCH`, `MABI`, `SPIKE_ISA`, and `HTIF_HEAP_SIZE` must agree. The
current RVV README uses `rv64gcv_zicntr_zihpm`/`lp64d` and reserves a 1 MiB
HTIF heap by default. Inspect `project/MiCo-Lib/targets/rvv/README.md` for
vector-specific toolchain and disassembly checks.

For `vexii`, `OPT=legacy` is not an independent optimization; it only changes
the SIMD source directory when the relevant SIMD path is selected. For
`vexii_soc`, the current variable spelling in the target file is `REPAET`, so
verify the file before trying to enable its `REPEAT` macro.

## Validation Ladder

Run the smallest check that proves the change, then expand according to risk:

1. Syntax and header check with the intended target and `make -n`.
2. Host compile with warnings:

   ```bash
   make -C project TARGET=host MAIN=tests/my_op_test BUILD=build/my-op \
     EXTRA_CFLAGS='-Wall -Wextra' OPT='ref' compile
   ```

3. Execute the focused host test:

   ```bash
   make -C project TARGET=host MAIN=tests/my_op_test BUILD=build/my-op \
     EXTRA_CFLAGS='-Wall -Wextra' OPT='ref' run-host
   ```

4. For low-bit or optimized kernels, compare the selected symbol with `_Ref`
   and run both default and alternate layout tests when supported.
5. For a target override, inspect symbols and disassembly:

   ```bash
   nm -C project/tests/my_op_test.elf | grep 'MiCo_'
   objdump -d project/tests/my_op_test.elf | grep -E 'vsetvli|vle8|vwmacc'
   ```

6. For CodeGen, trace a tiny model, inspect the generated `.h` and `.bin`,
   compile the generated model with the target Makefile, and run it.
7. Report target prerequisites and skipped hardware/simulator checks explicitly.

For a new packed MatMul, the usual host reference command is:

```bash
make -C project TARGET=host MAIN=tests/matmul_regression \
  BUILD=build/matmul-regression OPT=ref compile run-host
```

The regression test explicitly calls both the reference and selected low-bit
functions. Do not interpret a successful compile alone as numerical
validation.

## Troubleshooting

- **`No rule to make target ... .c`**: `MAIN` is a source stem. Use
  `MAIN=tests/foo`, not `MAIN=tests/foo.c`, and run from the repository root or
  `project/` consistently.
- **Undefined reference**: confirm the declaration, source directory, target,
  and `OPT` selection. A file under `src/optimized`, `src/mico_lut`, or a target
  directory is not included unless its make logic selects it.
- **Duplicate symbol**: inspect all selected sources for strong definitions of
  the same function. Separate `BUILD` paths do not fix a single link command
  containing conflicting implementations.
- **Wrong output with `USE_ALT_LAYOUT`**: update both shape metadata and byte
  order for weights; the macro alone does not transform existing buffers.
- **Wrong Q4/Q2/Q1 result**: verify values-per-byte, field shift, sign/Q2
  mapping, logical tail padding, and whether the kernel expects packed data or
  logical bytes.
- **Quantization buffer overflow**: reduce block size, increase
  `QUANTIZE_BUFFER_SIZE` for the target, or redesign workspace usage. Do not
  write past `MiCo_QBuffer`.
- **Host low-bit wrapper assertion**: the current default-layout FP32 wrappers
  restrict some low-bit combinations to RVV; check the assertions in
  `src/mico/bitlinear.c` and `bitconv2d.c` instead of assuming every qtype works
  on host.
- **Illegal instruction on Spike**: ensure `MARCH` and `SPIKE_ISA` both enable
  the required extension and match the selected target source.
- **No vector instructions in disassembly**: verify `TARGET=rvv`, avoid the
  conflicting `OPT` variants, and confirm the RVV object is linked instead of
  only the weak baseline.
- **`htif_nano.specs` not found**: use the Chipyard RISC-V compiler rather than
  a host compiler or unrelated RISC-V toolchain.
- **Generated C says a function is not registered**: import the registry before
  tracing, register the exact PyTorch callable/module type, and verify the
  handler's parameter extraction.
- **Generated C has an unknown `MiCo_*` symbol**: declare and compile the C
  implementation, then inspect the generated function name and dtype/rank
  suffix. `add_forward_call` emits output first and uses `f32` for FP32.
- **A view or flatten corrupts another tensor**: determine whether the operation
  intentionally aliases `data`; otherwise allocate/write a distinct output and
  validate lifetime against CodeGen memory-pool reuse.

## Completion Report

Finish with:

- changed headers, baseline/optimized sources, tests, and CodeGen handlers;
- the exact `make` commands and target/`OPT` combinations run;
- numerical comparison results, generated artifacts, and symbol/disassembly
  evidence when applicable;
- unavailable toolchains, simulators, data, or unverified hardware steps;
- any remaining contract limitation, especially layout, dilation, qtype, or
  alignment restrictions.
