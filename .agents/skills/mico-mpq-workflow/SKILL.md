---
name: mico-mpq-workflow
description: Work on the MiCo mixed-precision quantization repository: understand model, dataset, training, MPQ search, hardware-aware evaluation, code generation, C deployment, experiments, and related tests. Use when a request touches MiCoModel, MiCoQLayers, MiCoEval, searchers, MiCoCodeGen, MiCoRegistry, models/, examples/, deploy/, project/, or quantization experiments.
---

# MiCo MPQ Workflow

Use this skill to make repository changes and run experiments in a way that preserves MiCo's model, search, and deployment contracts.

## First Pass

Before editing or running an expensive experiment:

1. Read `readme.md` for the public workflow and `CLAUDE.md` for the short engineering guide.
2. Read the target module, its closest example under `examples/`, and the relevant test under `tests/`.
3. Check `git status --short`. Treat existing modifications and untracked experiment files as user-owned; do not reset or clean them.
4. Classify the request as one or more of: model/training, MPQ search, evaluation/metrics, hardware proxy, CodeGen, deployment, registry extension, or test/documentation.
5. Prefer a small, reproducible smoke run before a full training or search run. Ask before starting a long GPU, RTL, or multi-trial experiment when the request does not clearly require it.

Set the import path from the repository root for commands that import local modules:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
```

## Repository Map

- `MiCoModel.py`: base model API, training/testing, and qscheme application.
- `MiCoQLayers.py`: fake-quantized `BitLinear`, `BitConv1d`, `BitConv2d`, and related quantization code.
- `MiCoUtils.py`: layer replacement, fusion, export, and shared utilities.
- `MiCoEval.py`: accuracy, BOPs, MACs, latency, and constraint evaluation.
- `MiCoProxy.py`: offline hardware latency proxy models.
- `searchers/`: MiCo, Bayesian, HAQ, NLP, regression, and sampling searchers.
- `models/`: `MiCoModel` architectures and `model_zoo.py` lookup.
- `MiCoCodeGen.py`: Torch FX based C generator and memory-pool allocation.
- `MiCoRegistry.py`: extension point for custom functions and modules used by CodeGen.
- `MiCoGraphGen.py`, `MiCoLLaMaGen.py`: specialized deployment generators.
- `examples/`: executable training, search, evaluation, and generation patterns.
- `deploy/`: hardware-aware end-to-end examples.
- `project/`: C inference templates; depends on the MiCo-Lib git submodule.
- `tests/`: unit and integration tests.
- `output/`: generated checkpoints, traces, figures, text results, and exported artifacts.

## MiCo Contracts

### qscheme

A qscheme is exactly:

```python
[[w_bit_layer0, w_bit_layer1, ...], [a_bit_layer0, a_bit_layer1, ...]]
```

Both lists must have length `model.n_layers`. Validate this before calling `set_qscheme`. Keep weight and activation lists distinct even when they happen to contain the same values.

`set_qscheme(...)` replaces quantizable layers. Reapplying it replaces them again, so do not repeatedly apply it inside a hot evaluation loop unless the surrounding code expects that behavior. `qat=False` uses fake-quantized inference behavior; `qat=True` keeps the quantized layers trainable.

A model added to the search pipeline should inherit from `MiCoModel` (or use the repository's conversion helper), expose the correct `n_layers`, and use supported quantizable layers. Confirm the layer count from the model implementation instead of guessing it from the number of modules.

### Training and evaluation

- `MiCoModel.train_loop(...)` trains in place and does not return the model.
- Save a named checkpoint under `output/ckpt/`, normally `<model>_<dataset>.pth`.
- Set seeds for `random`, NumPy, and PyTorch when a result is intended to be compared or reported.
- Reuse the repository dataset loader and the model's existing train/test methods before introducing a new loop.
- For an experiment, record model, dataset, qtypes, seed, checkpoint, objective, constraint type, constraint value, trial count, and the resulting accuracy/metric.

### MPQ search

Use the smallest existing example that matches the request. The general entry point is:

```bash
python examples/mpq_search.py <model_name> --init 16 -n 40 -c 0.5 \
  -ctype bops -t 5 -m ptq_acc
```

Relevant choices include:

- `-ctype`: `bops`, `size`, `latency_mico`, `latency_bitfusion`, or `latency_proxy`.
- `-m`: `ptq_acc` or `qat_acc`.
- `--init`: initial random samples; `-n`: search iterations; `-t`: random-seed trials.

For new search-related code, start with a small configuration such as `n_inits=5`, `qtypes=[6, 7, 8]`, and 5-10 iterations. Scale only after imports, one evaluator call, constraint calculation, and result serialization work. For models with many quantizable layers, evaluate whether `DimTransform` is appropriate before increasing search dimensions.

When using hardware-aware search, configure both proxy types where needed, call the evaluator's proxy setup, and set the intended MiCo target. Do not present proxy latency as measured hardware latency.

### Code generation and deployment

Before CodeGen, apply the selected qscheme, load the matching checkpoint, fuse the model, and switch to evaluation mode:

```python
model.set_qscheme(qscheme)
model.load_state_dict(torch.load(checkpoint))
model = fuse_model(model)
model.eval()

codegen = MiCoCodeGen(model, align_to=32)
codegen.forward(example_input)
codegen.convert("output", "model_name")
```

The usual generated artifacts are `<name>.h` and `<name>.bin`. Verify they exist and inspect their sizes before compiling. `align_to=32` is the normal SIMD-friendly default unless the target requires another alignment.

For C compilation, initialize the submodule only when deployment is requested:

```bash
git submodule update --init
cd project
make clean
make MAIN=main TARGET=host OPT=unroll
make run-host
```

Use the target requested by the user (`host`, `vexii`, `rocket`, or `spike`) and do not claim successful hardware deployment when only host compilation was run. For models with skip connections, keep the memory-pool behavior in the validation scope; do not disable it just to make a generated file look simpler.

### Custom CodeGen operations

Use `MiCoOpRegistry.register_function` or `MiCoOpRegistry.register_module` for new operations. Do not add a one-off branch to `MiCoCodeGen.py` when the registry can express the handler. Confirm the exact handler signature from `doc/REGISTRY_USAGE.md` and add a focused regression test covering:

1. PyTorch forward behavior.
2. FX/CodeGen tracing and generated call.
3. C compilation or artifact generation when the required submodule/toolchain is available.

## Validation Ladder

Choose the lowest level that proves the change, then expand based on risk:

1. Syntax/import check for touched Python modules.
2. Focused unit test, for example `python -m pytest tests/test_micocodegen.py -v`.
3. Small model forward/evaluation or 5-10 iteration search.
4. Full relevant test file or `python -m pytest tests/ -v`.
5. CodeGen artifact inspection, host compile, and `make run-host` for deployment changes.

Do not substitute a passing import check for an accuracy, search, generated-code, or deployment verification. Report skipped checks and their prerequisite (GPU, data download, submodule, simulator, or package) explicitly.

## Change Rules

- Follow existing APIs and examples before adding abstractions.
- Keep experiment outputs in the established `output/` layout and use descriptive names.
- Avoid broad formatting or cleanup in research files with active local changes.
- Do not change core quantization or CodeGen behavior without a focused test and a minimal reproducer.
- Preserve user changes in modified files; inspect the diff before editing a file that is already dirty.
- Prefer deterministic fixtures and tiny tensors in tests; never make tests depend on a downloaded dataset or hardware simulator unless they are explicitly integration tests.

## Completion Report

Finish with a concise record of:

- files changed and the behavior changed;
- commands/tests run and their outcome;
- experiment settings and artifact paths when applicable;
- remaining limitations, unavailable dependencies, or unverified hardware steps.
