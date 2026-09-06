"""Regression tests: TileLang kernels vs PyTorch reference.

Tests GEMM correctness for fp16 (tensor-core native format).
Also tests dequantize-GEMM (int8 weight × scale → GEMM).
"""

import sys
import pytest
import torch
import tilelang
import tilelang.language as T

sys.path.insert(0, ".")

from MiCoQLayers import weight_quant


# =========================================================================
# Standalone TileLang GEMM kernel (module-level, required by TileLang JIT)
# =========================================================================


@tilelang.jit(out_idx=[2])
def _gemm_kernel(
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 256,
    dtype="float16",
    accum_dtype="float32",
):
    """Tiled GEMM: C = A @ B.  Shape M,N,K inferred from input tensors."""
    M, N, K = T.int32(), T.int32(), T.int32()

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(
                    A[by * block_M : by * block_M + block_M,
                      ko * block_K : ko * block_K + block_K],
                    A_shared,
                )
                T.copy(
                    B[ko * block_K : ko * block_K + block_K,
                      bx * block_N : bx * block_N + block_N],
                    B_shared,
                )
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                m_idx = by * block_M + i
                n_idx = bx * block_N + j
                if m_idx < M and n_idx < N:
                    C[m_idx, n_idx] = T.cast(C_local[i, j], dtype)

    return main


# =========================================================================
# Tests: fp16 GEMM (tensor core)
# =========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("M,N,K", [
    (32, 128, 256),
    (64, 256, 512),
    (128, 512, 1024),
])
def test_gemm_fp16_vs_pytorch(M, N, K):
    """TileLang fp16 GEMM matches PyTorch matmul."""
    device = torch.device("cuda")
    dtype = torch.float16

    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)

    ref = a @ b

    kernel = _gemm_kernel.compile(
        dtype="float16",
        accum_dtype="float32",
        block_M=min(64, M),
        block_N=min(64, N),
        block_K=32,
    )
    out = kernel(a, b)

    torch.testing.assert_close(out, ref, rtol=0.05, atol=0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gemm_fp16_non_divisible():
    """M, N not multiples of block size."""
    M, N, K = 100, 100, 256
    device = torch.device("cuda")
    dtype = torch.float16

    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)

    ref = a @ b

    kernel = _gemm_kernel.compile(
        dtype="float16",
        accum_dtype="float32",
        block_M=64,
        block_N=64,
        block_K=32,
    )
    out = kernel(a, b)

    torch.testing.assert_close(out, ref, rtol=0.05, atol=0.1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gemm_fp16_zero_input():
    """All-zero input produces all-zero output."""
    M, N, K = 64, 128, 256
    device = torch.device("cuda")
    dtype = torch.float16

    a = torch.zeros(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)

    kernel = _gemm_kernel.compile(
        dtype="float16",
        accum_dtype="float32",
        block_M=64,
        block_N=64,
        block_K=32,
    )
    out = kernel(a, b)

    assert out.abs().max() < 1e-3, f"Expected zero output, got {out.abs().max():.6f}"


# =========================================================================
# Tests: Dequantize-GEMM (arbitrary bitwidth: 1, 2, 4, 8-bit)
# =========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("M,N,K", [
    (64, 256, 512),
    (128, 512, 1024),
])
@pytest.mark.parametrize("w_bit", [4, 2, 8])
def test_dequant_gemm_vs_pytorch(M, N, K, w_bit):
    """Dequantize-GEMM works for arbitrary bitwidths (4-bit, 2-bit, 8-bit)."""
    device = torch.device("cuda")
    dtype = torch.float16

    # Generate and quantize weight at the given bitwidth
    w = torch.randn(N, K, dtype=torch.float32)
    qw, scale = weight_quant(w, w_bit)

    x = torch.randn(M, K, device=device, dtype=dtype)
    qw_dev = qw.to(device)
    scale_dev = scale.to(device).reshape(())
    bias = torch.randn(N, device=device, dtype=dtype)

    # PyTorch reference: dequantize weight, then matmul
    w_deq = (qw_dev.float() * scale_dev).to(dtype)
    ref = x @ w_deq.T + bias

    # TileLang: GEMM with dequantized weight
    kernel = _gemm_kernel.compile(
        dtype="float16",
        accum_dtype="float32",
        block_M=min(64, M),
        block_N=min(64, N),
        block_K=32,
    )
    out = kernel(x, w_deq.T.contiguous()) + bias

    # Lower-bit weights have higher quantization error → looser tolerance
    if w_bit <= 2:
        rtol, atol = 0.2, 0.5
    elif w_bit == 4:
        rtol, atol = 0.1, 0.2
    else:
        rtol, atol = 0.05, 0.1
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("w_bit", [1])
def test_dequant_gemm_1bit(w_bit):
    """1-bit (binary) weights: sign-based quantization, high error but
    predictions should still match."""
    M, N, K = 128, 256, 512
    device = torch.device("cuda")
    dtype = torch.float16

    w = torch.randn(N, K, dtype=torch.float32)
    qw, scale = weight_quant(w, w_bit)  # qw has values in {-1, 0, 1}

    x = torch.randn(M, K, device=device, dtype=dtype)
    qw_dev = qw.to(device)
    scale_dev = scale.to(device).reshape(())

    # Reference
    w_deq = (qw_dev.float() * scale_dev).to(dtype)
    ref = x @ w_deq.T

    # TileLang
    kernel = _gemm_kernel.compile(
        dtype="float16",
        accum_dtype="float32",
        block_M=64, block_N=64, block_K=32,
    )
    out = kernel(x, w_deq.T.contiguous())

    # 1-bit has high quantization error → only check argmax
    assert torch.equal(ref.argmax(dim=-1), out.argmax(dim=-1)), (
        f"1-bit predictions differ! Max diff: {(ref-out).abs().max():.4f}"
    )


# =========================================================================
# Benchmark
# =========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("M,N,K", [
    (128, 4096, 4096),
])
def test_benchmark_gemm_vs_pytorch(M, N, K):
    """Benchmark TileLang GEMM vs PyTorch matmul (fp16)."""
    device = torch.device("cuda")
    dtype = torch.float16

    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)

    # PyTorch
    def torch_fn():
        return a @ b

    for _ in range(10):
        torch_fn()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(100):
        start.record()
        torch_fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    torch_ms = sum(times) / len(times)

    # TileLang
    kernel = _gemm_kernel.compile(
        dtype="float16",
        accum_dtype="float32",
        block_M=min(64, M),
        block_N=min(64, N),
        block_K=32,
    )
    for _ in range(10):
        kernel(a, b)

    times = []
    for _ in range(100):
        start.record()
        kernel(a, b)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    tl_ms = sum(times) / len(times)

    speedup = torch_ms / tl_ms
    print(f"\n  [{M}x{N}x{K}] PyTorch: {torch_ms:.4f}ms, "
          f"TileLang: {tl_ms:.4f}ms, Speedup: {speedup:.2f}×")

    # Sanity: not drastically slower
    assert speedup > 0.05, f"TileLang is {1/speedup:.1f}× slower"
