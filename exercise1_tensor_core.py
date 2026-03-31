"""
Exercise 1: Tensor Core Benchmark — FP32 → FP16 → FP8 Matrix Multiplication
Run on an NVIDIA H100 (Hopper) or compatible datacenter GPU.

Goal: See how Tensor Cores accelerate the precision ladder.
      FP32 → FP16 → FP8: each step increases effective TFLOP/s and reduces memory traffic.

Why 8192×8192?  Smaller matrices (e.g. 4096) finish in <0.2 ms on H100, so Transformer
Engine's per-call overhead (scale-factor bookkeeping, quantise/dequantise) dominates and
FP8 appears no faster than FP16.  Larger matrices push compute time well above that
overhead floor, revealing the true Tensor Core throughput difference.

Expected output (H100 SXM 80 GB, 8192×8192 matrix):
  ── Standard PyTorch Matmul ──────────────────────────
  [FP32  ]    2.873 ms   382.74 TFLOP/s
  [FP16  ]    1.614 ms   681.06 TFLOP/s

  ── Transformer Engine (Tensor Core FP8) ────────────
  [FP8   ]    1.015 ms  1083.69 TFLOP/s

  ── Speedup vs FP32 ──────────────────────────────────
  FP16:  1.78x
  FP8:   2.83x

  Note: PyTorch uses TF32 Tensor Cores for float32 matmul by default on Hopper,
  so the "FP32" row is already Tensor-Core-accelerated.  Peak non-sparse rates for
  H100 SXM: TF32 495 / FP16 990 / FP8 1979 TFLOP/s — the FP8 path through
  Transformer Engine's te.Linear adds per-call overhead (scale-factor tracking,
  quantise/dequantise), which is why the measured 2.83x is below the 4x theoretical.

  FP4 block scaling (e.g. NVFP4) is available on Blackwell GPUs with the same TE APIs;
  this exercise focuses on the Hopper/H100 FP8 path.
"""

import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8CurrentScaling

MATRIX_SIZE = 8192
WARMUP = 50
ITERS = 100
DEVICE = "cuda"


def tflops(ms: float) -> float:
    flops = 2 * MATRIX_SIZE**3
    return flops / (ms * 1e-3) / 1e12


def bench_torch_mm(dtype, label: str) -> float:
    a = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=DEVICE, dtype=dtype)
    b = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=DEVICE, dtype=dtype)

    for _ in range(WARMUP):
        torch.mm(a, b)
    torch.cuda.synchronize()

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(ITERS):
        torch.mm(a, b)
    t1.record()
    torch.cuda.synchronize()

    ms = t0.elapsed_time(t1) / ITERS
    tf = tflops(ms)
    print(f"  [{label:6s}]  {ms:7.3f} ms   {tf:6.2f} TFLOP/s")
    return tf


def bench_te_linear(recipe, label: str) -> float:
    linear = te.Linear(MATRIX_SIZE, MATRIX_SIZE, bias=False).to(DEVICE).to(torch.bfloat16)
    x = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=DEVICE, dtype=torch.bfloat16)

    # TE JIT-compiles FP8 kernels during the first few calls; generous warmup absorbs that.
    for _ in range(WARMUP):
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            linear(x)
    torch.cuda.synchronize()

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(ITERS):
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            linear(x)
    t1.record()
    torch.cuda.synchronize()

    ms = t0.elapsed_time(t1) / ITERS
    tf = tflops(ms)
    print(f"  [{label:6s}]  {ms:7.3f} ms   {tf:6.2f} TFLOP/s")
    return tf


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Matrix: {MATRIX_SIZE}x{MATRIX_SIZE}  |  {ITERS} iterations\n")

    print("── Standard PyTorch Matmul ──────────────────────────")
    tf_fp32 = bench_torch_mm(torch.float32, "FP32")
    tf_fp16 = bench_torch_mm(torch.float16, "FP16")

    print("\n── Transformer Engine (Tensor Core FP8) ────────────")
    fp8_recipe = Float8CurrentScaling()
    tf_fp8 = bench_te_linear(fp8_recipe, "FP8")

    print("\n── Speedup vs FP32 ──────────────────────────────────")
    for label, tf in [("FP16", tf_fp16), ("FP8", tf_fp8)]:
        print(f"  {label}: {tf / tf_fp32:.2f}x")
