"""
Exercise 1: Tensor Core Benchmark — NVFP4 vs FP16 Matrix Multiplication
Run on a Blackwell GPU (DGX Spark, RTX 6000, or B200)

Goal: See how Tensor Cores accelerate lower-precision matmuls.
      FP16 → FP8 → NVFP4: each step roughly doubles TFLOP/s on Blackwell.

Expected output (B200 / GB10, 4096×4096 matrix) — NOT actual results:
  ── Standard PyTorch Matmul ──────────────────────────
  [FP32  ]    3.200 ms    43.01 TFLOP/s
  [FP16  ]    0.820 ms   167.88 TFLOP/s
  [BF16  ]    0.820 ms   167.88 TFLOP/s

  ── Transformer Engine (Tensor Core, low precision) ──
  [FP8   ]    0.420 ms   327.87 TFLOP/s
  [NVFP4 ]    0.220 ms   625.74 TFLOP/s  ← Blackwell Tensor Cores

  ── Speedup vs FP32 ──────────────────────────────────
  FP16:  3.90x
  BF16:  3.90x
  FP8:   7.62x
  NVFP4: 14.55x  ← Blackwell Tensor Cores
"""

import time
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Float8CurrentScaling, Format, NVFP4BlockScaling

MATRIX_SIZE = 4096
WARMUP = 20
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


def bench_te_linear(recipe, label: str, bf16: bool = False) -> float:
    dtype = torch.bfloat16 if bf16 else torch.float32
    linear = te.Linear(MATRIX_SIZE, MATRIX_SIZE, bias=False).to(DEVICE).to(dtype)
    x = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=DEVICE, dtype=dtype)

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
    tf_bf16 = bench_torch_mm(torch.bfloat16, "BF16")

    print("\n── Transformer Engine (Tensor Core, low precision) ──")
    # FP8 — works on Hopper and Blackwell
    fp8_recipe = Float8CurrentScaling()
    tf_fp8 = bench_te_linear(fp8_recipe, "FP8", bf16=True)

    # NVFP4 — Blackwell-exclusive (B200, RTX 6000, DGX Spark GB10)
    # Requires bfloat16 input; disable_rht=True avoids shared-memory constraint on GB10
    tf_fp4 = None
    try:
        fp4_recipe = NVFP4BlockScaling(disable_rht=True)
        tf_fp4 = bench_te_linear(fp4_recipe, "NVFP4", bf16=True)
    except Exception as e:
        print(f"  [NVFP4]  Not available on this GPU: {e}")

    print("\n── Speedup vs FP32 ──────────────────────────────────")
    for label, tf in [("FP16", tf_fp16), ("BF16", tf_bf16), ("FP8", tf_fp8)]:
        print(f"  {label}: {tf / tf_fp32:.2f}x")
    if tf_fp4:
        print(f"  NVFP4: {tf_fp4 / tf_fp32:.2f}x  ← Blackwell Tensor Cores")
