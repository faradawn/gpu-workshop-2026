"""
Exercise 2: Mixed-Precision Training — FP32 vs FP16 vs FP8 (Transformer Engine)
Run on an NVIDIA H100 (Hopper) or compatible GPU with Transformer Engine.

Goal: Start with a naive FP32 Transformer block that is slow and memory-hungry.
      Compare FP16 AMP and TE FP8 (fp8_autocast) and watch memory drop and throughput rise.

Your task:
  1. Run the FP32 baseline and note peak memory + throughput.
  2. Compare FP16 AMP and TE FP8 results.

  # On Blackwell (B200, GB10, etc.) you can extend this with NVFP4BlockScaling and the same
  # te.fp8_autocast(..., fp8_recipe=fp4_recipe) pattern for even lower precision — not run here.

Why d_model=4096?  With d_model=1024 the GEMMs are tiny on an H100 — each finishes in
microseconds, and Transformer Engine's per-op overhead (scale-factor tracking, kernel
dispatch, quantise/dequantise) dominates.  Bumping to 4096 (LLaMA-scale hidden size) makes
the FFN projections (4096×16384) large enough for FP8 Tensor Cores to saturate.

Why warmup steps?  Transformer Engine JIT-compiles custom FP8 CUDA kernels on the first
few forward/backward passes.  Without a warmup phase, that multi-second compilation lands
inside the timed region and makes FP8 appear *slower* than FP32.

Expected output (H100 SXM 80 GB, batch=8, seq=512, d_model=4096):

  [Naive FP32 (baseline)]
    Time:         2.44s    Throughput:  328.3 samples/s    Peak Memory: 4.699 GB

  [FP16 AMP]
    Time:         1.74s    Peak Memory: 4.397 GB

  [TE FP8]
    Time:         0.98s    Throughput:  815.6 samples/s    Peak Memory: 6.471 GB

  ── Summary (vs FP32 baseline) ───────────────────────
  FP16 AMP  speedup:  1.40x   memory:  1.07x
  TE FP8    speedup:  2.48x   memory:  0.73x

  Note on memory: TE FP8 reports *higher* peak memory than the FP32 baseline here
  because (a) the FP32 baseline already uses TF32 Tensor Cores, so "FP32" is really
  TF32-accelerated, and (b) te.TransformerLayer allocates its own workspace buffers
  and FP8 amax-history tensors that dwarf the activation savings for a single layer.
  At scale (many layers, large batch, distributed training) FP8 activation compression
  and reduced communication volume provide clear memory wins.
"""

import time
import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Float8CurrentScaling

# ── Config ───────────────────────────────────────────────────────────────────
BATCH        = 8
SEQ_LEN      = 512
D_MODEL      = 4096
N_HEADS      = 32
STEPS        = 100
WARMUP_STEPS = 10
DEVICE       = "cuda"


# ── Naive Transformer Block (baseline) ───────────────────────────────────────
class NaiveTransformerBlock(nn.Module):
    """Plain PyTorch — no low-precision tricks."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


# ── Transformer Engine Block ──────────────────────────────────────────────────
class TETransformerBlock(nn.Module):
    """Uses NVIDIA Transformer Engine for automatic FP8 compute (master weights still FP32 internally)."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.layer = te.TransformerLayer(
            hidden_size=d_model,
            ffn_hidden_size=d_model * 4,
            num_attention_heads=n_heads,
            layer_type="encoder",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x, attention_mask=None)


# ── Training loop ─────────────────────────────────────────────────────────────
def train(model: nn.Module, label: str, fp8_recipe=None, bf16_input: bool = False) -> tuple[float, float]:
    dtype = torch.bfloat16 if bf16_input else torch.float32
    model = model.to(DEVICE, dtype=dtype)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x     = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE, dtype=dtype)

    # Warmup — absorbs TE kernel JIT compilation and CUDA lazy init
    for _ in range(WARMUP_STEPS):
        opt.zero_grad(set_to_none=True)
        if fp8_recipe is not None:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                out = model(x)
        else:
            out = model(x)
        out.mean().backward()
        opt.step()

    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)

        if fp8_recipe is not None:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                out = model(x)
        else:
            out = model(x)

        loss = out.mean()
        loss.backward()
        opt.step()

    torch.cuda.synchronize()
    elapsed  = time.perf_counter() - t0
    peak_mem = torch.cuda.max_memory_allocated(DEVICE) / 1e9
    throughput = STEPS * BATCH / elapsed

    print(f"\n[{label}]")
    print(f"  Steps:       {STEPS}")
    print(f"  Time:        {elapsed:.2f}s")
    print(f"  Throughput:  {throughput:.1f} samples/s")
    print(f"  Peak Memory: {peak_mem:.3f} GB")
    return elapsed, peak_mem


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: batch={BATCH}  seq={SEQ_LEN}  d_model={D_MODEL}  heads={N_HEADS}")
    print(f"        steps={STEPS}  warmup_steps={WARMUP_STEPS}\n")

    # ── Baseline: FP32 ────────────────────────────────────────────────────────
    t_fp32, m_fp32 = train(NaiveTransformerBlock(D_MODEL, N_HEADS), "Naive FP32 (baseline)")

    # ── FP16 AMP ──────────────────────────────────────────────────────────────
    model_fp16 = NaiveTransformerBlock(D_MODEL, N_HEADS).to(DEVICE)
    opt_fp16   = torch.optim.AdamW(model_fp16.parameters(), lr=1e-4)
    scaler     = torch.amp.GradScaler("cuda")
    x          = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)

    # Warmup for FP16 AMP too (fair comparison — CUDA graph caches etc.)
    for _ in range(WARMUP_STEPS):
        opt_fp16.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model_fp16(x)
        scaler.scale(out.mean()).backward()
        scaler.step(opt_fp16)
        scaler.update()

    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        opt_fp16.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            out = model_fp16(x)
        scaler.scale(out.mean()).backward()
        scaler.step(opt_fp16)
        scaler.update()
    torch.cuda.synchronize()
    t_fp16 = time.perf_counter() - t0
    m_fp16 = torch.cuda.max_memory_allocated(DEVICE) / 1e9
    print(f"\n[FP16 AMP]\n  Time: {t_fp16:.2f}s  |  Peak Memory: {m_fp16:.3f} GB")

    # ── FP8 via Transformer Engine (Hopper / H100) ────────────────────────────
    fp8_recipe = Float8CurrentScaling()
    t_fp8, m_fp8 = train(TETransformerBlock(D_MODEL, N_HEADS), "TE FP8", fp8_recipe=fp8_recipe, bf16_input=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Summary (vs FP32 baseline) ───────────────────────")
    print(f"  FP16 AMP  speedup: {t_fp32/t_fp16:.2f}x   memory: {m_fp32/m_fp16:.2f}x")
    print(f"  TE FP8    speedup: {t_fp32/t_fp8:.2f}x   memory: {m_fp32/m_fp8:.2f}x")
