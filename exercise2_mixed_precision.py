"""
Exercise 2: Mixed-Precision Training — NVFP4 vs FP16
Run on a Blackwell GPU (DGX Spark, RTX 6000, or B200)

Goal: Start with a naive FP32 Transformer block that is slow and memory-hungry.
      Add te.fp8_autocast (or NVFP4) and watch memory drop and throughput rise.

Your task:
  1. Run the FP32 baseline and note peak memory + throughput.
  2. Uncomment the FP8 section (marked TODO) and re-run.
  3. Try NVFP4 on Blackwell and compare all three.
"""

import time
import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Float8CurrentScaling, Format, NVFP4BlockScaling

# ── Config ───────────────────────────────────────────────────────────────────
BATCH     = 8
SEQ_LEN   = 512
D_MODEL   = 1024
N_HEADS   = 16
STEPS     = 50
DEVICE    = "cuda"


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
    """Uses NVIDIA Transformer Engine for automatic FP8 / NVFP4 casting."""

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
    model = model.to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dtype = torch.bfloat16 if bf16_input else torch.float32
    x     = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE, dtype=dtype)

    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)

        # ── TODO: wrap this block with te.fp8_autocast to enable low precision ──
        if fp8_recipe is not None:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                out = model(x)
        else:
            out = model(x)
        # ────────────────────────────────────────────────────────────────────────

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
    print(f"Config: batch={BATCH}  seq={SEQ_LEN}  d_model={D_MODEL}  heads={N_HEADS}\n")

    # ── Baseline: FP32 ────────────────────────────────────────────────────────
    t_fp32, m_fp32 = train(NaiveTransformerBlock(D_MODEL, N_HEADS), "Naive FP32 (baseline)")

    # ── FP16 AMP ──────────────────────────────────────────────────────────────
    model_fp16 = NaiveTransformerBlock(D_MODEL, N_HEADS).to(DEVICE)
    opt_fp16   = torch.optim.AdamW(model_fp16.parameters(), lr=1e-4)
    scaler     = torch.cuda.amp.GradScaler()
    x          = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=DEVICE)

    torch.cuda.reset_peak_memory_stats(DEVICE)
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

    # ── FP8 via Transformer Engine ────────────────────────────────────────────
    # Float8CurrentScaling is the modern recipe (replaces DelayedScaling); needs bfloat16 input
    fp8_recipe = Float8CurrentScaling()
    t_fp8, m_fp8 = train(TETransformerBlock(D_MODEL, N_HEADS), "TE FP8", fp8_recipe=fp8_recipe, bf16_input=True)

    # ── NVFP4 via Transformer Engine (Blackwell only) ─────────────────────────
    # NVFP4BlockScaling uses Format.E2M1 internally.
    # disable_rht=True avoids a shared-memory limitation on the GB10 (DGX Spark).
    # Input must be bfloat16.
    t_fp4, m_fp4 = None, None
    try:
        fp4_recipe = NVFP4BlockScaling(disable_rht=True)
        t_fp4, m_fp4 = train(
            TETransformerBlock(D_MODEL, N_HEADS),
            "TE NVFP4 (Blackwell)",
            fp8_recipe=fp4_recipe,
            bf16_input=True,
        )
    except Exception as e:
        print(f"\n[NVFP4] Not available: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Summary (vs FP32 baseline) ───────────────────────")
    print(f"  FP16 AMP  speedup: {t_fp32/t_fp16:.2f}x   memory: {m_fp32/m_fp16:.2f}x")
    print(f"  TE FP8    speedup: {t_fp32/t_fp8:.2f}x   memory: {m_fp32/m_fp8:.2f}x")
    if t_fp4:
        print(f"  TE NVFP4  speedup: {t_fp32/t_fp4:.2f}x   memory: {m_fp32/m_fp4:.2f}x  ← Blackwell")
