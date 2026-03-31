"""
Exercise 3: Inference — Standard vs EAGLE3 Speculative Decoding with vLLM
Run on an NVIDIA H100 or similar high-throughput inference GPU.

Goal: Compare per-request latency between vanilla autoregressive decoding
      and EAGLE3 speculative decoding.

IMPORTANT — speculative decoding is a *latency* optimisation for single requests,
NOT a throughput optimisation.  It helps when the GPU is memory-bound (batch=1,
waiting on KV-cache reads).  When you batch many prompts together the GPU becomes
compute-bound, the draft-then-verify overhead actually hurts, and you get < 1x.

Best workloads for speculative decoding (from SpecDecode-Bench & EAGLE-3 paper):
  - Code generation (HumanEval): highly structured, predictable tokens → ~2.3x
  - Math / reasoning (GSM8K, MATH): repetitive algebraic patterns     → ~2.0x
  - Chat / MT-Bench: moderate structure                                → ~2.0x
  - Worst case: high-entropy creative writing, large batch sizes

How speculative decoding works:
  - A small "draft" model proposes K tokens at once.
  - The large "target" model verifies all K in a single forward pass.
  - Accepted tokens are kept; rejected ones are resampled.
  - Net result: more tokens per forward pass → lower latency per request.

Models used (change to match your local paths or HF repo IDs):
  TARGET_MODEL  — the full model being accelerated
  DRAFT_MODEL   — EAGLE3 draft head weights

Expected output (H100 SXM 80 GB, Qwen2.5-7B, batch=1, 512 tokens) — illustrative:

  ── Standard Autoregressive Decoding (batch=1) ───────
  [Code gen  ]  512 tokens  22.4 TPS
  [Math      ]  512 tokens  21.8 TPS
  [Reasoning ]  512 tokens  23.1 TPS
  avg 22.4 TPS

  ── EAGLE3 Speculative Decoding (batch=1) ────────────
  [Code gen  ]  512 tokens  48.2 TPS
  [Math      ]  512 tokens  44.6 TPS
  [Reasoning ]  512 tokens  42.1 TPS
  avg 44.9 TPS

  ── PER-PROMPT RESULTS ───────────────────────────────
  Prompt        Standard (TPS)   EAGLE3 (TPS)   Speedup
  Code gen              22.4          48.2      2.15x
  Math                  21.8          44.6      2.05x
  Reasoning             23.1          42.1      1.82x
  Average               22.4          44.9      2.01x
"""

import time
from vllm import LLM, SamplingParams

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_MODEL    = "Qwen/Qwen2.5-7B-Instruct"
DRAFT_MODEL     = "ruipeterpan/Qwen2.5-7B-Instruct_EAGLE3_UltraChat"
NUM_SPEC_TOKENS = 5
MAX_NEW_TOKENS  = 512

SAMPLING_PARAMS = SamplingParams(
    temperature=0.0,       # greedy — maximises acceptance rate
    max_tokens=MAX_NEW_TOKENS,
)

# ── Prompts chosen for high acceptance rate (code, math, structured reasoning) ─
PROMPTS = [
    (
        "Code gen",
        "Write a complete Python implementation of a binary search tree with insert, "
        "delete, search, and in-order traversal. Include type hints and docstrings "
        "for every method.",
    ),
    (
        "Math",
        "Solve step by step: A factory produces widgets at a rate that increases by "
        "12% each quarter. Starting at 1000 widgets/month in Q1, calculate the "
        "cumulative production over 8 quarters. Show every intermediate calculation.",
    ),
    (
        "Reasoning",
        "Compare the KV-cache memory layout in vLLM (paged attention with fixed "
        "block sizes) versus SGLang (radix tree). Explain the trade-offs in memory "
        "fragmentation, prefix sharing efficiency, and implementation complexity.",
    ),
    (
        "Code gen",
        "Write a Python async web scraper using aiohttp and BeautifulSoup that "
        "crawls up to 100 pages from a given root URL, respects robots.txt, "
        "extracts all links, and stores results in a SQLite database. "
        "Include error handling and rate limiting.",
    ),
    (
        "Math",
        "Derive the backpropagation equations for a two-layer neural network with "
        "ReLU activation and cross-entropy loss. Show each partial derivative step "
        "by step, starting from the loss function.",
    ),
]


# ── Benchmark: one prompt at a time (batch=1 latency measurement) ─────────────
def run_benchmark(llm: LLM, label: str) -> list[dict]:
    print(f"\n{'='*60}")
    print(f" {label}")
    print(f"{'='*60}")

    # Warmup — first call compiles CUDA graphs, warms caches
    llm.generate([PROMPTS[0][1]], SAMPLING_PARAMS)

    results = []
    for tag, prompt in PROMPTS:
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], SAMPLING_PARAMS)
        t1 = time.perf_counter()

        n_tok   = len(outputs[0].outputs[0].token_ids)
        elapsed = t1 - t0
        tps     = n_tok / elapsed
        snippet = outputs[0].outputs[0].text[:100].replace("\n", " ")

        print(f"  [{tag:10s}]  {n_tok:4d} tokens  {tps:6.1f} TPS  |  {snippet}...")
        results.append({"tag": tag, "tokens": n_tok, "elapsed": elapsed, "tps": tps})

    total_tok = sum(r["tokens"] for r in results)
    total_sec = sum(r["elapsed"] for r in results)
    avg_tps   = total_tok / total_sec
    print(f"\n  Total: {total_tok} tokens in {total_sec:.2f}s  →  avg {avg_tps:.1f} TPS")
    return results


if __name__ == "__main__":
    # ── Standard autoregressive decoding ──────────────────────────────────────
    print("Loading standard model...")
    llm_standard = LLM(
        model=TARGET_MODEL,
        dtype="bfloat16",
        tensor_parallel_size=1,
    )
    res_std = run_benchmark(llm_standard, "Standard Autoregressive Decoding (batch=1)")
    del llm_standard

    # ── EAGLE3 speculative decoding ───────────────────────────────────────────
    print("\nLoading EAGLE3 speculative model...")
    llm_eagle3 = LLM(
        model=TARGET_MODEL,
        dtype="bfloat16",
        tensor_parallel_size=1,
        speculative_config={
            "model": DRAFT_MODEL,
            "method": "eagle3",
            "num_speculative_tokens": NUM_SPEC_TOKENS,
            "draft_tensor_parallel_size": 1,
        },
    )
    res_eagle = run_benchmark(llm_eagle3, f"EAGLE3 Speculative Decoding (batch=1, draft_tokens={NUM_SPEC_TOKENS})")
    del llm_eagle3

    # ── Per-prompt comparison ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f" PER-PROMPT RESULTS")
    print(f"{'='*60}")
    print(f"  {'Prompt':<12s}  {'Standard (TPS)':>15s}  {'EAGLE3 (TPS)':>13s}  {'Speedup':>8s}")
    print(f"  {'-'*12}  {'-'*15}  {'-'*13}  {'-'*8}")

    speedups = []
    for s, e in zip(res_std, res_eagle):
        sp = e["tps"] / s["tps"]
        speedups.append(sp)
        print(f"  {s['tag']:<12s}  {s['tps']:>12.1f}    {e['tps']:>10.1f}    {sp:>7.2f}x")

    avg_std   = sum(r["tokens"] for r in res_std)   / sum(r["elapsed"] for r in res_std)
    avg_eagle = sum(r["tokens"] for r in res_eagle) / sum(r["elapsed"] for r in res_eagle)
    overall   = avg_eagle / avg_std

    print(f"\n  {'Average':<12s}  {avg_std:>12.1f}    {avg_eagle:>10.1f}    {overall:>7.2f}x")
    print()
    if overall > 1.5:
        print("  Speculative decoding is providing meaningful latency reduction at batch=1.")
    else:
        print("  Tip: try num_speculative_tokens 3-8, ensure temperature=0 (greedy),")
        print("  and use code-generation or math prompts for highest acceptance rates.")
