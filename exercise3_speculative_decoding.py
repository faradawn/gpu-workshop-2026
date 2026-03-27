"""
Exercise 3: Inference — Standard vs EAGLE3 Speculative Decoding with vLLM
Run on a Blackwell GPU (DGX Spark, RTX 6000, or B200)

Goal: Compare tokens/second between vanilla autoregressive decoding
      and EAGLE3 speculative decoding on a set of complex reasoning prompts.

How speculative decoding works:
  - A small "draft" model proposes K tokens at once.
  - The large "target" model verifies all K in a single forward pass.
  - Accepted tokens are kept; rejected ones are resampled.
  - Net result: more tokens per forward pass → higher throughput.

EAGLE3 improvements over EAGLE:
  - Uses a feature-level (not logit-level) draft head.
  - Trains with a tree-attention objective for better acceptance rates.
  - Achieves ~3-4x speedup on Llama-3 class models.

Models used (change to match your local paths or HF repo IDs):
  TARGET_MODEL  — the full model being accelerated
  DRAFT_MODEL   — EAGLE3 draft head weights
"""

import time
from vllm import LLM, SamplingParams

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DRAFT_MODEL  = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"   # EAGLE3 draft head
NUM_SPEC_TOKENS = 5    # candidate tokens proposed per draft step
MAX_NEW_TOKENS  = 256

SAMPLING_PARAMS = SamplingParams(
    temperature=0.0,       # greedy — maximises acceptance rate
    max_tokens=MAX_NEW_TOKENS,
)

# ── Reasoning prompts (complex → long output → bigger speedup) ────────────────
PROMPTS = [
    "Explain the difference between tensor parallelism and pipeline parallelism "
    "for training large language models. Give a concrete example for each.",

    "Walk through the math behind the Adam optimizer: why does it use first and "
    "second moment estimates, and what problem does each solve?",

    "A train leaves city A at 60 mph. Another leaves city B (300 miles away) "
    "toward city A at 40 mph. They leave at the same time. "
    "Where and when do they meet? Show all steps.",

    "Write a Python function that finds all prime numbers up to N using the "
    "Sieve of Eratosthenes. Explain the time and space complexity.",

    "Compare the KV-cache memory layout in vLLM (paged attention with fixed "
    "block sizes) versus SGLang (radix tree). What are the trade-offs?",
]


# ── Benchmark helper ──────────────────────────────────────────────────────────
def run_benchmark(llm: LLM, label: str) -> float:
    print(f"\n{'='*60}")
    print(f" {label}")
    print(f"{'='*60}")

    # Warmup with first prompt
    llm.generate([PROMPTS[0]], SAMPLING_PARAMS)

    total_tokens = 0
    t_start = time.perf_counter()
    outputs = llm.generate(PROMPTS, SAMPLING_PARAMS)
    t_end   = time.perf_counter()

    elapsed = t_end - t_start
    for i, out in enumerate(outputs):
        n_tok = len(out.outputs[0].token_ids)
        total_tokens += n_tok
        snippet = out.outputs[0].text[:120].replace("\n", " ")
        print(f"\n[Prompt {i+1}]  {n_tok} tokens")
        print(f"  {snippet}...")

    tps = total_tokens / elapsed
    print(f"\n  Total tokens : {total_tokens}")
    print(f"  Wall time    : {elapsed:.2f}s")
    print(f"  Throughput   : {tps:.1f} tokens/s")
    return tps


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Standard autoregressive decoding
    print("Loading standard model...")
    llm_standard = LLM(
        model=TARGET_MODEL,
        dtype="bfloat16",
        tensor_parallel_size=1,
    )
    tps_standard = run_benchmark(llm_standard, "Standard Autoregressive Decoding")
    del llm_standard

    # EAGLE3 speculative decoding
    print("\nLoading EAGLE3 speculative model...")
    llm_eagle3 = LLM(
        model=TARGET_MODEL,
        dtype="bfloat16",
        tensor_parallel_size=1,
        speculative_model=DRAFT_MODEL,
        num_speculative_tokens=NUM_SPEC_TOKENS,
        speculative_draft_tensor_parallel_size=1,
    )
    tps_eagle3 = run_benchmark(llm_eagle3, f"EAGLE3 Speculative Decoding  (draft_tokens={NUM_SPEC_TOKENS})")
    del llm_eagle3

    # ── Results ───────────────────────────────────────────────────────────────
    speedup = tps_eagle3 / tps_standard
    print(f"\n{'='*60}")
    print(f" RESULTS")
    print(f"{'='*60}")
    print(f"  Standard   : {tps_standard:.1f} tokens/s")
    print(f"  EAGLE3     : {tps_eagle3:.1f} tokens/s")
    print(f"  Speedup    : {speedup:.2f}x")
    print()
    if speedup > 2.0:
        print("  Great result! EAGLE3 is providing significant acceleration.")
    else:
        print("  Tip: try increasing num_speculative_tokens (5-8) or use greedy decoding (temp=0).")
