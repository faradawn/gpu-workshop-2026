# gpu-workshop-2026

**Title: Optimizing LLM Training and Inference on GPUs**

---

## Getting Started

Primary GPU for these exercises: **NVIDIA H100** (Hopper). The same code runs on other CUDA GPUs where the listed libraries support the dtypes (e.g. H200, L40S); FP8 Tensor Core paths require Hopper or newer.

### 1. Pull and Run the PyTorch Container
```bash
docker pull nvcr.io/nvidia/pytorch:26.01-py3

docker run --gpus all --rm -it \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd):/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:26.01-py3 bash
```

## Exercises

### Exercise 1 — Tensor Core Benchmark (FP32 → FP16 → FP8 matmul)

**File:** `exercise1_tensor_core.py`

```bash
python exercise1_tensor_core.py
```

Benchmarks **FP32 → FP16** with standard PyTorch `torch.mm`, then **FP8** matrix multiplication through Transformer Engine on Tensor Cores using an **8192×8192** matrix (large enough for FP8 Tensor Core throughput to dominate over Transformer Engine per-call overhead).

---

### Exercise 2 — Mixed-Precision Training (FP32, FP16, FP8)

**File:** `exercise2_mixed_precision.py`

```bash
python exercise2_mixed_precision.py
```

Starts with a naive FP32 Transformer block (`d_model=2048`) and runs 100 timed training steps after a 10-step warmup (to absorb Transformer Engine's one-time kernel JIT compilation). Compares **FP16 automatic mixed precision** and **FP8 via Transformer Engine** (`te.fp8_autocast` with `Float8CurrentScaling`):

| Precision   | Expected speedup (vs FP32/TF32) | Notes |
|-------------|--------------------------------|-------|
| FP16 AMP    | ~1.4x                          | PyTorch uses TF32 Tensor Cores by default for float32, so baseline is already fast |
| FP8 (TE)    | ~2.5x                          | Attention, norms, and optimizer are memory-bound and don't benefit from precision; ~2.5x is realistic for a single-layer bench |

The `TETransformerBlock` uses NVIDIA Transformer Engine's `te.TransformerLayer`, which keeps master weights in FP32 and handles scaling for stable low-precision training.

---

### Exercise 3 — Inference: Standard vs EAGLE3 Speculative Decoding

**File:** `exercise3_speculative_decoding.py`

```bash
pip install vllm

python exercise3_speculative_decoding.py
```

Runs 5 prompts (code generation, math, reasoning) through vLLM **one at a time** (batch=1) in two modes:

1. **Standard autoregressive decoding** — one token per forward pass.
2. **EAGLE3 speculative decoding** — a small draft head proposes `N` tokens; the
   target model verifies all of them in a single forward pass, accepting any that match.

Speculative decoding is a **latency** optimisation for single requests — it helps when the GPU is memory-bound (batch=1). At large batch sizes the GPU becomes compute-bound and the draft-verify overhead can actually hurt. The prompts are chosen for high draft-acceptance rates (code, math, structured reasoning). On H100 at batch=1, expect **~1.5–2x** per-request speedup.

> Default models: `Qwen/Qwen2.5-7B-Instruct` (target) +
> `ruipeterpan/Qwen2.5-7B-Instruct_EAGLE3_UltraChat` (draft).
> Edit `TARGET_MODEL` / `DRAFT_MODEL` at the top of the file to use a different checkpoint.

---

## Workshop Content

### Part 1: A Deep Dive into GPUs

- **Evolution of AI** — AlexNet (2012) → general-purpose transformers (ChatGPT) → test-time
  scaling with reasoning models, multi-modal and physical-based AI.

- **Model Scaling & Hardware** — how models keep getting bigger; CPU vs. GPU architecture;
  the energy overhead of an instruction (30 pJ) vs. a Fused Multiply-Add (1.5 pJ).

- **GPU Architecture** — transition from CUDA cores to Tensor Cores; hardware evolution:
  Kepler (GTX 580) → Volta (Tensor Core) → Ampere (BF16) → Hopper (**FP8**, H100) → Blackwell (FP4).
  Lower precision yields higher FLOP/s and less memory; **this workshop focuses on the FP32 → FP16 → FP8 ladder on Hopper.**

### Part 2: Mixed-Precision Training

- **The Challenge** — training directly with low precision (e.g. FP16) causes small gradients
  to round to zero, destabilising training.

- **The Solution** — mixed-precision training keeps a "master" weight copy in FP32 while
  computing forward/backward passes in lower precision (FP16 or FP8).

- **Implementation** — Transformer Engine (`fp8_autocast` with an FP8 recipe) automates
  much of the scaling for large speedups with minimal code changes on H100-class GPUs.

- **Memory Bottlenecks** — the Adam optimizer stores momentum and variance (2× model size).
  Gradient and optimizer-state sharding (e.g. ZeRO) addresses this.

- **Parallelism strategies** — Tensor Parallelism (TP), Pipeline Parallelism (PP),
  Sequence Parallelism (SP), Data Parallelism (DP), Expert Parallelism (EP).

### Part 3: Inference

- **KV Cache & Memory** — traditional KV managers use contiguous memory; three optimisation
  opportunities: memory layout, prefix re-use, speculative decoding.

- **vLLM** — Prefix Cache with fixed block sizes (16 or 32); no partial matches due to
  hash collisions with different prefixes.

- **SGLang** — Radix Tree of common prefixes with node splitting for more efficient caching.

- **Speculative Decoding** — improves token-generation throughput during inference by
  parallelising draft proposal and target verification.

---

## Extending to FP4 (Blackwell)

The exercises above stop at **FP8**, which is the practical sweet spot on **H100**. The same Transformer Engine APIs (`fp8_autocast` with an FP4 block-scaling recipe such as `NVFP4BlockScaling`) can extend this workflow to **FP4 on NVIDIA Blackwell** GPUs (e.g. B200, GB10) for even higher compute density and lower activation/weight memory — at the cost of stricter hardware requirements and tuning. Use that path when your cluster has Blackwell and your training stack supports FP4 recipes end-to-end.
