# gpu-workshop-2026

**Title: Optimizing LLM Training and Inference on GPUs**

---

## Getting Started

### Requirements

You will need a **Blackwell GPU** — DGX Spark, RTX 6000, or B200.

### 1. Pull the PyTorch Container

```bash
docker pull nvcr.io/nvidia/pytorch:25.11-py3
```

### 2. Launch the Container

```bash
docker run --gpus all --rm -it \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd):/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:25.11-py3 bash
```

> `--ipc=host` and the `--ulimit` flags are **required** on DGX Spark / GB10 — without them
> the shared-memory kernel used by Transformer Engine's amax computation will fail.
>
> The container ships with PyTorch 2.10, CUDA 13, Transformer Engine 2.9, and all
> NVFP4 / FP8 support pre-installed. No extra `pip install` needed for Exercises 1 and 2.

### 3. Install vLLM (Exercise 3 only)

```bash
pip install vllm
```

---

## Exercises

### Exercise 1 — Tensor Core Benchmark (NVFP4 vs FP16 matmul)

**File:** `exercise1_tensor_core.py`

```bash
python exercise1_tensor_core.py
```

Benchmarks FP32 → FP16 → BF16 → FP8 → **NVFP4** matrix multiplication on Tensor Cores.
Each lower-precision step roughly doubles TFLOP/s on Blackwell hardware.
Watch the speedup column grow as precision drops.

---

### Exercise 2 — Mixed-Precision Training (NVFP4 vs FP16)

**File:** `exercise2_mixed_precision.py`

```bash
python exercise2_mixed_precision.py
```

Starts with a naive FP32 Transformer block that is slow and memory-hungry.
Your task is to add the `te.fp8_autocast` context manager (marked `TODO` in the file)
and observe memory and throughput change:

| Precision   | Expected speedup | Memory reduction |
|-------------|-----------------|-----------------|
| FP16 AMP    | ~2x             | ~1.5x           |
| FP8 (TE)    | ~4–6x           | ~2x             |
| NVFP4 (TE)  | ~10–15x         | ~4x             |

The `TETransformerBlock` uses NVIDIA Transformer Engine's `te.TransformerLayer` which
automatically handles the master-weight FP32 copy and gradient scaling under the hood.

---

### Exercise 3 — Inference: Standard vs EAGLE3 Speculative Decoding

**File:** `exercise3_speculative_decoding.py`

```bash
python exercise3_speculative_decoding.py
```

Runs 5 complex reasoning prompts through vLLM in two modes:

1. **Standard autoregressive decoding** — one token per forward pass.
2. **EAGLE3 speculative decoding** — a small draft head proposes `N` tokens; the
   target model verifies all of them in a single forward pass, accepting any that match.

Prints tokens/second side-by-side and the final speedup ratio.
Expected result on a B200: **3–4x** throughput improvement with `num_speculative_tokens=5`.

> Default models: `meta-llama/Meta-Llama-3.1-8B-Instruct` (target) +
> `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` (draft).
> Edit `TARGET_MODEL` / `DRAFT_MODEL` at the top of the file to use a different checkpoint.

---

## Workshop Content

### Part 1: A Deep Dive into GPUs

- **Evolution of AI** — AlexNet (2012) → general-purpose transformers (ChatGPT) → test-time
  scaling with reasoning models, multi-modal and physical-based AI.

- **Model Scaling & Hardware** — how models keep getting bigger; CPU vs. GPU architecture;
  the energy overhead of an instruction (30 pJ) vs. a Fused Multiply-Add (1.5 pJ).

- **GPU Architecture** — transition from CUDA cores to Tensor Cores; hardware evolution:
  Kepler (GTX 580) → Volta (Tensor Core) → Ampere (BF16) → Hopper (FP8) → Blackwell (FP4).
  Lower precision yields higher FLOP/s and less memory.

### Part 2: Mixed-Precision Training

- **The Challenge** — training directly with low precision (e.g. FP16) causes small gradients
  to round to zero, destabilising training.

- **The Solution** — mixed-precision training keeps a "master" weight copy in FP32 while
  computing forward/backward passes in low precision.

- **Implementation** — Transformer Engine (`te_low_precision_autocast`) automatically scales
  precision (FP8 or NVFP4) for 10–15x speedups with minimal code changes.

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
