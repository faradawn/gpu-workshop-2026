# gpu-workshop-2026

Title: Optimizing LLM Training and Inference on GPUs 

Part 1: A Deep Dive into GPUs

Evolution of AI: Traces the history from AlexNet (2012) to general-purpose transformers (ChatGPT), and test time scaling with reasoning models, as well as multi-modal and physical-based AI.


Model Scaling & Hardware: Discusses how models keep getting bigger and contrasts CPU vs. GPU architectures, noting the energy overhead of an instruction (30pJ) compared to a Fused Multiply-Add operation (1.5pJ).


GPU Architecture: Covers the transition from CUDA cores to Tensor Cores, and the evolution of hardware supporting lower precision—from Kepler (GTX 580) to Volta (Tensor Core), Ampere (BF16), Hopper (FP8), and Blackwell (FP4). This section explains how lower precision yields higher FLOPs and less memory consumption.

Part 2: Mixed-Precision Training


The Challenge: Explains that training directly with low precision (like FP16) leads to imprecise weight updates where small gradients get rounded to zero.


The Solution: Introduces mixed-precision training by keeping a "Master" weight in full precision (FP32).


Implementation: Shows how to use the Transformer Engine (te_low_precision_autocast) to automatically scale precision (like FP8 or NVFP4) to maximize speed (10-15x speed up) and minimize memory.


Memory Bottlenecks: Analyzes the memory consumption of the Adam Optimizer (momentum, variance) and proposes splitting the gradients and optimizer states.


Briefly outlines parallelism strategies for scaling models across multiple GPUs: Tensor Parallelism (TP), Pipeline Parallelism (PP), Sequence Parallelism (SP), Data Parallelism (DP), and Expert Parallelism (EP).

Part 3: Inference


KV Cache & Memory: Starts with traditional KV managers (contiguous memory) and highlights three optimization opportunities: memory layout, prefix re-use, and speculative decoding.


Caching Frameworks: * vLLM: Uses Prefix Cache with fixed block sizes (16 or 32) but suffers from no partial matches with different hashes.


SGLang: Utilizes a Radix Tree of common prefixes and node splitting for more efficient caching.


Speculative Decoding: Discusses improving token generation throughput during inference.


## Hands-On Session


Exercise 1: Tensor Core (NVFP4 vs FP16 matrix multiplication speed comparison).


Exercise 2: Training (NVFP4 training vs FP16).

Provide attendees with a naive PyTorch Transformer block that struggles with memory in FP32. Have them add the te.fp8_autocast (or NVFP4) context manager from the Transformer Engine and watch the memory footprint drop and speed increase.


Execise 3. Inference with Speculative Decoding (No-Speculative vs EAGLE 3 speculative decoding in vLLM)

Instead of MT-Bench, provide a pre-baked script that runs 3-5 complex reasoning prompts and prints the side-by-side tokens-per-second generation speed.