# Lab 3b — Production Paged-Attention Baseline (FlashInfer)

**Date:** 2026-05-29 · **HW:** H100 (devgpu014) · **CUDA 12.8**
**Stack:** torch 2.11.0+cu128, flashinfer-python 0.6.12 (FlashInfer FA2/FA3 decode, split-K, tensor-core GQA)

## Why this lab exists

Lab 3 reported SDPA-contiguous is **2.3× faster** than a paged kernel. A committee
reviewer (codex) flagged this **YELLOW**:

> "cuDNN SDPA vs minimal Triton paged kernel ≠ ForkedKV vs vLLM. Need a production
> paged-attention baseline."

The reviewer was **right**. Lab 3's paged side was a hand-written *minimum-viable*
Triton kernel (correct asymptotics, no split-K / no tuned loads). It is not what
vLLM or SGLang actually run. Lab 3b replaces it with **FlashInfer** — the production
paged-attention library used by SGLang — and runs the *same* decode workload.

It also fixes a second, subtler unfairness on the SDPA side (see "Two SDPA variants").

## Setup (apples-to-apples, GQA — matches Qwen2.5-7B)

| param | value |
|---|---|
| num_qo_heads | 28 |
| num_kv_heads | 4 (GQA, group size 7) |
| head_dim | 128 |
| page_size | 16 tokens (FlashInfer-typical) |
| dtype | float16 |
| batch_sizes | 32, 64 |
| seqlens | 512, 2048, 8192 |
| workload | DECODE: Q = 1 token/seq, full KV context |
| reps | 100 timed (median), 30 warmup, CUDA-event timing |

> **Note vs Lab 3:** Lab 3 used MHA (32 heads, no GQA). Lab 3b uses the correct
> Qwen2.5-7B GQA config (28 qo / 4 kv). GQA is the realistic serving case and is
> exactly where the choice of baseline matters most.

Correctness: FlashInfer paged output vs reference SDPA on identical data →
**max_abs 0.0002, rel 0.0004** (well within fp16 tolerance). Layout verified.

## Two SDPA variants (this is the crux)

- **`sdpa_contig_dense`** — GQA expanded to dense MHA via `repeat_interleave`
  (KV blown up from 4→28 heads), then standard SDPA. This is what a *naive*
  contiguous kernel does, and is implicitly what Lab 3 measured.
- **`sdpa_contig_gqa`** — native GQA SDPA (`enable_gqa=True`); KV stays at 4 heads,
  kernel broadcasts internally. **This is the fairest "ForkedKV / kernel-transparent
  VA" representative** and the honest baseline to compare against.

## Results — median per-call latency (ms)

| B | S | SDPA dense | **SDPA-GQA** | **FlashInfer paged** | FI / SDPA-GQA | FI / SDPA-dense |
|---|---|---|---|---|---|---|
| 32 | 512  | 0.1184 | **0.0236** | **0.0360** | **1.52×** | 0.30× |
| 32 | 2048 | 0.4184 | **0.0751** | **0.0913** | **1.22×** | 0.22× |
| 32 | 8192 | 1.6095 | **0.2463** | **0.2649** | **1.08×** | 0.16× |
| 64 | 512  | 0.2188 | **0.0481** | **0.0583** | **1.21×** | 0.27× |
| 64 | 2048 | 0.8130 | **0.1320** | **0.1447** | **1.10×** | 0.18× |
| 64 | 8192 | 3.1857 | **0.4740** | **0.4992** | **1.05×** | 0.16× |

Ratio > 1 means FlashInfer is slower; < 1 means faster. CSV: `data/lab3b_flashinfer.csv`.

## Honest reading of the result

**The 2.3× claim does NOT survive a production paged baseline.** Two corrections,
each shrinking the gap:

1. **Production paged kernel ≫ minimal Triton kernel.** FlashInfer's paged decode is
   **3–6× faster** than Lab 3's minimal Triton paged kernel (compare FlashInfer
   medians here to `data/lab3_kernel_comparison.csv`: e.g. B=64 S=8192 paged_minimal
   8.59ms → FlashInfer 0.50ms). The reviewer's core objection is **validated**: the
   minimal kernel was a weak strawman.

2. **Native GQA ≫ dense-expanded SDPA.** Against dense MHA SDPA (`fi/dense` column),
   FlashInfer looks 3–6× *faster* — but that's an unfair comparison the other way:
   dense SDPA wastefully materializes 28 KV heads. Against the fair `enable_gqa=True`
   baseline, the picture inverts.

**Net, against fair baselines on both sides:**
- At realistic long context (S=2048, 8192): **FlashInfer paged is within ~5–22% of
  contiguous SDPA-GQA** (1.05×–1.22×). Essentially a wash; paging overhead is small
  and shrinks as seqlen grows (kernel work dominates the block-table indirection).
- At short context (S=512): contiguous SDPA-GQA is ~1.2–1.5× faster — paging's
  fixed per-call overhead (plan metadata, block-table loads) is proportionally larger
  when there's little KV to read.

## Implication for the paper / ForkedKV claim

- **Drop / heavily caveat the "2.3× faster" headline.** It compared cuDNN SDPA (dense)
  against a minimal Triton paged kernel — a weak-vs-weak mismatch in opposite directions.
- The defensible claim is narrower and honest: **kernel-transparent contiguous VA is
  competitive with — and modestly faster than (≈5–50%, largest at short context) — a
  production paged kernel, with no block-table machinery.** The advantage is real but
  *modest at the long contexts that dominate agent serving*, not 2.3×.
- ForkedKV's case should rest on what paging *cannot* cheaply do (zero-copy fork /
  CoW branch sharing, fragmentation, allocator simplicity), **not** on a large raw
  decode-kernel speed gap — because at production quality that gap is small.

## Labeling for the writeup

- "our minimal Triton paged baseline" → `paged_minimal` (Lab 3) — **strawman, do not cite as vLLM-equivalent.**
- "production paged baseline (FlashInfer)" → `flashinfer_paged` (Lab 3b) — **this is the SGLang-grade number.**
- "contiguous SDPA (ForkedKV path)" → use `sdpa_contig_gqa` (native GQA), **not** the dense-expanded variant.

## Reproduce

```bash
cd /home/dengcchi/branchable_replay && source .venv/bin/activate
python bench/bench_lab3b_flashinfer.py   # writes data/lab3b_flashinfer.csv
```

## Install / environment notes (gotchas)

- `pip install flashinfer-python` (v0.6.12) pulled in mismatched `nvidia-nccl/nvshmem/cudnn`
  and **downgraded torch 2.11.0+cu128 → torch 2.9.1**, which then hung on `import torch`
  (broken CUDA lib set). **Fix:** reinstall torch afterward:
  `pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128`.
  flashinfer 0.6.12 then works fine *with* torch 2.11.0+cu128 (import ~1.5s warm).
- The `flashinfer.ai/whl/cu124/torch2.6` index from the task brief is for torch 2.6/cu124;
  it does not match this env (torch 2.11/cu128). `flashinfer-python` (JIT/AOT) is the right
  package here.
- devgpu014 is a **heavily shared box** (load avg ~50, 384 CPUs). Cold `import torch`
  (912 MB libtorch_cuda.so) can take 30s–several min under contention and occasionally
  appears to hang; warming the .so into page cache and/or retrying resolves it. GPU itself
  was idle throughout.
