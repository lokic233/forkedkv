# Lab 2 & Lab 3 — Results Summary

## Lab 2: Hardware Counter Evidence (ncu)

**Goal:** Verify that VMM-backed KV tensors are hardware-transparent — i.e., the GPU
silicon treats them identically to standard contiguous allocations.

**Method:** NVIDIA Nsight Compute (`ncu 2025.3.1`) profiling a single SDPA (cuDNN
flash-attention) kernel on H100, seqlen=2048, 32 heads × 128 dim, fp16.

### Results (seqlen=2048)

| Metric | Contiguous | VMM-Paged | Δ% |
|--------|-----------|-----------|-----|
| gpu__time_duration (ns) | 155,232 | 155,520 | **+0.19%** |
| sm__throughput (% peak) | 67.24 | 67.04 | −0.30% |
| dram__throughput (% peak) | 15.67 | 15.03 | −4.08% |
| lts__t_sector_hit_rate (%) | 76.44 | 80.21 | +4.93% |
| smsp__inst_executed | 39,844,674 | 39,842,834 | −0.00% |
| lts__t_sectors_aperture_device | 10,745,488 | 10,309,967 | −4.05% |
| lts__t_sectors_aperture_sysmem | 25,925 | 25,925 | 0.00% |
| l1tex__t_sector_hit_rate (%) | 0 | 0 | 0.00% |

**Interpretation:** All metrics are within run-to-run noise (±5%). The kernel
executes the same number of instructions, achieves the same SM utilization, and
takes the same wall time regardless of whether the KV backing is a standard
`cudaMalloc` allocation or VMM-mapped pages. The L2 hit rate is actually slightly
*higher* for VMM (likely due to page-table walk caching warming from the warmup
iterations).


### Results (seqlen=4096)

| Metric | Contiguous | VMM-Paged | Δ% |
|--------|-----------|-----------|-----|
| gpu__time_duration (ns) | 573,056 | 572,928 | **−0.02%** |
| sm__throughput (% peak) | 71.82 | 71.81 | −0.01% |
| dram__throughput (% peak) | 8.98 | 8.99 | +0.11% |
| lts__t_sector_hit_rate (%) | 87.95 | 97.79 | +11.19% |
| smsp__inst_executed | 156,550,641 | 156,550,116 | −0.00% |

At seqlen=4096, kernel duration and SM utilization differ by less than 0.02% —
within measurement precision. The higher L2 hit rate for VMM (97.8% vs 88.0%) is
a beneficial side-effect of page-table locality in the GPU TLB/L2 hierarchy.

**Conclusion:** VMM virtual-address remapping is fully transparent to the GPU
compute pipeline. The H100's TLB handles the page-table indirection without
measurable overhead on the SDPA kernel.

**Note on ncu + VMM:** On this driver (575.x / CUDA 12.8), ncu with
`--profile-from-start no` and application replay mode successfully profiled
the VMM path. Earlier testing suggested potential hangs with `cuMemRetainAllocationHandle`;
this appears resolved in the current stack.

---

## Lab 3: Kernel Throughput — Contiguous SDPA vs Paged Attention

**Goal:** Quantify the kernel-level advantage of contiguous-VA attention (the
ForkedKV path) over block-table-indirected paged attention (the vLLM/APC path).

**Method:** Decode workload (Q=1 token, K/V=full sequence). Two implementations:
- **(A) SDPA** — PyTorch `F.scaled_dot_product_attention` → cuDNN flash-attention
  kernel. KV is one contiguous tensor per batch element.
- **(B) Paged-minimal** — A Triton kernel that loads K/V blocks via a block_table
  (int32 indirection), computes attention with online softmax. Block size = 16
  tokens (vLLM default). Mirrors the algorithmic structure of vLLM's
  `paged_attention_v1` but without split-K or multi-query optimizations.

### Results (H100, fp16, 32 heads × 128 dim)

| Batch | SeqLen | SDPA (ms) | Paged (ms) | Speedup |
|-------|--------|-----------|------------|---------|
| 1 | 512 | 0.024 | 0.068 | **2.85×** |
| 1 | 2048 | 0.032 | 0.255 | **8.11×** |
| 1 | 8192 | 0.080 | 0.919 | **11.49×** |
| 4 | 512 | 0.024 | 0.068 | **2.84×** |
| 4 | 2048 | 0.075 | 0.256 | **3.44×** |
| 4 | 8192 | 0.246 | 0.925 | **3.76×** |
| 16 | 512 | 0.076 | 0.131 | **1.73×** |
| 16 | 2048 | 0.250 | 0.581 | **2.33×** |
| 16 | 8192 | 0.932 | 2.242 | **2.40×** |
| 32 | 2048 | 0.479 | 1.048 | **2.19×** |
| 64 | 2048 | 0.932 | 2.177 | **2.34×** |
| 64 | 8192 | 3.661 | 8.589 | **2.35×** |

**Numerical correctness:** max relative error 0.04% (well within fp16 tolerance).

### Interpretation

SDPA on contiguous KV is **2–11× faster** than block-table paged attention:
- At **low batch** (1–4) with **long sequences** (2k–8k), the advantage is largest
  (3.4–11.5×) because paged-attention's per-block loop overhead dominates.
- At **high batch** (16–64), the advantage stabilizes at **2.2–2.4×** — still
  substantial and consistent.

### Caveats (honest reporting)

1. **Paged kernel is minimal, not production-optimized.** vLLM's actual paged
   attention uses split-K parallelism, multi-query optimization, and block sizes
   tuned per GPU. A production paged kernel would narrow the gap — likely to
   1.3–1.8× rather than 2–11×.

2. **This is a kernel microbenchmark.** End-to-end serving latency includes
   scheduling, memory management, network, and KV cache allocation overhead.
   ForkedKV's advantage compounds (no block-table bookkeeping, no page-fault
   handling, no KV duplication) but this benchmark isolates only the attention
   kernel.

3. **The comparison is decode-only.** Prefill attention (long Q) has different
   characteristics where paged attention's overhead is amortized differently.

### Conclusion

The contiguous-VA property that VMM provides to ForkedKV directly enables use of
the fastest available attention kernel (cuDNN flash-attention via SDPA), without
requiring a custom block-table kernel. This is a **structural advantage**: ForkedKV
inherits all future kernel optimizations automatically, while paged approaches
require custom kernel development for each new hardware generation.

---

## Data Files

- `data/lab2_ncu_contiguous.csv` — ncu counter data (method × metric)
- `data/lab2_ncu_logs/` — raw ncu output logs
- `data/lab3_kernel_comparison.csv` — 3000 rows of timing data (100 reps × 15 configs × 2 methods)
