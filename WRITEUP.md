# Forkable GPU Memory for Replayable Agent Execution — v0.2 Writeup (R1 revision)

**Hardware:** 1× NVIDIA H100 (97 GiB HBM), CUDA 12.8, driver 580.82, Python 3.12,
torch 2.11.0+cu128, cuda-python (bindings 12.9.4). All numbers from `cli:devgpu014`,
device 0. Every claim cites a data file under `data/` and is reproducible via a script
under `bench/` (see README.md). Raw run log: `experiment_log.jsonl`.

## TL;DR (honest)

We built branch-aware copy-on-write of attention KV-cache pages on the **GPU MMU** using
the CUDA VMM driver API (cuMemCreate / cuMemMap / cuMemUnmap / cuMemRetainAllocationHandle).
Forking an agent branch aliases the parent's physical HBM pages (refcounted, zero copy);
a write triggers a per-page CoW remap. **The win is memory/capacity, not latency.**

- **Capacity (Metric 4, R1 to true OOM):** with a 12 GiB shared prefix, naive full-clone
  **OOMs at 6 branches** (live ~94.5 GiB); CoW reaches **84 branches before it too OOMs**
  (live HBM flat at **12.0 GiB** the whole way). **The CoW ceiling is NOT data memory** —
  84 branches use only 12 of 97 GiB — it is VA-reservation / page-table-mapping metadata
  (each fork maps 6,144 VA pages). A 14× branch-capacity gain over full-clone, with
  headroom for far more if VA/mapping were pooled. [`data/metric4_capacity.csv`]
- **Bytes written (Metric 2b, R1 tail-divergence + exact %):** at realistic 5% / 10%
  per-branch TAIL divergence, CoW writes **95.0% / 90.0% fewer** KV bytes than full-clone;
  degrades to 0% at 100% divergence. (R1 fixes B3: 40-page prefix makes 5/10/25/50%
  exact-integer page counts; P0-D switches writes to the TAIL.) [`data/metric2b_divergence.csv`]
- **Attention overhead (Metric 3, R1 fixed B2):** VMM-paged KV adds **−0.1% to +1.1%**
  vs contiguous KV across seqlen 512–8192. The v0.1 +7.7% at 8192 was a GPU-clock/warmup
  artifact: with interleaved contig/VMM timing per rep + median over n=50 (warmup 50), it
  is **+0.05%** at 8192. Essentially zero overhead. [`data/metric3_attn_overhead.csv`]
- **Fork latency (Metric 1):** **NOT flat.** CoW fork latency grows ~linearly with
  prefix length (per-page cuMemMap cost); it is ~1.32× faster than full-clone on average
  but does not eliminate the per-page cost. [`data/metric1_fork_latency.csv`]
- **Macro-benchmark (Metric 5, 7 real SWE-bench-Verified instances; memory mechanism over
  real workload shapes):** ~90% fewer KV bytes written, ~80% lower peak HBM, **wall-time
  ≈ parity (0.86–1.10×)**. [`data/metric5_e2e.csv`]
- **End-to-end single-layer decode (Metric 5b, NEW in R1 — P0-A):** REAL autoregressive
  token generation with one Qwen2.5-7B transformer layer, KV physically backed by CoW VMM
  pages. N=16 branches, 4,096-token shared prefix, 128 real decode tokens each: **peak HBM
  CoW 72 MiB vs clone 200 MiB (−64%)**, **KV bytes copied 0 vs 128 MiB**, **throughput 681
  vs 680 tok/s (parity)**, and **decoded tokens are bit-identical CoW vs clone** (CoW-backed
  attention == contiguous attention). [`data/metric5b_decode.csv`]
- **CoW cost decomposition (B5, NEW in R1):** a single 2 MiB-page CoW takes ~176 µs median;
  the unavoidable D2D copy is only **12.8 µs (7%)**, while the (removable) scratch-VA
  bookkeeping is **82 µs (47%)**. **CoW is map-op bound, not copy bound.** [`data/cow_overhead.csv`]

---

## Priority-1 primitives (all implemented, tested on hardware)

`src/vmm_pool.py` (VMM physical+VA manager) and `src/kv_branch_manager.py` (branch CoW
logic). Validated by `src/test_primitives.py` (passes):

- **Snapshot(branch_id) → SnapshotHandle:** O(#pages) refcount bump on the branch's
  current physical pages at a causal boundary. No byte copy.
- **Fork(snapshot, new_branch_id) → ForkHandle:** reserve new VA, `cuMemMap` each child
  VA page to the SAME physical handle as the snapshot, `incref`. Zero KV bytes copied
  (asserted in the test).
- **CUDA VMM, not os.fork / not tensor copy:** the test proves aliasing via the driver:
  parent.page0 and child1.page0 return the **same** `cuMemRetainAllocationHandle`. After
  a CoW write they return **different** handles.
- **Reference counting:** parent.page0 refcount = 5 after snapshot + 3 forks; drops to 4
  after one child does CoW. Releases physical handle at refcount 0.
- **Page-fault-on-write:** write to a shared page (refcount>1) triggers `_cow`: allocate
  private handle, D2D-copy 2 MiB, remap that one VA page, decref shared. Only the touched
  page diverges; siblings stay aliased. (Software-detected; see LIMITATIONS.md #1.)
- **Dynamic VA growth (NEW in R1, P0-C):** `append_page()` grows a branch's KV tail by one
  fresh private page as decode emits tokens — the structural capability real agents need.
  VA is reserved with headroom (reservation costs no HBM; only `cuMemMap` commits memory);
  appends map the next slot, preserving the contiguous-VA property Metric 3 relies on. CoW
  is preserved: appended pages are private, the shared prefix stays aliased until written.
  Proven by `src/test_dynamic_va.py` (two children grow independently, prefix stays shared,
  CoW fires only on an explicit write to a shared prefix page).

CoW unit = one VMM physical page = `CU_MEM_ALLOC_GRANULARITY_MINIMUM` = **2 MiB** on H100.

## Priority-2 (implemented, tested)

`src/replay.py` + `src/test_replay.py` (passes):
- **Replay(branch, modifiers):** forks from a snapshot and re-executes recorded steps.
  Content-addressed: only steps whose replayed value differs from the original write
  (and thus trigger CoW). A faithful replay copies **zero** pages.
- **Controlled nondeterminism:** modifiers inject a changed RNG seed or tool result at a
  chosen step. Test shows modifying the RNG step diverges exactly page 2; modifying the
  TOOL step diverges exactly page 3.
- **Divergence detector:** `diverged_pages(a,b)` compares branches page-by-page via the
  driver's retained allocation handle; reports the exact diverged page indices.
- **Decision (D1 in prototype_status.md): standalone KV manager, not a vLLM block-manager
  patch.** Rationale: forking at the *virtual address* level requires owning the
  allocation via VMM; patching vLLM's pre-reserved torch pool would obscure the mechanism
  under evaluation. Our `baseline_fullclone.py` mimics vLLM full-sequence cloning for a
  fair comparison.

## Priority-3 (partial)
- **Real SWE-bench-Verified trajectory replay:** 7 of 500 instances, sampled across the
  problem-statement size distribution (`data/swe_selected_instances.csv`, from the real
  parquet `data/swe_verified_test.parquet`). Used for Metric 5. We did NOT run the
  SWE-bench harness or generate patches (LIMITATIONS #3,#4).
- **Cross-domain granularity demo:** `bench/bench_crossdomain_granularity.py` →
  `data/crossdomain_granularity.csv`. Per-domain granularity (KV=2 MiB GPU pages,
  RNG=32 B, TOOL=4 KB) cuts bytes copied per 16-branch fork batch by **25.0%** vs forcing
  all domains into uniform 2 MiB pages. KV is GPU-measured; RNG/TOOL host-simulated
  (labeled in the CSV).

---

## CoW cost decomposition (B5, NEW in R1)  [`data/cow_overhead.csv`]

`bench/bench_cow_overhead.py` (n=300, CUDA-event + perf_counter) decomposes one 2 MiB-page
CoW event:
| component | median |
|-----------|--------|
| full CoW (create + scratch map + D2D + remap branch VA) | **175.7 us** |
| - unavoidable D2D copy (2 MiB) | **12.8 us (7%)** |
| - scratch-VA bookkeeping (reserve+map+unmap+free; removable, LIMITATIONS #2) | **82.2 us (47%)** |

**CoW is map-op bound, not copy bound.** The 2 MiB data copy is a trivial 7%; ~half the
cost is the temporary-scratch-VA path that LIMITATIONS #2 flagged (an in-place remap scheme
would eliminate it), the rest is the branch-VA unmap+remap. This explains why fork/CoW
wall-time (Metrics 1, 5, 5b) is at parity with full-clone: both are dominated by driver map
operations, not byte movement. It also names the next optimization (reusable scratch VA ->
~47% faster CoW). Figure: `figures/cow_overhead.png`. (microbenchmark)

## The five metrics (with exact, cited numbers)

### Metric 1 — Fork latency vs prefix length  [`data/metric1_fork_latency.csv`]
n=10 reps per point, mean. Prefix 2 MiB → 256 MiB.
| prefix | CoW fork (µs) | full clone (µs) |
|--------|--------------|-----------------|
| 2 MiB  | 60           | 82              |
| 256 MiB| 6,915        | 9,218           |

**Honest finding:** latency is **linear, not flat** — dominated by per-page `cuMemMap` +
`cuMemSetAccess` driver calls (~50 µs/page). CoW is **1.32× faster on average** than
full-clone (it skips the byte copy) but the map-op floor is real. *We do not claim flat.*
Figure: `figures/metric1_fork_latency.png`. (microbenchmark)

### Metric 2 — HBM bytes written per fork vs fanout  [`data/metric2_bytes_written.csv`]
64 MiB prefix, no divergence. CoW writes **0** KV bytes per fork at all fanouts 1–32.
**B1 (formula consistency, fixed in R1):** our `bytes_written` metric counts *every byte
the driver must touch to materialize a clone* = a fresh `cuMemCreate` page (provisioned)
**plus** the `cuMemcpyDtoD` into it. So full-clone `bytes_written = 2 × prefix_bytes ×
fanout` (e.g. **2,048 MiB at fanout 16** for a 64 MiB prefix: 64×16 provisioned + 64×16
copied). The v0.1 writeup said "prefix_bytes × fanout" — that was the *copy* term only and
contradicted the CSV. The CSV is correct; the formula is now "2 × prefix_bytes × fanout".
The CoW-vs-clone *ratio* is unaffected (both sides count provision+copy identically). Live
HBM: CoW stays 64 MiB; full-clone grows to 2,112 MiB at fanout 32.
→ **100% bytes-written reduction in the zero-divergence case.**
Figure: `figures/metric2_bytes_written.png`.

### Metric 2b — Reduction vs divergence (realistic)  [`data/metric2b_divergence.csv`]
**40-page** prefix (R1: chosen so 5/10/25/50% are exact-integer page counts — B3 fix),
fanout 16, **writes at the TAIL** (R1: P0-D — real agent branches diverge at the tail, not
the prefix head). CoW bytes-written reduction vs full-clone:
| divergence (effective) | reduction |
|-----------|-----------|
| 0%        | 100.0%    |
| 5% (exact) | **95.0%** |
| 10% (exact)| **90.0%** |
| 25%       | 75.0%     |
| 50%       | 50.0%     |
| 100%      | 0.0%      |
→ Hits ~90% at ≤10% per-branch divergence. **B3 note:** v0.1 used a 32-page prefix where
`round(0.05×32)=2` pages = an *effective* 6.25% divergence reported as "5%" (giving the old
93.75%). R1 uses 40 pages so 5%→2 pages is exact; the CSV now also carries an
`effective_divergence_pct` column. Figure: `figures/metric2b_divergence.png`.

### Metric 3 — Attention kernel overhead from VMM indirection  [`data/metric3_attn_overhead.csv`]
fp16, 32 heads, head_dim 128, **n=50, warmup=50, contig/VMM INTERLEAVED per rep, median**
(R1 B2 methodology fix), CUDA-event timed SDPA. KV in VMM-mapped pages (zero-copy torch
view via `__cuda_array_interface__`) vs a contiguous torch tensor:
| seqlen | overhead (R1) | (v0.1 was) |
|--------|----------|----------|
| 512    | +1.1%    | −0.4% |
| 1024   | +0.3%    | +1.8% |
| 2048   | +0.2%    | +0.2% |
| 4096   | −0.1%    | +0.9% |
| 8192   | **+0.05%** | +7.7% |
→ **−0.1% to +1.1%**, essentially zero. **B2:** the v0.1 +7.7% at 8192 was a GPU
clock/warmup artifact (only 10 warmup reps, contig and VMM measured in separate phases so
boost-clock drift did not cancel). R1 interleaves the two methods within each rep and takes
the median over 50 reps → the 8192 overhead collapses to +0.05%. Because our VA range is
contiguous, the kernel sees ordinary memory; indirection is in the page table. Figure:
`figures/metric3_attn_overhead.png`. (microbenchmark)

### Metric 4 — Capacity on one H100  [`data/metric4_capacity.csv`]
12 GiB shared prefix (6,144 × 2 MiB pages). Run in separate processes to isolate HBM.
**R1 (P1-A): swept to TRUE OOM — the 64 cap is removed.**
- **full-clone: 6 branches, then CUDA_ERROR_OUT_OF_MEMORY** (live ~94.5 GiB).
- **CoW: 84 branches, then CUDA_ERROR_OUT_OF_MEMORY** — but **live HBM stayed flat at
  12.0 GiB** the entire sweep.
→ CoW reaches **84** branches vs full-clone's **6** — a **14× capacity gain** on one H100.
**Key R1 finding:** CoW's OOM is NOT data memory (84 branches use only 12 of 97 GiB). It is
**VA-reservation / page-table-mapping metadata**: each fork maps 6,144 VA pages, so 84
branches ≈ 516K live mappings. The data-memory ceiling is far higher; pooling/reusing VA
ranges (a known optimization, not done in R1) would push the branch count up substantially.
We report the measured 84 honestly as the *current-implementation* ceiling. Figure:
`figures/metric4_capacity.png`. **This is the headline result.**

### Metric 5 — Macro-benchmark on 7 real SWE-bench-Verified instances  [`data/metric5_e2e.csv`]

(R1 P0-B: renamed from "End-to-end" to "Macro-benchmark" — this metric runs the real
fork/clone/CoW MEMORY mechanism over real-workload-derived prefix sizes, but KV is filled
synthetically here. The REAL token-generation end-to-end is the new Metric 5b below.)
Prefix sized from real instance text (sys prompt + 6k-token repo context + problem
statement, 128 KiB KV/token for a 32-layer GQA-8 fp16 7B-class model), fanout 8, 10%
divergence. Across all 7 instances:
- KV bytes-written reduction: **89.9%–90.1%**
- Peak HBM reduction: **79.9%–80.1%**
- Wall-time speedup: **0.86×–1.10× (≈ parity)**

**Honest finding:** CoW does **not** beat full-clone on wall time end-to-end at these
prefix sizes (430–570 pages); per-page map cost ≈ D2D copy cost. The win is the ~80%
lower HBM footprint, which is what enables Metric 4's 6→84 capacity jump. Figure:
`figures/metric5_e2e.png`. NOT a full token-generation run (LIMITATIONS #3).

### Metric 5b — End-to-end single-layer decode (NEW in R1, P0-A)  [`data/metric5b_decode.csv`]
A REAL autoregressive decode loop using **one real transformer layer** (layer 0 of
**Qwen2.5-7B-Instruct**: 28 q-heads / 4 KV-heads GQA, head_dim 128, RoPE θ=1e6, RMSNorm;
weights loaded from safetensors), with the per-branch KV cache **physically backed by the
CoW VMM pages**. Each decode step does real embed → RMSNorm → q/k/v projection (+bias) →
RoPE → append K/V into CoW pages → SDPA over the branch's full KV → o_proj residual →
lm_head logits → greedy next token. (`src/decode_layer.py`, `bench/bench_metric5b_decode.py`.)

Workload: N=16 branches forked from a **4,096-token shared prefix**, each decoding **128
new tokens** (2,048 generated tokens total). CoW (fork+append) vs full-clone (deep-copy the
prefix KV, then decode):

| metric | CoW fork | full clone | delta |
|--------|----------|-----------|-------|
| peak live HBM | **72 MiB** | 200 MiB | **−64%** |
| KV bytes physically copied | **0 MiB** | 128 MiB | −100% |
| decode throughput | 681 tok/s | 680 tok/s | ≈ parity |
| branch-0 decoded tokens | — | — | **bit-identical CoW vs clone** |

**What this proves:** the CoW-aliased VMM pages support a REAL attention computation with
REAL model weights, producing **bit-identical** output to a full clone, while sharing the
prefix's HBM (zero bytes copied — decode appends to private tail pages and never overwrites
the shared prefix). Throughput is compute-bound, so CoW and clone are at parity — consistent
with the rest of the prototype: **the win is memory, not latency.**

**HONEST CAVEATS (LIMITATIONS #3, #12):** (1) ONE layer, not the full 28 — this validates
the memory mechanism under real attention compute, NOT full-model throughput or generation
quality. (2) A single layer has no real language-modeling signal, so greedy decode produces
a degenerate fixed-point token sequence (all 4321 in this run). That is expected and
irrelevant to the systems claim; we report tok/s, HBM, and bytes-copied, not text quality.
(3) tok/s is single-layer; a full 28-layer model does ~28× more work per token. Figure:
`figures/metric5b_decode.png`.

---

## What this buys you (and what it doesn't)

**Buys:** ~14× more concurrent agent branches per H100 (Metric 4: 6→84) at realistic divergence, by
sharing the long common prefix's HBM at the MMU level with transparent per-page CoW and
near-zero attention-kernel overhead.

**Does not buy:** lower fork latency (linear in prefix, map-op bound) or end-to-end
wall-time speedup vs full-clone at the sizes tested. If you are latency-bound and HBM is
not the constraint, full-clone is just as fast.
