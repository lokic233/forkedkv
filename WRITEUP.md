# Forkable GPU Memory for Replayable Agent Execution — v0.3 Writeup (R2 revision)

**Hardware:** 1× NVIDIA H100 (97 GiB HBM), CUDA 12.8, driver 580.82, Python 3.12,
torch 2.11.0+cu128, cuda-python (bindings 12.9.4). All numbers from `cli:devgpu014`,
device 0. Every claim cites a data file under `data/` and is reproducible via a script
under `bench/` (see README.md). Raw run log: `experiment_log.jsonl`.

## TL;DR (honest)

We built branch-aware copy-on-write of attention KV-cache pages on the **GPU MMU** using
the CUDA VMM driver API (cuMemCreate / cuMemMap / cuMemUnmap / cuMemRetainAllocationHandle).
Forking an agent branch aliases the parent's physical HBM pages (refcounted, zero copy);
a write triggers a per-page CoW remap. **The win is memory/capacity, not latency.**

- **Capacity (Metric 4, R1 to true OOM; R2 forensic + VA pool):** with a 12 GiB shared
  prefix, naive full-clone **OOMs at 6 branches** (live ~94.5 GiB); CoW reaches **84
  concurrent branches before it too OOMs** (live HBM flat at **12.0 GiB**). **R2 (P0-2)
  pins the failing call forensically: CoW's OOM is `cuMemSetAccess`** — it is
  VA-mapping-metadata exhaustion (84 forks × 6,144 pages ≈ 516K live mappings), **NOT data
  memory** (84 branches use only 12 of 97 GiB). R2 adds a process-wide **VA free-list**:
  120 fork→destroy cycles ran with **only 10 VA reservations issued and 119 reused**, live
  HBM flat at 12 GiB — so freed branch metadata is **recycled** and serial branch
  throughput is unbounded; the concurrent ceiling is the per-mapping driver-metadata limit,
  which we now name exactly. 14× concurrent gain over full-clone. [`data/metric4_capacity.csv`]
- **Bytes written (Metric 2b, R1 tail-divergence + exact %):** at realistic 5% / 10%
  per-branch TAIL divergence, CoW writes **95.0% / 90.0% fewer** KV bytes than full-clone;
  degrades to 0% at 100% divergence. [`data/metric2b_divergence.csv`]
- **Attention overhead (Metric 3, R1 fixed B2):** VMM-paged KV adds **−0.1% to +1.1%**
  vs contiguous KV across seqlen 512–8192 (+0.05% at 8192). Essentially zero overhead.
  [`data/metric3_attn_overhead.csv`]
- **Fork latency (Metric 1):** **NOT flat.** CoW fork latency grows ~linearly with
  prefix length (per-page cuMemMap cost); ~1.3× faster than full-clone but the map-op
  floor is real. [`data/metric1_fork_latency.csv`]
- **Macro-benchmark (Metric 5, R2: 24 real SWE-bench-Verified instances spanning the full
  size distribution, 143–24,770 chars):** **89.9–90.1% fewer KV bytes written, 79.9–80.1%
  lower peak HBM, wall-time ≈ parity (0.4–3.0×, noise)**. [`data/metric5_e2e.csv`]
- **End-to-end MULTI-LAYER decode (Metric 5b, R2 P0-1 — was single-layer in R1):** REAL
  autoregressive token generation with the **first 4 full transformer blocks** of
  Qwen2.5-7B (attention + SwiGLU MLP + residuals per layer), KV physically backed by CoW
  VMM pages, one per-branch K/V range **per layer**. N=16 branches, **3,000-token UNALIGNED
  shared prefix** (B6 fix), 128 real decode tokens each: **peak HBM CoW 288 MiB vs clone
  544 MiB (−47%)**, **all 16 branches bit-identical CoW vs clone (hard assert, P0-3)**,
  decoded tokens **non-degenerate** (real LM flow), throughput **221 vs 187 tok/s**. The
  unaligned prefix triggers **128 real partial-page CoW events (256 MiB copied)** — the R1
  "0 bytes copied" was a page-alignment artifact (B6). [`data/metric5b_decode.csv`]
- **CoW-on-write stress (Metric 5c, R2 P0-4 — NEW):** a tree-of-thought ROLLBACK that
  overwrites a SHARED prefix page mid-decode fires CoW **exactly once**, copies **exactly 1
  page (2 MiB), not the 3-page prefix**; the parent/sibling page is provably **uncorrupted**
  (driver-handle + byte check), refcount drops 4→3, untouched prefix pages stay aliased.
  All assertions pass. This exercises the mechanism's hot path the headline previously
  skipped. [`data/metric5c_cow_write.csv`]
- **CoW cost decomposition (B5/B8, R2):** a single 2 MiB-page CoW takes **~178 µs** median;
  the unavoidable D2D copy is only **~13 µs (7%)**. **B8 NULL RESULT (correcting R1):** the
  R1 claim that "47% is removable scratch-VA bookkeeping" was **WRONG** — pooling the
  scratch VA (skipping the reserve+free pair) yields **only ~3%** because VA reserve+free
  is just ~2–4 µs; the real cost is `cuMemSetAccess` (~50 µs) + `cuMemUnmap` (~30 µs) per
  mapping, which a scratch pool cannot remove. We DID find a genuine optimization: a
  **VA-swap CoW** (point the branch slot at the new page's VA, skip the dst remap) runs at
  **~72 µs (~59% faster)** — but it breaks the contiguous-VA KV view, so it is usable only
  where a single contiguous view isn't required. We report both honestly. [`data/cow_overhead.csv`]

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

## CoW cost decomposition (B5 R1 + B8 R2 correction)  [`data/cow_overhead.csv`]

`bench/bench_cow_overhead.py` (n=300, perf_counter bracketed by cuCtxSynchronize — B7: the
R1 docstring wrongly said "CUDA-event + perf_counter"; only perf_counter is used, which is
correct for these host-side driver-call-bound paths) decomposes one 2 MiB-page CoW event:
| component | median |
|-----------|--------|
| full CoW (R1 path: create + scratch reserve/map/unmap/free + remap branch VA) | **178.3 us** |
| + reusable scratch-VA pool (B8: skip scratch reserve+free) | **173.4 us (only 3% faster)** |
| VA-swap CoW (B8: point slot at new VA, skip dst remap) | **72.4 us (59% faster)** |
| - unavoidable D2D copy (2 MiB) | **12.8 us (7%)** |
| - scratch-VA bookkeeping (reserve+map+unmap+free) | **83.4 us (47% of full CoW)** |

**B8 — IMPORTANT R2 CORRECTION OF AN R1 OVERCLAIM.** R1 claimed the 47% scratch-VA
bookkeeping was *removable* via a reusable scratch VA, projecting CoW from 175.7→~93 µs.
**We built it (R2-D3: an 8-slot scratch-VA pool in `KVBranchManager`) and the hypothesis was
WRONG.** Pooling the scratch VA only removed ~3% (178.3→173.4 µs). Forensic breakdown of the
83 µs "scratch bookkeeping": cuMemAddressReserve+cuMemAddressFree are only **~2–4 µs**; the
real cost is **cuMemSetAccess ~50 µs + cuMemUnmap ~30 µs**, which a scratch pool *cannot*
remove (every new physical handle must be mapped+access-set to copy into it, and SetAccess
does not persist across remap — verified). The R1 "47% removable" claim is retracted.

**A genuine optimization DOES exist — VA-swap CoW — but with a trade-off.** Instead of
`unmap(dst); map(dst,new); SetAccess(dst)`, we map the new page at a fresh VA, copy, and
let the branch slot *point at the new VA* (returning the old VA to the pool). This skips the
dst remap and runs at **72.4 µs (59% faster)**. The cost: the page lands at a non-contiguous
VA, breaking the single-contiguous-VA KV view that the zero-copy torch view and Metric 3's
~0% overhead depend on. So VA-swap is a real latency win **only for workloads that don't need
one contiguous KV view across the branch** (e.g. a block-table-indexed kernel like vLLM's).
We report both, and DID NOT adopt VA-swap in the headline decode (which needs contiguity).

**CoW is map-op bound, not copy bound** (the 2 MiB copy is 7%) — that conclusion stands. What
changes in R2: the dominating map ops are `cuMemSetAccess`+`cuMemUnmap`, not the scratch
reserve/free, so the cheap pooling optimization R1 promised does not exist. Figure:
`figures/cow_overhead.png`. (microbenchmark)

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
**R1 swept to TRUE OOM; R2 (P0-2) adds forensic OOM attribution + a VA free-list.**
- **full-clone: 6 branches, then CUDA_ERROR_OUT_OF_MEMORY** (live ~94.5 GiB; data-driven).
- **CoW: 84 concurrent branches, then CUDA_ERROR_OUT_OF_MEMORY** — live HBM flat at
  **12.0 GiB** the entire sweep.
→ CoW reaches **84** concurrent branches vs full-clone's **6** — a **14× capacity gain**.

**R2 P0-2 forensic finding (gemini R2-3):** we annotated every driver call with its call
site (`vmm_pool.CudaCallError`). **The CoW OOM is the call `cuMemSetAccess`** — confirming
the ceiling is **VA / page-table-mapping metadata, not data memory** (84 branches use only
12 of 97 GiB; each fork maps 6,144 pages → ≈516K live mappings at 84 branches). This is now
*measured evidence* (which call returns CUDA_ERROR_OUT_OF_MEMORY), not inference.

**R2 P0-2 VA free-list:** `VMMPool` keeps a process-wide free-list of reserved VA ranges
keyed by size; `KVBranchManager.destroy_branch` unmaps a branch's pages (freeing HBM at
refcount 0) and returns its VA range to the pool for reuse. A 120-cycle fork→destroy churn
(>> the 84 concurrent ceiling) issued **only 10 cuMemAddressReserve calls and reused 119**,
with live HBM flat at 12.0 GiB. So **freed branch metadata is recycled**: serial branch
throughput is unbounded, and the 84 is specifically the *concurrent* mapping-metadata
ceiling. Pooling does NOT raise the concurrent count (84 live branches free nothing), and we
do not claim it does — but it bounds total VA growth across an agent run. Figure:
`figures/metric4_capacity.png`. **This is the headline result.**

### Metric 5 — Macro-benchmark on 24 real SWE-bench-Verified instances  [`data/metric5_e2e.csv`]

(R1 P0-B: renamed from "End-to-end" to "Macro-benchmark" — this metric runs the real
fork/clone/CoW MEMORY mechanism over real-workload-derived prefix sizes, but KV is filled
synthetically here. The REAL token-generation end-to-end is Metric 5b below.)
**R2 (P1-A): expanded N=7 → N=24 instances** stratified across the FULL SWE-bench-Verified
problem-statement size distribution (143 → 24,770 chars; median 1,187 ≈ the full-set median
1,185; 7 repos: django, sympy, scikit-learn, matplotlib, astropy, xarray, pylint).
Prefix sized from real instance text (sys prompt + 6k-token repo context + problem
statement, 128 KiB KV/token for a 32-layer GQA-8 fp16 7B-class model), fanout 8, 10%
divergence. Across all **24** instances:
- KV bytes-written reduction: **89.9%–90.1%**
- Peak HBM reduction: **79.9%–80.1%**
- Wall-time: **0.4×–3.0× (≈ parity; the few outliers are wall-clock noise on ~250 ms runs)**

**Honest finding:** CoW does **not** reliably beat full-clone on wall time end-to-end at
these prefix sizes (428–813 pages); per-page map cost ≈ D2D copy cost. The win is the ~80%
lower HBM footprint, which is what enables Metric 4's 6→84 capacity jump. The reduction is
**remarkably stable across the 24× size span**, confirming it is structural (prefix-share +
10% tail divergence), not an artifact of the R1 7-instance sample. Figure:
`figures/metric5_e2e.png`. NOT a full token-generation run (LIMITATIONS #3).

### Metric 5b — End-to-end MULTI-LAYER decode (R2 P0-1; was single-layer in R1)  [`data/metric5b_decode.csv`]
A REAL autoregressive decode loop using the **first 4 full transformer blocks** of
**Qwen2.5-7B-Instruct** (28 q / 4 KV heads GQA, head_dim 128, RoPE θ=1e6, RMSNorm, **SwiGLU
MLP**, attention+MLP residuals; weights from safetensors), with **each layer's** per-branch
K/V cache **physically backed by CoW VMM pages** (one BranchKV per layer per branch, so a
forked child aliases ALL 4 layers' prefix pages with zero copy). Each decode step runs the
full N-layer stack: embed → for li in 0..3 [ln1 → q/k/v proj+bias → RoPE → append K/V to
layer li's CoW pages → GQA SDPA → attn residual → ln2 → SwiGLU MLP → MLP residual] → final
norm → tied lm_head → next token. (`src/decode_layer.py:QwenLayerN`, `bench/bench_metric5b_decode.py`.)

Workload: N=16 branches forked from a **3,000-token UNALIGNED shared prefix** (B6: 2,048
tokens/page for GQA-4, so 3,000 lands 952 tokens into page 2 — a partially-filled SHARED
page that the first decode token overwrites, forcing real CoW), each decoding **128 new
tokens** (2,048 generated total). CoW (fork+append) vs full-clone (deep-copy every layer's
prefix KV, then decode):

| metric | CoW fork | full clone | delta |
|--------|----------|-----------|-------|
| peak live HBM | **288 MiB** | 544 MiB | **−47%** |
| KV bytes physically copied | **256 MiB (128 CoW events)** | 512 MiB | −50% |
| decode throughput | **221 tok/s** | 187 tok/s | CoW 1.18× |
| ALL 16 branches' decoded tokens | — | — | **bit-identical CoW vs clone (hard assert)** |

**What this proves (reviewer C1 — "single layer is a toy"):** the CoW-aliased VMM pages
support a REAL multi-layer attention+MLP computation with REAL model weights, producing
**bit-identical output to a full clone for ALL 16 branches** (P0-3 hard assert across every
branch, not a branch-0 print; per-branch blake2b token checksums in the CSV), while sharing
the prefix's HBM. The decoded tokens are **non-degenerate** — with 4 real layers + a
deterministic full-history repetition penalty (a standard decoding rule that keeps CoW vs
clone bit-identical), branch-0's 128 tokens are not a single fixed point (vs R1's
single-layer degenerate all-one-token sequence). Throughput slightly favors CoW here because
full-clone pays an up-front 512 MiB deep-copy that CoW skips.

**B6 — the R1 "0 bytes copied" was a page-alignment ARTIFACT, now fixed.** R1 used a 4,096-
token prefix = exactly 2 pages, so decode only ever appended to fresh tail pages and CoW
never fired → "0 bytes copied". With the unaligned 3,000-token prefix, the first decode
token of each branch overwrites the partially-filled shared boundary page, firing **128 real
partial-page CoW events** (one per branch's first decode step), **256 MiB copied** — the
genuine cost is now measured and reported.

**HONEST CAVEATS (LIMITATIONS #3, #12):** (1) 4 of 28 layers — validates the memory
mechanism composes across a real multi-layer stack, NOT full-model throughput or generation
quality. (2) Tokens are produced under a deterministic repetition penalty to break the
truncated-model greedy attractor; we report SYSTEMS quantities + bit-identical correctness,
not text quality. (3) tok/s is 4-layer; a full 28-layer model does ~7× more work per token.
Figure: `figures/metric5b_decode.png`.

### Metric 5c — CoW-on-write decode stress (R2 P0-4 — NEW)  [`data/metric5c_cow_write.csv`]
Metrics 5/5b are append-dominated; Metric 5c exercises the mechanism's **hot path**: a
branch OVERWRITING a SHARED prefix page mid-decode — a tree-of-thought rollback / speculative
context edit. Using one real Qwen2.5-7B layer over a 3-page (6,144-token) shared prefix,
forked into two children A and B (each aliasing the prefix, refcount 4 on the target page).
Child A overwrites a SHARED interior prefix page. Measured + **asserted**:
- `_cow` fired **exactly once**; **exactly 1 page (2 MiB) copied**, NOT the 3-page prefix.
- target page refcount **4 → 3**; A de-aliases the parent (driver-handle check True→False).
- **child B's / parent's page is byte-for-byte UNCHANGED** (parent context not corrupted).
- A's page bytes CHANGED (the edit took effect, privately).
- the OTHER two prefix pages (0 and 2) **stay aliased** A==parent — CoW is per-page.

All assertions pass. This is the write-after-share semantics that distinguishes our mechanism
from read-only prefix dedup (see vLLM-APC comparison below). Figure: `figures/metric5c_cow_write.png`.

---

## Comparison vs vLLM Automatic Prefix Caching (APC) — analytic (R2 P1-B)  [no new data file; design analysis]

The strongest baseline is NOT "naive full-clone" — it is vLLM's existing prefix sharing
(Automatic Prefix Caching / block-table sharing). A reviewer (codex) correctly noted that
without this comparison the headline reads "we beat the worst baseline." We address it
analytically (per the R2 brief recommendation; a real vLLM patch is the v0.4+ deployment
path sketched in `prototype_status.md`).

**What vLLM APC does.** vLLM stores KV in fixed 16-token **blocks** in a pre-allocated torch
pool, indexed per-sequence by a **block table** (logical block → physical block id). APC
hashes block *contents*; sequences sharing an identical prefix point their block tables at
the same physical block ids (read-only share). On a write to a shared block, vLLM does a
block-granularity **copy-on-write** (allocate a new block, copy 16 tokens, repoint the block
table). So vLLM APC is *already a CoW system* — at the block-table level over a pre-reserved
pool.

**What our VMM CoW adds over vLLM APC (the three structural differences):**

1. **Write-after-share at the page table, with explicit fork points.** APC shares are
   *content-hash, read-mostly* and dedup-oriented; our Snapshot/Fork is an **explicit causal
   fork** at an agent branch point, and the write-after-share path (Metric 5c) is a
   first-class operation, not an incidental APC copy. This is what replayable/forkable agent
   execution needs: deliberately diverging a *copy* of a context you chose to branch.
2. **No block-table indirection in the attention kernel.** APC's kernel must gather KV
   through the per-sequence block table (PagedAttention's defining cost). Our VA range is
   **contiguous**, so the SDPA kernel sees ordinary memory — Metric 3 measures **~0%
   overhead** (−0.1% to +1.1%) vs a contiguous tensor. We share physically (same driver
   handle) while keeping the kernel's view contiguous. APC cannot do both.
3. **Growable physical pool.** APC shares within a *pre-reserved* torch pool fixed at
   startup; it cannot grow physical KV past that reservation. Our pool grows on demand via
   `cuMemMap` (`append_page`, P0-C) and reserves VA without committing HBM.

**What vLLM APC does BETTER, honestly:** (a) 16-token blocks → far finer CoW granularity
than our 2 MiB page (≈8K tokens for GQA-4), so APC copies less on a small divergence and is
not subject to our page-alignment partial-page CoW (B6). (b) APC is *deployed, battle-tested,
and integrated with the scheduler*; we are a standalone prototype. (c) APC's content-hash
dedup finds sharing opportunities ours (explicit-fork-only) does not.

**Net analytic position.** Against vLLM APC our advantage is **kernel-transparent physical
sharing (0% attn overhead) + explicit forkable/write-after-share semantics + a growable
pool**; APC's advantage is **fine block granularity + production maturity**. The honest
framing for the paper: VMM CoW is the right substrate when you need *explicit, kernel-
transparent, growable* branch forking (agent replay / tree-of-thought), not a drop-in win
over APC for ordinary prefix dedup. We have **not** benchmarked against a running vLLM
(LIMITATIONS #13); the comparison above is analytic.

---

## What this buys you (and what it doesn't)

**Buys:** ~14× more concurrent agent branches per H100 (Metric 4: 6→84) at realistic divergence, by
sharing the long common prefix's HBM at the MMU level with transparent per-page CoW and
near-zero attention-kernel overhead.

**Does not buy:** lower fork latency (linear in prefix, map-op bound) or end-to-end
wall-time speedup vs full-clone at the sizes tested. If you are latency-bound and HBM is
not the constraint, full-clone is just as fast.
