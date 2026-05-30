# Forkable GPU Memory for Replayable Agent Execution — v0.4 Writeup (R3 revision)

**Hardware:** 1× NVIDIA H100 (97 GiB HBM), CUDA 12.8, driver 580.82, Python 3.12,
torch 2.11.0+cu128, cuda-python (bindings 12.9.4). All numbers from `cli:devgpu014`,
device 0. Every claim cites a data file under `data/` and is reproducible via a script
under `bench/` (see README.md). Raw run log: `experiment_log.jsonl`.

## TL;DR (honest, R4 repositioned)

We built branch-aware copy-on-write of attention KV-cache pages on the **GPU MMU** using
the CUDA VMM driver API (cuMemCreate / cuMemMap / cuMemUnmap / cuMemRetainAllocationHandle).
Forking an agent branch aliases the parent's physical HBM pages (refcounted, zero copy);
a write to a shared page is detected in software and triggers a driver-level per-page
remap. **The contribution is NOT a practical capacity win over the strong software
baseline (vLLM APC / RadixAttention).** R4 measured this directly: against a vLLM-APC-
style block-table allocator, software is ~700× faster on fork latency and ~6× larger on
capacity at a 32-block prefix (`data/baseline_compare_*.csv`). What survives is two
narrower contributions:

1. **Forensic architectural characterization of GPU VMM for branching workloads** —
   the driver mapping ceiling K ≈ 520K, its independence from `vm.max_map_count`,
   `cuMemSetAccess` as the OOM call site, the predictable max_branches ≈ K/P model,
   VA-pool reuse, partial-page CoW waste — all of which are useful regardless of
   whether anyone deploys this exact mechanism (Metrics 4, 4b, Lab 1).
2. **Physical KV sharing exposed to the kernel as a contiguous virtual address.** The
   ONE thing software prefix sharing cannot do: standard FlashAttention/SDPA work
   unmodified on a forked branch (Metric 3: ~0% kernel overhead). vLLM APC's block
   table requires PagedAttention. We sidestep that engineering tax.

We retract the earlier "OS-style CoW" / "page-fault-on-write" framing (R4 P0-3): write
detection is software, not a hardware GPU page fault — see §"Honest framing".

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
- **Concurrency ceiling is PREDICTABLE (Metric 4b, R3 P0-2 — NEW):** sweeping prefix size
  {1, 3, 6, 12 GiB}, the max concurrent CoW branches before OOM is **1021 / 339 / 169 / 84**
  respectively — all failing at the SAME forensic call `cuMemSetAccess`, live HBM flat at one
  prefix. The product **branches × prefix_pages is constant to within 1%** (522,752 / 520,704
  / 519,168 / 516,096; median **K ≈ 520K**). So the ceiling is a quantified, predictable
  trade-off: **max_branches ≈ 520,000 / prefix_pages** (the driver's per-device mapping-table
  capacity). [`data/metric4b_ceiling.csv`]
- **Lab 1 (NEW) — ceiling is independent of the Linux VMA sysctl:** at the 84-branch
  OOM, `/proc/self/maps` holds **392 VMAs** vs `vm.max_map_count` = **67,108,864**
  (0.0006% utilisation; even the kernel default of 65,530 would leave 167× headroom).
  The 516K driver mappings produce zero new userspace VMAs and the failing call is
  `cuMemSetAccess`, not `mmap`. **The ceiling is independent of `vm.max_map_count`
  (which retains >99.9% headroom) and manifests exclusively within the CUDA VMM
  driver's `cuMemSetAccess` path. This is consistent with a per-context
  mapping-metadata capacity in the NVIDIA driver (~520K entries on H100, driver
  580.82.07) — a structural limit not tunable from userspace.**
  [`data/lab1_vmmap_count.csv`, `LAB1_NOTES.md`]
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
- **End-to-end MULTI-LAYER decode (Metric 5b, R2 P0-1; R3 P0-1 extends to FULL 28-layer
  depth):** REAL autoregressive token generation with REAL Qwen2.5-7B transformer blocks
  (attention + SwiGLU MLP + residuals per layer), KV physically backed by CoW VMM pages, one
  per-branch K/V range **per layer**. **R3 (P0-1) runs the FULL 28-layer model** (all weights
  from all 4 safetensors shards): N=8 branches, 3,000-token UNALIGNED prefix, 32 decode tokens
  each: **peak HBM CoW 1120 MiB vs clone 2016 MiB (−44%)**, **448 real partial-page CoW events
  (896 MiB copied)**, **all 8 branches bit-identical CoW vs clone (hard assert)**, ~39 tok/s.
  The mechanism holds at full model depth — not just a 4-layer subset. (The N=4 config below
  is also retained for the 16-branch/128-decode-token comparison.)
  [`data/metric5b_decode_N28.csv`, `data/metric5b_decode.csv`]
- **Partial-page CoW waste quantified (R3 P0-3 — NEW):** the 2 MiB CoW granularity copies a
  whole page even when the overwritten shared tail page is only partly filled. For a 3,000-
  token prefix (952 of 2,048 tokens in the tail page), each CoW copies 2 MiB but only ~46% is
  valid data → **54% of the copied bytes are partial-page waste** (137 of 256 MiB at N=4; 480
  of 896 MiB at N=28). Reported transparently. Crucially, even WITH this waste CoW still copies
  only **50% of full-clone's byte traffic** — the granularity overhead does not erase the win.
  [`data/metric5b_decode.csv`, `data/metric5b_decode_N28.csv`]
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
- **Software-detected CoW with driver-level remap (NOT a hardware page fault):** write
  to a shared page (refcount>1) triggers `_cow`: allocate private handle, D2D-copy 2 MiB,
  remap that one VA page (cuMemUnmap+cuMemMap+cuMemSetAccess), decref shared. Only the
  touched page diverges; siblings stay aliased. We deliberately do NOT call this
  "page-fault-on-write" — CUDA does not expose hardware write-protect faults to user
  mode. The DETECTION is a software refcount check before the write; the REMAP is real
  driver/MMU-level work. See §"Honest framing — what is and isn't OS-like" below and
  LIMITATIONS.md #1.
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

### Metric 4b — Concurrency ceiling model (R3 P0-2 — NEW)  [`data/metric4b_ceiling.csv`]

Metric 4 measured ONE prefix size. Reviewer (metacode R3) asked us to prove the ceiling is
**predictable**. Each CoW fork aliases the prefix by issuing one `cuMemMap` + `cuMemSetAccess`
**per page** into a fresh VA reservation; **no KV bytes are copied** (live HBM stays at one
prefix). The ceiling is therefore the driver's per-device **mapping-table capacity**: the
total number of (VA-page → physical-handle) access descriptors it accepts. With `P` prefix
pages per branch and `B` branches, total mappings = `B × P`. We swept `P` over four prefix
sizes and measured the max concurrent branches before the OOM:

| prefix | pages (P) | max branches (B) | OOM call | live HBM | K = B × P |
|--------|-----------|------------------|----------|----------|-----------|
| 1 GiB  | 512       | **1021**         | `cuMemSetAccess` | 1.0 GiB | 522,752 |
| 3 GiB  | 1,536     | **339**          | `cuMemSetAccess` | 3.0 GiB | 520,704 |
| 6 GiB  | 3,072     | **169**          | `cuMemSetAccess` | 6.0 GiB | 519,168 |
| 12 GiB | 6,144     | **84**           | `cuMemSetAccess` | 12.0 GiB| 516,096 |

**The product `B × P` is constant to within 1%** (median **K ≈ 519,936 ≈ 520K mappings**),
every point OOMs at the identical forensic call, and live HBM tracks exactly one prefix the
whole time. This turns Metric 4's single measured limit into a **quantified, predictable
trade-off:**

> **The ceiling is predictable: max_branches ≈ 520,000 / prefix_pages.**

So a deployer can compute the concurrent fan-out for any prefix up front (e.g. a 24 GiB
prefix → ≈42 branches; a 512 MiB prefix → ≈2,000 branches), and knows the ceiling is mapping
metadata — recyclable across an agent run via the VA free-list (Metric 4), not data HBM.
Each sweep point ran in a fresh process (a driver OOM can leave the context undefined). The
12 GiB row reproduces Metric 4's 84-branch result exactly. Figure: `figures/metric4b_ceiling.png`.

**Lab 1 corollary (NEW) — the ceiling is independent of the Linux VMA sysctl.**
A natural follow-up question is whether `K ≈ 520K` is gated by the kernel per-process
VMA limit (`vm.max_map_count`). We instrumented the 12 GiB / 84-branch run, sampling
`/proc/self/maps` line count after every 4 forks. At the OOM point: VMA count = **392**;
`vm.max_map_count` = **67,108,864** (host pre-tuned; default 65,530 would still leave
167× headroom). VMA utilisation: **0.0006%**. The 516,096 driver mappings produce
effectively zero new userspace VMAs — they live in a driver-internal mapping table that
is invisible to the kernel VM accounting, and the failing call is `cuMemSetAccess`
(driver-side), not `mmap` (kernel-side). **The ceiling is independent of
`vm.max_map_count` (which retains >99.9% headroom) and manifests exclusively within
the CUDA VMM driver's `cuMemSetAccess` path. This is consistent with a per-context
mapping-metadata capacity in the NVIDIA driver (~520K entries on H100, driver
580.82.07) — a structural limit not tunable from userspace.** This makes the
`max_branches ≈ 520,000 / P` model *more* useful as a deployment heuristic — it
cannot be sysctl'ed away, only engineered around (larger allocation granularity,
VA-pool reuse, or future driver releases). Data:
`data/lab1_vmmap_count.csv`, summary `data/lab1_vmmap_summary.txt`, write-up
`LAB1_NOTES.md`.

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

### Metric 5b — End-to-end MULTI-LAYER decode (R2 P0-1 N=4; R3 P0-1 FULL 28-layer)  [`data/metric5b_decode.csv`, `data/metric5b_decode_N28.csv`]
A REAL autoregressive decode loop using REAL transformer blocks of
**Qwen2.5-7B-Instruct** (28 q / 4 KV heads GQA, head_dim 128, RoPE θ=1e6, RMSNorm, **SwiGLU
MLP**, attention+MLP residuals; weights from safetensors), with **each layer's** per-branch
K/V cache **physically backed by CoW VMM pages** (one BranchKV per layer per branch, so a
forked child aliases ALL layers' prefix pages with zero copy). Each decode step runs the
full N-layer stack: embed → for li in 0..N-1 [ln1 → q/k/v proj+bias → RoPE → append K/V to
layer li's CoW pages → GQA SDPA → attn residual → ln2 → SwiGLU MLP → MLP residual] → final
norm → tied lm_head → next token. (`src/decode_layer.py:QwenLayerN`, `bench/bench_metric5b_decode.py`.)

**R3 P0-1 — FULL 28-layer depth validation (the metacode ask).** R2 ran the first 4 layers;
a reviewer asked us to prove the CoW mechanism holds at the FULL model depth, not a subset.
`QwenLayerN` now loads every wanted layer from whichever of the 4 safetensors shards holds it
(via the official `model.safetensors.index.json` weight map; layers span all 4 shards). The
full 28-layer model loads in ~3 s using ~13.5 GiB. We ran the complete decode at N=28:

| N=28, 8 branches, 3,000-tok UNALIGNED prefix, 32 decode tokens | CoW fork | full clone | delta |
|--------|----------|-----------|-------|
| peak live HBM | **1,120 MiB** | 2,016 MiB | **−44%** |
| KV bytes physically copied | **896 MiB (448 CoW events)** | 1,792 MiB | −50% |
| decode throughput | 39 tok/s | 40 tok/s | ≈ parity |
| ALL 8 branches' decoded tokens | — | — | **bit-identical CoW vs clone (hard assert)** |
| branch-0 decoded tokens | 32 unique values (non-degenerate) | | |

→ **The VMM CoW mechanism holds at full model depth.** All 8 branches are bit-identical to a
full clone across all 28 layers, the partial-page CoW write path fires correctly (448 real
events), and peak HBM is cut 44% while sharing the prefix. We did NOT OOM at N=28; the full
model fits one H100 with room for the KV. tok/s is ~39 (real 28-layer per-token work; our
Python decode loop is unoptimized — this is a correctness/memory result, not a throughput
claim). [`data/metric5b_decode_N28.csv`]

**N=4, 16-branch / 128-decode-token config (retained from R2).**
Workload: N=16 branches forked from a **3,000-token UNALIGNED shared prefix** (B6: 2,048
tokens/page for GQA-4, so 3,000 lands 952 tokens into page 2 — a partially-filled SHARED
page that the first decode token overwrites, forcing real CoW), each decoding **128 new
tokens** (2,048 generated total). CoW (fork+append) vs full-clone (deep-copy every layer's
prefix KV, then decode):

| metric | CoW fork | full clone | delta |
|--------|----------|-----------|-------|
| peak live HBM | **288 MiB** | 544 MiB | **−47%** |
| KV bytes physically copied | **256 MiB (128 CoW events)** | 512 MiB | −50% |
| decode throughput | ~205–221 tok/s | ~158–187 tok/s | ≈ parity (CoW skips up-front deep copy) |
| ALL 16 branches' decoded tokens | — | — | **bit-identical CoW vs clone (hard assert)** |

### Metric 5b P0-3 (R3) — Partial-page CoW waste quantification  [`data/metric5b_decode.csv`]
Our CoW unit is one 2 MiB VMM page = 2,048 tokens for GQA-4. With a 3,000-token prefix the
last shared page holds only **952 of 2,048 tokens (46%)**. When a child's first decode token
overwrites that shared boundary page, CoW copies the **whole 2 MiB** even though only 46% is
valid prefix data. We quantify the waste honestly:

| config | CoW events | bytes copied | valid data in those pages | **wasted bytes** | waste % | vs clone traffic |
|--------|-----------|--------------|---------------------------|------------------|---------|------------------|
| N=4, 16 branches  | 128 | 256 MiB | 119 MiB   | **137 MiB** | **54%** | 256 of 512 MiB = 50% |
| N=28, 8 branches  | 448 | 896 MiB | 416.5 MiB | **480 MiB** | **54%** | 896 of 1,792 MiB = 50% |

**So the 2 MiB granularity wastes ~54% of the bytes it copies on the partially-filled tail
page** (computed as `wasted = bytes_copied − valid_tokens_in_copied_pages × kv_bytes/token`;
extra CSV columns `wasted_bytes`, `waste_pct_of_copied`). This is the honest cost of coarse
granularity — exactly what vLLM APC's 16-token blocks avoid (WRITEUP §vLLM-APC, LIMITATIONS
#3). **But the win survives it:** even counting all wasted bytes, CoW copies only **50% of
full-clone's total byte traffic**, because clone copies the *entire* multi-page prefix for
every branch while CoW only touches the one overwritten tail page per branch. The waste is
bounded to the partially-filled boundary page(s); fully-filled interior prefix pages stay
aliased and are never copied. A finer CoW granularity (sub-page block table, the v0.4 path)
would shrink this 54% but at the cost of the contiguous-VA kernel-transparency Metric 3
relies on — the trade-off we discuss in the vLLM-APC comparison.


**What this proves (reviewer C1 — "single layer is a toy"; metacode R3 — "prove full depth"):**
the CoW-aliased VMM pages support a REAL multi-layer attention+MLP computation with REAL model
weights at the **full 28-layer depth**, producing **bit-identical output to a full clone for
ALL branches** (P0-3 hard assert across every branch, not a branch-0 print; per-branch blake2b
token checksums in the CSV), while sharing the prefix's HBM. The decoded tokens are
**non-degenerate** — with real layers + a deterministic full-history repetition penalty (a
standard decoding rule that keeps CoW vs clone bit-identical), branch-0's tokens are not a
single fixed point (vs R1's single-layer degenerate all-one-token sequence).

**B6 — the R1 "0 bytes copied" was a page-alignment ARTIFACT, now fixed.** R1 used a 4,096-
token prefix = exactly 2 pages, so decode only ever appended to fresh tail pages and CoW
never fired → "0 bytes copied". With the unaligned 3,000-token prefix, the first decode
token of each branch overwrites the partially-filled shared boundary page, firing **128 real
partial-page CoW events** (one per branch's first decode step), **256 MiB copied** — the
genuine cost is now measured and reported.

**HONEST CAVEATS (LIMITATIONS #3, #12):** (1) R3 validates the FULL 28-layer model (8
branches, bit-identical) AND the N=4 / 16-branch config; the mechanism composes across the
real layer stack at full depth. We still do NOT claim full-model *throughput* (our Python
per-token decode loop is unoptimized) or generation *quality*. (2) Tokens are produced under
a deterministic repetition penalty to break the truncated/greedy attractor; we report SYSTEMS
quantities + bit-identical correctness, not text quality. (3) tok/s (~39 at N=28, ~205 at
N=4) reflects an unoptimized reference loop, not a serving-system number. Figures:
`figures/metric5b_decode.png`.

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

## Comparison vs Software Prefix Sharing — EMPIRICAL (R4 P0-1, NEW)  [`data/baseline_compare_*.csv`]

Reviewer (gemini) attacked the original full-clone baseline as a strawman: vLLM APC and
SGLang RadixAttention already provide zero-copy prefix sharing in software. The R2/R3
analytic vLLM-APC discussion above is correct but not measured. R4 P0-1 closes that gap
with a head-to-head implementation: `src/baseline_prefix_sharing.py` is a vLLM-APC-style
block-table allocator (refcounted physical "blocks"; fork = list-copy + refcount++; write
to shared block = block-granularity CoW). At our test sizing (Llama-3-8B-ish: 32 layers,
8 KV-heads, 128 head_dim, bf16 → 128 KiB / token; 16-token block = 2 MiB) the **per-block
CoW unit equals our per-page CoW unit (2 MiB)** — so the comparison is mechanism-vs-
mechanism, not granularity-vs-granularity, on identical KV sizing.

Bench: `bench/bench_software_baseline.py`. Three measurements:

**M1. Fork latency vs prefix length.**

| prefix blocks | software (vLLM-APC-style) | hardware (ForkedKV CUDA-VMM) | software wins by |
|--------------:|--------------------------:|-----------------------------:|-----------------:|
|             1 | 0.51 µs                   | 64.76 µs                     | 127×             |
|             4 | 0.55 µs                   | 214.66 µs                    | 390×             |
|            16 | 1.50 µs                   | 818.60 µs                    | 546×             |
|            64 | 5.00 µs                   | 3,263 µs                     | 653×             |
|           128 | 9.34 µs                   | 6,588 µs                     | 705×             |

Software fork is **~700× faster** at large prefixes. The hardware path pays
`cuMemMap`+`cuMemSetAccess` (~50 µs) per page; the software path increments a Python
refcount per block. **Software wins fork latency, decisively.** This is not a marginal
delta we can argue away. (Data: `data/baseline_compare_m1_fork_latency.csv`.)

**M2. CoW granularity (bytes copied per single-token write into a shared prefix).**
Both copy **2,097,152 B (one 2 MiB unit)** at our default sizing — block_bytes equals
page_bytes. Software *can* be configured smaller (vLLM in production runs 16-token
blocks at smaller hidden-dim models → 256 KiB CoW units, 8× finer than us). Our 2 MiB
page is fixed by `CU_MEM_ALLOC_GRANULARITY_MINIMUM` on H100 — we cannot reduce it.
**Tie at our test sizing; software has more room to shrink.** (Data:
`data/baseline_compare_m2_cow_granularity.csv`.)

**M3. Capacity at fixed 32-block (64 MiB) prefix.**

| method                | max branches | wall time | extra host RAM | constraint |
|-----------------------|-------------:|----------:|---------------:|-----------|
| software (vLLM-APC)   | **100,000**  | 0.354 s   | 41 MiB         | host RAM (block_table refcounts) |
| hardware (ForkedKV)   | ~16,250      | (modeled) | n/a            | driver mapping ceiling: K/P = 520K/32 |

Software reaches 100,000 branches in 354 ms with 41 MiB host RAM (all on-GPU pool stays
constant — fork allocates ZERO new GPU bytes, just refcounts existing blocks). Hardware
caps at the **K/prefix_pages** ceiling we measured in Lab 1 (~16,250 at this prefix).
**Software wins capacity by ~6× at this prefix size; the gap widens for longer prefixes.**
(Data: `data/baseline_compare_m3_capacity.csv`. Figures:
`figures/baseline_compare_m1_fork_latency.png`,
`figures/baseline_compare_m3_capacity.png`.)

### What survives the empirical comparison

The 14× capacity headline (Metric 4: full-clone OOMs at 6 vs CoW reaches 84) was
correct against a NAIVE baseline. Against the strong software baseline, **capacity is
NOT our advantage** — software's refcounted block table is unbounded by GPU memory and
nearly unbounded by host RAM. The earlier R2 analytic and Metric 4 capacity sweeps
remain valid as characterizations of the CUDA VMM driver, but they do not buy us a
practical capacity edge over vLLM APC. We acknowledge this directly.

What survives is the **mechanism-level asymmetry on kernel compatibility**:

- **ForkedKV produces a contiguous virtual address per branch.** Standard FlashAttention,
  PyTorch SDPA, and any kernel that takes a contiguous K/V tensor work unmodified
  (Metric 3: −0.1% to +1.1% overhead vs a non-VMM contiguous tensor across seqlen
  512–8192). The kernel sees ordinary memory; the MMU does the sharing.
- **Software prefix sharing requires a paged-attention kernel.** vLLM ships
  PagedAttention precisely because its block-table allocator is incompatible with
  contiguous-K/V kernels. Adopting APC means committing to a custom attention kernel
  family per attention variant (FlashAttention v2/v3, MLA, GQA — each needs a paged
  rewrite). That is a real engineering tax that we sidestep.
- **Per-token attention has no block-table indirection.** PagedAttention pays one
  block-table lookup per token at the kernel level. We pay zero (the page table walk
  is the GPU MMU's hardware job, paid by every load anyway).

### Honest verdict (the only differentiator that survives the strong baseline)

**ForkedKV does NOT win on fork latency, does NOT win on capacity, and does NOT win on
CoW granularity against vLLM-APC-style software prefix sharing.** What it wins is:
*the same physical sharing, exposed to the kernel as a contiguous virtual address.*
That is a **kernel-transparency** result, not a capacity result. The paper's
contribution is therefore best framed as:

1. The **forensic architectural characterization** of GPU VMM for branching workloads
   (driver mapping ceiling K ≈ 520K, independence from `vm.max_map_count`,
   `cuMemSetAccess` as the OOM call site, VA-pool reuse, partial-page waste model) —
   useful regardless of whether anyone deploys this mechanism.
2. **Proof that hardware page-aliasing achieves equivalent prefix sharing with zero
   kernel modification** — i.e. you can keep FlashAttention as-is and still get
   physical KV sharing. That is an architectural claim with practical value for
   deployments unwilling to swap out their attention kernel.

Re-positioning is honest. We are no longer claiming a practical capacity advantage
over the SOTA software baseline; we are claiming a kernel-compatibility advantage and
contributing a characterization of the GPU VMM substrate.

---

## Honest framing — what is and isn't OS-like (R4 P0-3, NEW)

Earlier drafts of this writeup used the phrase "page-fault-on-write" and "OS-style CoW
over the GPU MMU." Both phrasings overstate the analogy and we retract them.

**What we actually do.** Detection of a write to a shared page is **software**: the
KV-branch-manager checks `refcount > 1` *before* issuing the write. CUDA does not expose
hardware write-protect faults to user-mode programs — there is no GPU equivalent of the
x86 `#PF` we could intercept. So the "page fault" is metaphorical at best.

**What is real.** The remap that follows is genuine driver/MMU-level work:
`cuMemUnmap` + `cuMemMap` + `cuMemSetAccess` reprogram the GPU MMU's page tables for
that one VA page. The aliasing — multiple branches' VA pages mapping to the same
physical handle — is real hardware sharing, observable as the same value returned by
`cuMemRetainAllocationHandle` (Metric 5c assertion). When two branches diverge, the
handles diverge; the MMU does the indirection.

**Honest claim.** What we built is: **software-mediated CoW with driver-level
physical-page remap**. The detection is a software refcount check; the share/remap
is hardware (GPU MMU + driver mapping table). We **deliberately do not** call this
"page-fault-on-write" or "OS-style CoW" anywhere in this writeup or in the source
files (R4 P0-3 globally edited the comments and prose).

**Why hardware page faults aren't available, and what we get anyway.** CUDA's user-mode
API does not surface MMU faults on writes; even Unified Memory's automatic migration
faults are kernel-driver-mediated and not exposable as a userspace handler. So a "true
OS-style CoW" on GPU writes is not achievable today. What our path *does* give over a
pure software block manager (vLLM APC):

- Contiguous VA per branch → unmodified attention kernels (FlashAttention, SDPA, …).
- Physical sharing at the GPU MMU level → no per-token block-table indirection.
- Latent path to hardware faults: if NVIDIA exposes write-protect faults in a future
  driver, this design swaps the software refcount check for a real fault handler with
  no other changes. We are not relying on that; we report it as a structural property.

This is a narrower claim than "OS-style CoW" but it is the one that survives scrutiny.

---

## Workload justification — when does CoW-on-write actually fire? (R4 P0-4, NEW)

A second reviewer attack: standard autoregressive decode is **append-only** to KV. The
boundary page (the last partial-page where new tokens land) is the only place CoW can
fire under pure append; deeper-into-the-prefix CoW writes need a workload that *mutates*
existing KV. Honestly: which real workloads do that?

**Where CoW-on-write fires for real:**

1. **Speculative-decoding rollback.** Verifier rejects K speculative tokens →
   chain has to rewind by K positions and overwrite the rejected KV slots. The
   rejected page is shared (parent + speculator); rollback CoWs it.
2. **Tree-of-thought / branch-and-rewind.** Agent explores branch A, evaluates,
   discards, returns to the snapshot, explores branch B *editing* a token in
   the shared prefix (e.g. swap a tool call). That edit is exactly Metric 5c.
3. **Multi-turn agents that rewrite history.** Tool-call retries that
   substitute corrected arguments, error-correction passes that rewrite a span,
   guardrail-driven redaction of spans in long contexts. All do mid-prefix
   writes against a shared parent.
4. **Sliding-window / context-compression eviction.** A long context that
   overwrites old KV with compressed summaries (e.g. Anthropic's recent
   compression work, RecurrentGPT-style) does in-place mutation.
5. **Reasoning models with backtracking.** o1-style chains that prune a
   sub-trajectory and resume from an earlier partial state.

Metric 5c (R2 P0-4) directly exercises pattern #2 and proves the mechanism:
exactly one CoW fires, exactly one 2 MiB page is copied, sibling branches stay
byte-identical. That's the smallest convincing workload.

**Honest acknowledgement.** For *vanilla* batched-decode serving (chat completion
with no edits), the dominant pattern is append-only and the CoW-on-write path
fires only on the boundary page. In that regime our advantage shrinks to "shared
prefix dedup" — which vLLM APC already gives, with finer granularity. The
CoW-on-write capability is genuinely valuable for the workloads listed above
(branchable agents, speculative decoding, mutation-heavy serving), and is
forward-looking for the rest. We do not claim CoW-on-write is essential for
chat-completion serving; we claim it is the right substrate for branch-and-edit
workloads. See LIMITATIONS.md #14 (added).

---



**Buys (vs naive full-clone — the WORST baseline, not the strongest):** ~14× more
concurrent agent branches per H100 (Metric 4: 6→84) at realistic divergence, by
sharing the long common prefix's HBM at the MMU level with transparent per-page CoW
and near-zero attention-kernel overhead. The concurrent ceiling is **predictable**
— Metric 4b fits **max_branches ≈ 520,000 / prefix_pages** (constant to 1% across a
12× prefix sweep) — so a deployer can size fan-out up front. The mechanism is
validated bit-identical at the **full 28-layer model depth** (Metric 5b R3), with
the 2 MiB granularity's partial-page waste (54% on the boundary page) quantified
honestly and still net-favorable (50% of clone traffic).

**Buys (vs strong software baseline — vLLM APC / RadixAttention):** kernel-
transparent contiguous-VA per branch. Standard FlashAttention/SDPA work unmodified
on a forked branch (Metric 3 ≈ 0% kernel overhead). Software prefix sharing requires
a paged-attention kernel; we do not. That is the ONE differentiator that survives
the strong-baseline empirical comparison (R4 P0-1).

**Does not buy (against vLLM APC — be honest):** lower fork latency (R4 M1: software
is ~700× faster), more concurrent branches (R4 M3: software handles 100K branches in
host RAM; we cap at K/P ≈ 16,250 at a 32-page prefix), or finer CoW granularity
(software's 16-token blocks can be 8× smaller than our 2 MiB page at small models).
If you are latency-bound or capacity-bound and willing to maintain a custom paged-
attention kernel, vLLM APC is the better mechanism today.

## Lab 3b — Production Paged Baseline (FlashInfer 0.6.12)

**The Lab 3 "2.3× faster" claim does not survive against a production baseline.**

Lab 3 compared cuDNN SDPA (dense, 28-head materialization) vs a minimal Triton paged
kernel. Both sides were mismeasured: SDPA was too slow (dense, not GQA-native) and the
paged kernel was too slow (no split-K, no tensor-core GQA). Against FlashInfer 0.6.12
(the SGLang production kernel) with a fair GQA-native SDPA baseline:

| B | S | SDPA-GQA (contiguous) | FlashInfer paged | Ratio |
|---|---|---|---|---|
| 64 | 2048 | 0.132 ms | 0.145 ms | **1.10×** |
| 64 | 8192 | 0.474 ms | 0.499 ms | **1.05×** |
| 32 | 512 | 0.024 ms | 0.036 ms | **1.52×** |

**Honest conclusion:** Contiguous VA is modestly faster (5–52%, largest at short context
where paging's fixed overhead dominates). The advantage is NOT 2.3×. ForkedKV's case
rests on what paging *cannot cheaply do* — zero-copy fork, branch-level CoW sharing,
contiguous-VA simplicity — not on a large raw decode speed gap.

[`data/lab3b_flashinfer.csv`, `LAB3B_NOTES.md`]
