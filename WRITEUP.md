# Forkable GPU Memory for Replayable Agent Execution — v0.1 Writeup

**Hardware:** 1× NVIDIA H100 (97 GiB HBM), CUDA 12.8, driver 580.82, Python 3.12,
torch 2.11.0+cu128, cuda-python (bindings 12.9.4). All numbers from `cli:devgpu014`,
device 0. Every claim cites a data file under `data/` and is reproducible via a script
under `bench/` (see README.md). Raw run log: `experiment_log.jsonl`.

## TL;DR (honest)

We built branch-aware copy-on-write of attention KV-cache pages on the **GPU MMU** using
the CUDA VMM driver API (cuMemCreate / cuMemMap / cuMemUnmap / cuMemRetainAllocationHandle).
Forking an agent branch aliases the parent's physical HBM pages (refcounted, zero copy);
a write triggers a per-page CoW remap. **The win is memory/capacity, not latency.**

- **Capacity (Metric 4):** with a 12 GiB shared prefix, naive full-clone **OOMs at 6
  branches** on one H100; CoW **reaches 64 branches (our cap, did not OOM)**, holding
  ~12 GiB live vs ~93.9 GiB. [`data/metric4_capacity.csv`]
- **Bytes written (Metric 2b):** at realistic 5–10% per-branch divergence, CoW writes
  **93.8% / 90.6% fewer** KV bytes than full-clone; degrades to 0% at 100% divergence
  (correctly no better than clone). [`data/metric2b_divergence.csv`]
- **Attention overhead (Metric 3):** VMM-paged KV adds **−0.4% to +7.7%** vs contiguous
  KV across seqlen 512–8192 — under the 10% target. [`data/metric3_attn_overhead.csv`]
- **Fork latency (Metric 1):** **NOT flat.** CoW fork latency grows ~linearly with
  prefix length (per-page cuMemMap cost); it is ~1.32× faster than full-clone on average
  but does not eliminate the per-page cost. [`data/metric1_fork_latency.csv`]
- **End-to-end (Metric 5, 7 real SWE-bench-Verified instances):** ~90% fewer KV bytes
  written, ~80% lower peak HBM, but **wall-time speedup 0.86–1.10× (≈ parity)**. CoW does
  not beat full-clone on latency at these sizes. [`data/metric5_e2e.csv`]

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
64 MiB prefix, no divergence. CoW writes **0** KV bytes per fork at all fanouts 1–32;
full-clone writes prefix_bytes × fanout (e.g. 2,048 MiB at fanout 16). Live HBM: CoW
stays 64 MiB; full-clone grows to 2,112 MiB at fanout 32.
→ **100% bytes-written reduction in the zero-divergence case.**
Figure: `figures/metric2_bytes_written.png`.

### Metric 2b — Reduction vs divergence (realistic)  [`data/metric2b_divergence.csv`]
32-page prefix, fanout 16. CoW bytes-written reduction vs full-clone:
| divergence | reduction |
|-----------|-----------|
| 0%        | 100.0%    |
| 5%        | **93.8%** |
| 10%       | **90.6%** |
| 25%       | 75.0%     |
| 50%       | 50.0%     |
| 100%      | 0.0%      |
→ Hits the >90% target at ≤10% per-branch divergence (the realistic regime for agent
branches sharing a long prefix). Figure: `figures/metric2b_divergence.png`.

### Metric 3 — Attention kernel overhead from VMM indirection  [`data/metric3_attn_overhead.csv`]
fp16, 32 heads, head_dim 128, n=30, CUDA-event timed SDPA. KV stored in VMM-mapped pages
(wrapped zero-copy as a torch tensor via `__cuda_array_interface__`) vs a contiguous
torch tensor:
| seqlen | overhead |
|--------|----------|
| 512    | −0.4%    |
| 1024   | +1.8%    |
| 2048   | +0.2%    |
| 4096   | +0.9%    |
| 8192   | +7.7%    |
→ **−0.4% to +7.7%**, under the 10% target. Because our VA range is contiguous, the
kernel sees ordinary memory; indirection is in the page table. Figure:
`figures/metric3_attn_overhead.png`. (microbenchmark)

### Metric 4 — Capacity on one H100  [`data/metric4_capacity.csv`]
12 GiB shared prefix (6,144 × 2 MiB pages). Run in separate processes to isolate HBM.
- **full-clone: 6 branches, then CUDA_ERROR_OUT_OF_MEMORY** (live ~93.9 GiB).
- **CoW: 64 branches (our MAX_BRANCHES cap), did NOT OOM** (live ~12.0 GiB).
→ CoW reaches **≥64** branches where full-clone **OOMs at 6**. (We cannot claim the CoW
ceiling; we hit our own cap, not hardware — LIMITATIONS #8.) Figure:
`figures/metric4_capacity.png`. **This is the headline result.**

### Metric 5 — End-to-end on 7 real SWE-bench-Verified instances  [`data/metric5_e2e.csv`]
Prefix sized from real instance text (sys prompt + 6k-token repo context + problem
statement, 128 KiB KV/token for a 32-layer GQA-8 fp16 7B-class model), fanout 8, 10%
divergence. Across all 7 instances:
- KV bytes-written reduction: **89.9%–90.1%**
- Peak HBM reduction: **79.9%–80.1%**
- Wall-time speedup: **0.86×–1.10× (≈ parity)**

**Honest finding:** CoW does **not** beat full-clone on wall time end-to-end at these
prefix sizes (430–570 pages); per-page map cost ≈ D2D copy cost. The win is the ~80%
lower HBM footprint, which is what enables Metric 4's 6→64 capacity jump. Figure:
`figures/metric5_e2e.png`. NOT a full token-generation run (LIMITATIONS #3).

---

## What this buys you (and what it doesn't)

**Buys:** ~10× more concurrent agent branches per H100 at realistic divergence, by
sharing the long common prefix's HBM at the MMU level with transparent per-page CoW and
near-zero attention-kernel overhead.

**Does not buy:** lower fork latency (linear in prefix, map-op bound) or end-to-end
wall-time speedup vs full-clone at the sizes tested. If you are latency-bound and HBM is
not the constraint, full-clone is just as fast.
