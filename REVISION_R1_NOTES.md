# Revision R1 — what was done, what was skipped, why

Round 1 of revisions in response to the 4-of-4 YELLOW committee review (ASPLOS accept
~15-30%). Goal: land all P0 + B1-B4, ≥1 P1, honestly. Hardware: cli:devgpu014, H100
device 0, CUDA 12.8, torch 2.11.0+cu128, Qwen2.5-7B-Instruct (open weights).

## Engineering decisions (committed in prototype_status.md "R1 decisions")
- R1-D1: Qwen2.5-7B-Instruct for the decode loop (open, fits one H100). Real config:
  28 layers, 28 q-heads, 4 KV-heads (GQA), head_dim 128, RoPE θ=1e6, RMSNorm.
- R1-D2: ONE layer (layer 0) for Metric 5b.
- R1-D3: vLLM = Option B (design sketch), not a real patch.
- R1-D4: landed P1-A (true OOM) AND P1-B (vLLM sketch). P1-C (expand SWE-bench) skipped.

## Status per item

### P0 (all 4 reviewers required) — ALL DONE
- **P0-A real decode loop: DONE.** `src/decode_layer.py` (Qwen2.5-7B layer-0: embed,
  RMSNorm, q/k/v proj+bias, RoPE, GQA SDPA, o_proj, lm_head) + `bench/bench_metric5b_decode.py`.
  Real autoregressive decode; per-branch KV physically in CoW VMM pages. N=16 branches,
  4096-tok prefix, 128 real decode tokens each. CoW vs full-clone. **Decoded tokens are
  bit-identical CoW vs clone** (correctness assertion in the bench).
- **P0-B rename: DONE.** Metric 5 → "Macro-benchmark" everywhere; new Metric 5b is the
  honestly-named "End-to-end single-layer decode". WRITEUP/LIMITATIONS/status propagated.
- **P0-C dynamic VA: DONE.** `vmm_pool.reserve_va_range()` + `KVBranchManager.append_page()`;
  branches reserve VA headroom (no HBM cost) and grow the tail by mapping the next slot.
  CoW preserved. `src/test_dynamic_va.py` passes (two children grow independently; prefix
  stays shared; CoW fires only on explicit write).
- **P0-D tail divergence: DONE.** `bench_metric2b_divergence.py` now writes the LAST n pages.

### Hidden bugs — B1-B5 ALL DONE
- **B1 bytes formula: DONE.** WRITEUP Metric 2 now states `2 × prefix_bytes × fanout`
  (provision via cuMemCreate + D2D copy), matching the CSV. Ratio unaffected.
- **B2 Metric 3 clock drift: DONE.** Warmup 10→50, contig/VMM interleaved per rep, median
  over n=50. 8192 overhead **+7.7% → +0.05%**. Confirmed it was a clock/warmup artifact.
- **B3 divergence rounding: DONE.** Prefix 32→40 pages so 5/10/25/50% are exact integers;
  CSV adds `effective_divergence_pct`. "5%" reduction 93.8%→**95.0%** (now truly 5%).
- **B4 reproducible hash: DONE.** `replay.py` Python `hash()` → `hashlib.blake2b`
  (`stable_byte()`). test_replay.py still passes, now process-stable.
- **B5 CoW scratch overhead: DONE (measured).** `bench/bench_cow_overhead.py` (n=300):
  full CoW 175.7µs; D2D copy 12.8µs (7%); scratch-VA bookkeeping 82.2µs (47%, removable).
  CoW is map-op bound. NOT optimized — measured and documented per discipline.

### P1 (3-of-4 asked)
- **P1-A true OOM: DONE.** 64 cap removed. CoW OOMs at **84 branches** (12 GiB prefix),
  live HBM flat at 12.0 GiB → ceiling is VA/mapping metadata, not data. 14× over clone's 6.
- **P1-B vLLM sketch: DONE.** ~2-page design sketch in prototype_status.md (block allocator
  → VMM pool, fork_seq Snapshot/Fork API, scheduler accounting; ~10-14 eng-day estimate;
  cites vAttention standalone path).
- **P1-C expand SWE-bench 7→50+: SKIPPED.** Time budget. Metric 5 still N=7. Lower priority
  than landing all P0/B; the prefix-size sizing is the only thing the instances feed, and
  the distribution is already spanned 292–9252 chars. Flagged for R2.

## New numbers — old vs new
| metric | v0.1 | R1 (v0.2) |
|--------|------|-----------|
| Metric 3 overhead @8192 | +7.7% | **+0.05%** (range −0.1%..+1.1%) |
| Metric 2b reduction @5% | 93.8% (eff 6.25%) | **95.0%** (exact 5%, tail) |
| Metric 2b reduction @10% | 90.6% | **90.0%** (exact, tail) |
| Metric 4 CoW ceiling | 64 (cap, no OOM) | **84 (true OOM)**; live flat 12 GiB |
| Metric 4 clone ceiling | 6 (OOM) | 6 (OOM), live 94.5 GiB |
| Metric 5b (NEW) peak HBM | — | CoW 72 vs clone 200 MiB (−64%) |
| Metric 5b (NEW) bytes copied | — | CoW 0 vs clone 128 MiB |
| Metric 5b (NEW) tok/s | — | 681 vs 680 (parity); bit-identical |
| B5 CoW cost (NEW) | — | 175.7µs; D2D 7%, scratch 47% |

## Files changed/created in R1
- NEW: `src/decode_layer.py`, `src/test_dynamic_va.py`, `bench/bench_metric5b_decode.py`,
  `bench/bench_cow_overhead.py`, `REVISION_R1_NOTES.md`, `data/metric5b_decode.csv`,
  `data/cow_overhead.csv`, `figures/metric5b_decode.png`, `figures/cow_overhead.png`.
- EDITED: `src/vmm_pool.py` (reserve_va_range), `src/kv_branch_manager.py` (append_page,
  headroom, Branch.capacity), `src/replay.py` (blake2b), `bench/bench_metric2b_divergence.py`
  (tail+B3), `bench/bench_metric3_attn_overhead.py` (B2 interleaved/median),
  `bench/bench_metric4_capacity.py` (true OOM sweep), `bench/make_figures.py` (+2 figs),
  `WRITEUP.md`, `LIMITATIONS.md`, `prototype_status.md`.
- Note: `bench/bench_metric4b_capacity_divergence.py` (the optional NEW file claude asked
  for) was NOT created — see "skipped" below.

## Skipped / not done in R1 (honest)
- **P1-C** (SWE-bench 7→50+): time. R2 target.
- **bench_metric4b_capacity_divergence.py**: the brief listed this as a NEW file claude
  asked for, but it overlaps Metric 4 (capacity) + Metric 2b (divergence); rather than a
  thin combination bench, R2 should do a proper capacity-under-divergence sweep (vary
  divergence_frac AND fanout, find OOM for each). Not done in R1.
- **CoW in-place remap optimization**: B5 measured the 47% removable overhead; the
  optimization itself (reusable scratch VA) is NOT implemented — measured only.
- **Full multi-layer decode, vLLM Option A patch, multi-GPU, SWE-bench harness**: out of
  R1 scope (documented in LIMITATIONS).

## Top concerns for R2 committee (builder's honest view)
1. Metric 5b is ONE layer. A reviewer may still want multi-layer (even 2-4 layers) decode
   to show the mechanism holds across the real KV footprint. ~1-2 days.
2. Metric 4's 84-branch OOM is a VA/mapping-metadata limit, not data. Reviewers will ask
   "so what's the real ceiling with VA pooling?" — implementing a VA-range pool/recycler and
   re-sweeping would turn 84 into a much larger, more compelling number. ~2-3 days.
3. The CoW scratch-VA optimization (B5, 47% removable) is identified but not built. Building
   it would make fork/CoW wall-time actually BEAT clone, not just match it — flipping the
   "no latency win" story. ~1-2 days.
4. P1-C still open: N=7 SWE-bench is thin.
5. vLLM is still a sketch; an actual minimal patch (even a toy) would harden deployability.
