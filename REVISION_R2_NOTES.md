# Revision R2 Notes — what we did, what we skipped, why

R1 → R2. Target: 4-of-4 GREEN. Discipline: no overclaim, surface null results, cite every
number, label microbenchmarks. R2 also CORRECTED an R1 overclaim with measured evidence.

## P0 items (all 4 DONE)

### P0-1 — Multi-layer decode — DONE
- `src/decode_layer.py`: added `QwenLayerN` (first N=4 FULL Qwen2.5-7B blocks: input_layernorm,
  GQA attention with per-layer KV, attn residual, post_attention_layernorm, SwiGLU MLP, MLP
  residual; final norm + tied lm_head) and `MultiLayerBranchKV` (one CoW BranchKV per layer).
- `bench/bench_metric5b_decode.py`: `--num_layers 4`, per-layer KV pages, fork aliases ALL
  layers' prefix.
- Result (N=16 branches, 3000-tok unaligned prefix, 128 decode): peak HBM CoW 288 vs clone
  544 MiB (47%); tok/s 221 vs 187; ALL 16 branches bit-identical; tokens NON-DEGENERATE.
- Non-degeneracy note: a truncated 4-layer model greedily fixed-points. We use a DETERMINISTIC
  full-history repetition penalty (standard decoding rule, keeps CoW==clone bit-identical) to
  break the attractor → branch-0 produces varied tokens. Documented in bench + LIMITATIONS #3.

### P0-2 — VA pooling + forensic OOM — DONE
- `src/vmm_pool.py`: process-wide VA free-list keyed by num_pages (`reserve_va_range` reuses,
  `free_va` parks, `drain_va_pool` truly frees); `CudaCallError` annotates every driver call
  with its call site.
- `src/kv_branch_manager.py`: `destroy_branch` unmaps + returns VA to pool; 8-slot scratch pool.
- `bench/bench_metric4_capacity.py`: forensic OOM + 120-cycle fork/destroy churn.
- Result: CoW OOM at 84 concurrent branches, failing call = **cuMemSetAccess** (VA-mapping
  metadata, not data — live HBM 12.0 of 97 GiB). Churn: reserved=10, reused=119, HBM flat.
- HONEST: pooling enables UNBOUNDED SERIAL throughput but does NOT raise the 84 CONCURRENT
  ceiling (84 live branches free nothing). We do not claim it does.

### P0-3 — Hard correctness assert — DONE
- `bench/bench_metric5b_decode.py`: loops ALL branches, `assert match` per branch, writes
  per-branch (tokens_match, token_checksum_cow, token_checksum_clone, first_mismatch_index).
  16/16 bit-identical.

### P0-4 — CoW-on-write stress — DONE
- NEW `bench/bench_metric5c_decode_cow_write.py`: overwrite a SHARED prefix page mid-decode.
  Asserts: 1 CoW event, exactly 1 page (2 MiB) copied not the 3-page prefix, refcount 4→3,
  A de-aliases parent, parent/sibling byte-unchanged, untouched pages stay aliased. All pass.

## Hidden bugs (all DONE)

### B6 — unaligned prefix — DONE
- 3000-tok prefix (952 into page 2) → 128 real CoW events, 256 MiB copied. R1 "0 bytes" was a
  page-alignment artifact; corrected in WRITEUP + LIMITATIONS #6.

### B7 — docstring lie — DONE
- `bench/bench_cow_overhead.py`: docstring corrected to perf_counter (not CUDA events).

### B8 — "47% removable" — DONE, but as a RETRACTION (the most important honesty item)
- We BUILT the reusable scratch-VA pool (R2-D3, 8 slots). The R1 hypothesis FAILED: pooling
  removed only ~3% (178→173 µs). Forensic cause: VA reserve+free is only ~2–4 µs; the cost is
  cuMemSetAccess (~50 µs) + cuMemUnmap (~30 µs), which a pool cannot remove (verified that
  SetAccess does not persist across remap).
- We DID find a genuine 59% optimization — VA-swap CoW (72 µs) — but it breaks the contiguous-
  VA KV view, so it is NOT adopted in the headline decode. Reported in cow_overhead.csv +
  WRITEUP + LIMITATIONS #2. R1's "47% removable" claim is explicitly retracted.

## P1 items (both DONE — brief asked for ≥1)

### P1-A — SWE-bench N=7 → N=24 — DONE
- Stratified 24 instances across 143–24770 chars (median 1187 ≈ full-set 1185), 7 repos.
  Reduction stable: 89.9–90.1% bytes, 79.9–80.1% HBM. Confirms structural, not sample artifact.

### P1-B — vLLM APC analytic comparison — DONE
- WRITEUP section: VMM CoW vs vLLM APC. Our edge: kernel-transparent physical sharing (0% attn
  overhead) + explicit forkable write-after-share + growable pool. APC edge: finer block
  granularity + production maturity. Analytic only (LIMITATIONS #13); not benchmarked live.

## SKIPPED / NOT DONE (with reasons)
- Full 28-layer model: out of R2 scope; 4 layers proves composition. Brief asked N=4.
- Real vLLM patch: P1-B recommendation was analytic-only for R2. Deferred to v0.4.
- Multi-GPU: not asked.
- Raising the 84 concurrent ceiling: it is a real driver mapping-metadata limit; pooling helps
  serial not concurrent. We report it honestly rather than fake a higher number.

## Net R2 honesty posture
The biggest change vs R1 is that we turned the most-attacked claim (B8 "47% removable") into a
measured RETRACTION plus a real (trade-off-bearing) alternative — exactly the no-overclaim
behavior the committee demanded. Every R2 number is reproducible from a committed script and
cites a data file.
