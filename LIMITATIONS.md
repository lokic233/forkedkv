# Limitations — what we did NOT measure / what is mocked (v0.2, R1)

Read this before trusting any number in WRITEUP.md.

## Mechanism caveats

1. **"Page-fault-on-write" is software-enforced, not a hardware GPU #PF.**
   The CUDA VMM user API does not expose a fault handler for read-only device mappings
   (no SIGSEGV-on-GPU-write path at user level). Our CoW is triggered by the KV manager
   checking `refcount > 1` before a write and performing the copy+remap itself
   (`KVBranchManager._cow`). The *remap* is real GPU MMU work (cuMemUnmap + cuMemMap to a
   new physical handle). What is NOT hardware is the fault *detection*. A production
   system would need either (a) the manager to mediate all writes (as here), or
   (b) NVIDIA to expose write-protect faults. Documented in `prototype_status.md` D4.

2. **CoW copy uses a temporary VA window + cuMemcpyDtoD — NOW MEASURED (R1, B5).**
   `_cow` reserves a 1-page scratch VA, maps the new physical handle, D2D-copies, then
   remaps. `bench/bench_cow_overhead.py` (n=300) measures: full CoW ~175.7 us median, of
   which the unavoidable D2D copy is only 12.8 us (7%) and the removable scratch-VA
   bookkeeping is 82.2 us (47%). **CoW is map-op bound, not copy bound; ~47% of it is
   removable** with an in-place remap (reusable scratch VA). Still NOT optimized in R1 —
   we measured and documented it rather than claiming the optimization.

## Workload caveats

3. **Metric 5 (macro-benchmark) still fills KV synthetically; Metric 5b (R1) adds REAL
   single-layer token generation.** Metric 5 sizes prefixes from real SWE-bench text and
   runs the real memory mechanism but fills KV synthetically (no forward pass) — it is now
   honestly named "Macro-benchmark" (P0-B). The NEW Metric 5b (P0-A,
   `bench/bench_metric5b_decode.py`) runs a REAL autoregressive decode loop with ONE
   Qwen2.5-7B transformer layer over CoW-backed KV. CAVEAT for 5b: only ONE of 28 layers
   (proves CoW pages support real attention compute; NOT full-model throughput); single-layer
   greedy decode is a degenerate fixed-point token sequence (no LM signal) — we report
   systems quantities (tok/s, HBM, bytes copied, bit-identical correctness), NOT text
   quality. We did NOT run the full 28-layer model, did NOT measure full-model tok/s, and
   did NOT evaluate generation quality.

4. **We did NOT run the SWE-bench-Verified test harness or solve any instances.**
   We use 7 of 500 instances purely to derive realistic prefix sizes. No patches were
   generated or evaluated. No pass@k. This is a systems prototype, not an agent eval.

5. **RNG and TOOL domains in the cross-domain demo (Priority 3) are host-simulated.**
   Only the KV domain numbers are GPU-measured. RNG (32 B) and TOOL (4 KB) state sizes
   are representative constants, copied in host memory. Clearly labeled `host-simulated`
   in `data/crossdomain_granularity.csv`.

6. **Divergence model (R1 P0-D: now TAIL).** Metric 2b now writes the LAST
   `divergence_frac * prefix_pages` pages (the tail), matching how real agent branches
   diverge. B3 (R1): the prefix is now 40 pages so 5/10/25/50% are exact-integer page
   counts (v0.1's 32-page prefix made "5%" actually 6.25%); the CSV carries an
   `effective_divergence_pct` column. Metric 5 (macro) still uses head writes for the
   synthetic fill (byte count identical). Metric 5b (real decode) appends at the genuine
   tail.

## Scale / coverage caveats

7. **Single H100, single "layer-equivalent" KV shape.** Metric 3 uses one attention op
   (32 heads, head_dim 128, fp16); Metric 5b uses one real Qwen2.5-7B layer (28 q / 4 KV
   heads). Neither is a full multi-layer model. Multi-GPU not attempted. R1 B2: Metric 3
   overhead at 8192 is now +0.05% (was +7.7%) after interleaved+median timing.

8. **Metric 4 swept to TRUE OOM (R1 P1-A).** The 64 cap is removed. CoW now genuinely
   OOMs at **84 branches** (CUDA_ERROR_OUT_OF_MEMORY) with a 12 GiB prefix — but **live
   data HBM stays flat at 12.0 GiB** (84 branches use only 12 of 97 GiB). So the CoW
   ceiling is **VA-reservation / page-table-mapping metadata, NOT data memory** (each fork
   maps 6,144 VA pages -> ~516K live mappings at 84 branches). We report the measured 84 as
   the current-implementation ceiling; pooling/reusing VA (not done in R1) would raise it.
   Full-clone genuinely OOMs at 6. NOT claimed: that 84 is the data-memory ceiling — it is
   the mapping-metadata ceiling.

9. **Fork latency is NOT flat (Metric 1).** CoW fork latency grows ~LINEARLY with prefix
   pages (per-page cuMemMap + cuMemSetAccess). We do NOT claim flat. R1's B5 decomposition
   confirms WHY: CoW/fork is map-op bound (D2D copy is only 7% of a CoW), so avoiding the
   byte copy buys little when the map ops dominate. CoW is ~1.32x faster than full-clone on
   average; the map-op floor is real and reported.

10. **Wall-time is ~equal to full-clone (Metrics 5 and 5b), not faster.**
    At the tested sizes CoW takes roughly the same wall time as full-clone (Metric 5:
    0.86-1.10x; Metric 5b real decode: 681 vs 680 tok/s). CoW's win is MEMORY/CAPACITY,
    not latency. We do not claim a latency win.

12. **Metric 5b uses ONE transformer layer.** It proves CoW pages support real attention
    compute with real weights (bit-identical to clone), not full-model behavior. Greedy
    single-layer decode produces a degenerate token sequence; we report only systems
    quantities. Full 28-layer decode, generation quality, and full-model tok/s are NOT
    measured.

## Reproducibility caveats

11. **N replications vary by metric.** M1: n=10; M3: n=30; M2/M2b/M4/M5: n=1 batch
    (deterministic byte accounting — bytes are exact, not sampled). Timing in M1/M5 is
    wall-clock `perf_counter`; M3 uses CUDA events.
