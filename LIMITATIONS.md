# Limitations — what we did NOT measure / what is mocked (v0.3, R2)

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

2. **CoW cost: the R1 "47% removable" claim was WRONG and is retracted (R2 / B8).**
   R1 measured full CoW ~176 µs, of which D2D copy is 12.8 µs (7%) and "scratch-VA
   bookkeeping" 82 µs (47%), and HYPOTHESIZED the 47% was removable via a reusable scratch
   VA. **R2 built it (8-slot scratch-VA pool) and the hypothesis failed: pooling removes
   only ~3%** (178→173 µs). Forensic breakdown: cuMemAddressReserve+Free are only ~2–4 µs;
   the cost is cuMemSetAccess (~50 µs) + cuMemUnmap (~30 µs) per mapping, which pooling
   cannot remove. A genuine 59% win exists — **VA-swap CoW** (point the slot at the new
   page's VA, skip the dst remap, ~72 µs) — but it BREAKS the contiguous-VA KV view that
   Metric 3's ~0% overhead and the zero-copy torch view depend on, so we did NOT adopt it
   in the headline decode. CoW remains map-op bound (SetAccess+Unmap), not copy bound.
   `bench/bench_cow_overhead.py` (n=300). This corrects an R1 overclaim.

## Workload caveats

3. **Metric 5 (macro) fills KV synthetically; Metric 5b (R2) runs REAL MULTI-LAYER decode.**
   Metric 5 sizes prefixes from real SWE-bench text (R2: 24 instances spanning the full size
   distribution) and runs the real memory mechanism but fills KV synthetically (no forward
   pass). Metric 5b (R2 P0-1) runs a REAL autoregressive decode loop with the **first 4 full
   transformer blocks** of Qwen2.5-7B (attention + SwiGLU MLP + residuals), each layer's KV
   on CoW pages. CAVEAT for 5b: 4 of 28 layers (proves the mechanism COMPOSES across a real
   multi-layer stack; NOT full-model throughput); tokens are produced under a deterministic
   repetition penalty to break the truncated-model greedy attractor (so they are
   non-degenerate AND bit-identical CoW-vs-clone). We did NOT run the full 28-layer model,
   did NOT measure full-model tok/s, and did NOT evaluate generation quality.

4. **We did NOT run the SWE-bench-Verified test harness or solve any instances.**
   We use 24 of 500 instances purely to derive realistic prefix sizes. No patches were
   generated or evaluated. No pass@k. This is a systems prototype, not an agent eval.

5. **RNG and TOOL domains in the cross-domain demo (Priority 3) are host-simulated.**
   Only the KV domain numbers are GPU-measured. RNG (32 B) and TOOL (4 KB) state sizes
   are representative constants, copied in host memory. Clearly labeled `host-simulated`
   in `data/crossdomain_granularity.csv`.

6. **B6 (R2): Metric 5b now uses an UNALIGNED prefix — the R1 "0 bytes copied" was an
   artifact.** R1's 4,096-token prefix = exactly 2 pages (2,048 tok/page for GQA-4), so
   decode only appended to fresh tail pages and CoW never fired → "0 bytes copied" was a
   page-alignment artifact, not a property of the mechanism. R2 uses a 3,000-token prefix
   (lands 952 tokens into page 2), so the first decode token overwrites the partially-filled
   SHARED boundary page → 128 real partial-page CoW events, 256 MiB copied. Metric 2b still
   uses tail writes; Metric 5 (macro) still uses head writes for the synthetic fill (byte
   count identical).

## Scale / coverage caveats

7. **Single H100; 4 of 28 layers.** Metric 3 uses one attention op (32 heads, head_dim 128,
   fp16); Metric 5b uses the first 4 real Qwen2.5-7B blocks. Neither is a full 28-layer
   model. Multi-GPU not attempted.

8. **Metric 4: 84 concurrent CoW branches, OOM forensically attributed (R2 P0-2).** Full-
   clone OOMs at 6 (data-driven, 94.5 GiB). CoW OOMs at **84 concurrent branches at the
   exact call `cuMemSetAccess`** (instrumented via `vmm_pool.CudaCallError`), with live data
   HBM flat at 12.0 GiB — so the ceiling is VA/mapping metadata, NOT data (now measured, not
   inferred). A VA free-list recycles freed branches' VA (120 fork→destroy cycles:
   reserved=10, reused=119, live HBM flat) so SERIAL throughput is unbounded; but pooling
   does NOT raise the CONCURRENT count (84 live branches free nothing) and we do not claim it
   does. NOT claimed: that 84 is the data-memory ceiling, or that pooling raises concurrency.

9. **Fork latency is NOT flat (Metric 1).** CoW fork latency grows ~LINEARLY with prefix
   pages (per-page cuMemMap + cuMemSetAccess). We do NOT claim flat. CoW is ~1.3× faster than
   full-clone on average; the map-op floor is real and reported.

10. **Wall-time is ~equal to full-clone (Metrics 5 and 5b), not faster.**
    At the tested sizes CoW takes roughly the same wall time as full-clone (Metric 5:
    0.4–3.0× — parity, noise; Metric 5b: 221 vs 187 tok/s, CoW slightly faster only because
    clone pays an up-front deep copy). CoW's win is MEMORY/CAPACITY, not latency.

12. **Metric 5b uses 4 transformer layers.** It proves CoW pages support real multi-layer
    attention+MLP compute (bit-identical to clone for all 16 branches), not full-model
    behavior. Full 28-layer decode, generation quality, and full-model tok/s are NOT measured.

13. **vLLM APC comparison is ANALYTIC, not benchmarked (R2 P1-B).** WRITEUP compares our VMM
    CoW vs vLLM Automatic Prefix Caching / block-table sharing on design grounds (kernel-
    transparent physical sharing + explicit forkable write-after-share + growable pool, vs
    APC's finer block granularity + production maturity). We did NOT run a head-to-head
    benchmark against a live vLLM. A real vLLM patch is the v0.4+ deployment path
    (`prototype_status.md`).

## Reproducibility caveats

11. **N replications vary by metric.** M1: n=10; M3: n=50 (interleaved+median); B5/B8: n=300;
    M2/M2b/M4/M5: n=1 batch (deterministic byte accounting — bytes are exact, not sampled);
    M5b/5c: deterministic (bit-identical asserted). Timing in M1/M5 is wall-clock
    `perf_counter`; M3 uses CUDA events; B5/B8 use perf_counter bracketed by device sync
    (B7: R1 docstring wrongly said "CUDA events" — corrected).
