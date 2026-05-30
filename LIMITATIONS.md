# Limitations — what we did NOT measure / what is mocked (v0.4, R3)

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

3. **Metric 5 (macro) fills KV synthetically; Metric 5b runs REAL MULTI-LAYER decode at FULL depth (R3).**
   Metric 5 sizes prefixes from real SWE-bench text (R2: 24 instances spanning the full size
   distribution) and runs the real memory mechanism but fills KV synthetically (no forward
   pass). Metric 5b runs a REAL autoregressive decode loop with REAL Qwen2.5-7B transformer
   blocks (attention + SwiGLU MLP + residuals), each layer's KV on CoW pages. **R3 (P0-1)
   extends it from the R2 first-4-layers subset to the FULL 28-layer model** (8 branches,
   all bit-identical CoW vs clone). CAVEAT: tokens are produced under a deterministic
   repetition penalty to break the truncated/greedy attractor (so they are non-degenerate AND
   bit-identical CoW-vs-clone). Our per-token decode loop is an **unoptimized Python reference
   loop** (~39 tok/s at N=28, ~205 at N=4) — we report SYSTEMS quantities (peak HBM, bytes
   copied) + bit-identical correctness, NOT serving-grade throughput or generation quality. We
   did NOT run a production inference engine and did NOT evaluate text quality.

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

7. **Single H100. Metric 3 uses one attention op; Metric 5b runs the FULL 28 layers (R3).**
   Metric 3 uses one attention op (32 heads, head_dim 128, fp16). Metric 5b (R3 P0-1) runs the
   full 28 real Qwen2.5-7B blocks (loaded across all 4 safetensors shards), bit-identical CoW
   vs clone — no longer a layer subset. Multi-GPU not attempted.

8. **Metric 4 / 4b: concurrency ceiling forensically attributed AND modeled (R2 P0-2 + R3 P0-2).**
   Full-clone OOMs at 6 (data-driven, 94.5 GiB). CoW OOMs at the exact call `cuMemSetAccess`
   (instrumented via `vmm_pool.CudaCallError`), with live data HBM flat at one prefix — so the
   ceiling is VA/mapping metadata, NOT data (measured, not inferred). **R3 (Metric 4b) sweeps
   prefix size and fits the ceiling: max_branches ≈ 520,000 / prefix_pages** (product B × P
   constant to within 1% across 1/3/6/12 GiB → 1021/339/169/84 branches). A VA free-list
   recycles freed branches' VA (120 fork→destroy cycles: reserved=10, reused=119, live HBM
   flat) so SERIAL throughput is unbounded; but pooling does NOT raise the CONCURRENT count
   (live branches free nothing) and we do not claim it does. NOT claimed: that the ceiling is
   data-memory bound, or that pooling raises concurrency.

9. **Fork latency is NOT flat (Metric 1).** CoW fork latency grows ~LINEARLY with prefix
   pages (per-page cuMemMap + cuMemSetAccess). We do NOT claim flat. CoW is ~1.3× faster than
   full-clone on average; the map-op floor is real and reported.

10. **Wall-time is ~equal to full-clone (Metrics 5 and 5b), not faster.**
    At the tested sizes CoW takes roughly the same wall time as full-clone (Metric 5:
    0.4–3.0× — parity, noise; Metric 5b: 221 vs 187 tok/s, CoW slightly faster only because
    clone pays an up-front deep copy). CoW's win is MEMORY/CAPACITY, not latency.

12. **Metric 5b validated at FULL 28-layer depth (R3 P0-1).** R2 ran 4 layers; R3 runs the
    full 28-layer Qwen2.5-7B (8 branches, all bit-identical CoW vs clone, 448 real CoW events
    at full depth). It proves CoW pages support real multi-layer attention+MLP compute at full
    model depth, NOT serving-grade throughput or generation quality (unoptimized Python decode
    loop; ~39 tok/s at N=28). Generation quality and full-model tok/s in a real engine are NOT
    measured.

14. **Partial-page CoW waste: 54% of copied bytes on the boundary page (R3 P0-3).** Our CoW
    unit is one 2 MiB VMM page (2,048 tokens for GQA-4). When a child overwrites a *partially-
    filled* shared tail page (e.g. a 3,000-token prefix → 952 of 2,048 tokens valid), CoW
    copies the whole 2 MiB even though only ~46% is real data — so **54% of the bytes copied on
    that page are waste** (137 of 256 MiB at N=4; 480 of 896 MiB at N=28; `wasted_bytes` /
    `waste_pct_of_copied` columns in `data/metric5b_decode*.csv`). We report this transparently.
    It does NOT erase the win — even counting all waste, CoW copies only 50% of full-clone's
    total byte traffic (clone copies the whole multi-page prefix per branch; CoW touches only
    the one overwritten tail page). Fully-filled interior prefix pages stay aliased and are
    never copied. A sub-page block-table granularity (v0.4 path) would shrink this but at the
    cost of contiguous-VA kernel transparency (Metric 3); see the vLLM-APC comparison.

15. **CUDA Graphs: CoW remaps must NOT occur inside a captured/replaying graph (design note,
    R3 P1-1).** Production serving engines (vLLM, TensorRT-LLM) capture the decode step into a
    CUDA Graph for launch-overhead amortization. Our CoW path issues driver calls
    (`cuMemUnmap` / `cuMemMap` / `cuMemSetAccess`) that mutate the **page-table mapping** of a
    captured VA. CUDA Graphs capture a fixed sequence of *kernel/memcpy* nodes against a fixed
    address space; **changing the VA→physical mapping under a graph is not a graph-legal
    operation** and would either be ignored at replay or corrupt the captured node's memory
    view. The correct integration (not built here) is to perform all fork/CoW remaps at a
    **branch boundary OUTSIDE graph capture** — i.e. fork → remap → (re)capture/replay the
    decode graph against the now-stable mapping. Because a fork is an explicit causal boundary
    in our model (not a per-token event), this composes naturally: the steady-state per-token
    decode inside the graph never remaps; only the (rare, explicit) branch point does, between
    graph replays. We did NOT integrate with a CUDA-Graph-capturing engine; this is a documented
    constraint for the deployment path, not a measured result.

13. **vLLM APC comparison: analytic in R2; EMPIRICAL software-equivalent baseline added in
    R4 P0-1.** R2 compared us vs vLLM Automatic Prefix Caching on design grounds (the
    "Comparison vs vLLM APC" §). R4 P0-1 added a head-to-head implementation
    (`src/baseline_prefix_sharing.py`, `bench/bench_software_baseline.py`) — a vLLM-APC-
    style block-table allocator with refcounted blocks. Result: software is ~700× faster
    on fork latency, ~6× larger on capacity at 32-page prefix, and tied on CoW
    granularity at our default sizing (and finer in production). We do NOT win on those
    axes against the strong baseline. The ONE remaining ForkedKV advantage is
    kernel-transparent contiguous VA (Metric 3 ≈ 0% overhead) — software prefix sharing
    requires a paged-attention kernel; we do not. We have NOT benchmarked against a live
    vLLM (only against our own vLLM-APC-equivalent simulation in `baseline_prefix_sharing.py`).

16. **The original "OS-style CoW" / "page-fault-on-write" framing is RETRACTED (R4 P0-3).**
    Earlier drafts called this "OS-style CoW over the GPU MMU" and labelled the write
    detection a "software page fault." Both phrasings overstate the analogy: CUDA does
    not expose hardware write-protect faults to user-mode programs, so detection is a
    software refcount check, NOT a hardware fault. The correct framing is
    "software-mediated CoW with driver-level physical-page remap." See WRITEUP §"Honest
    framing — what is and isn't OS-like." Source comments in `vmm_pool.py` and
    `kv_branch_manager.py` were updated to match.

17. **Append-only decode acknowledgment (R4 P0-4).** Standard autoregressive batched
    decode is dominated by appends to the KV tail; the only page where CoW fires under
    pure append is the boundary page. Mid-prefix CoW writes need a workload that
    *mutates* existing KV — speculative-decoding rollback, tree-of-thought branch-and-
    edit (Metric 5c), tool-call retries, sliding-window/context-compression eviction,
    or backtracking reasoning. We claim the CoW-on-write capability is right for those
    workloads (and forward-looking for the rest), NOT that it is essential for vanilla
    chat-completion serving — for that regime, vLLM APC's read-mostly prefix dedup is
    sufficient. See WRITEUP §"Workload justification — when does CoW-on-write actually
    fire?".


18. **Lab 2 hardware-counter coverage is partial (ncu + CUDA 12.8 limitations).**
    Lab 2 profiles one production cuDNN sm90 flash-attention SDPA kernel at `seqlen=4096` and
    shows kernel time, SM throughput, DRAM throughput, and instruction count are effectively
    identical for contiguous vs VMM-backed contiguous-VA tensors. However, CUDA 12.8 / driver
    580 exposes no direct TLB metric (`ncu --query-metrics | grep -i tlb` is empty), so the
    claim is supported through downstream pipeline counters rather than a direct TLB-walk
    counter. ncu replay also hangs on the VMM case for sub-200 µs kernels (`seqlen=2048` in
    this harness), so the committed CSV has clean VMM data for `seqlen=4096` only. Finally,
    only the cuDNN sm90 flash-attention kernel dispatched by PyTorch SDPA was profiled; other
    kernels / architectures remain future work.

## Reproducibility caveats

11. **N replications vary by metric.** M1: n=10; M3: n=50 (interleaved+median); B5/B8: n=300;
    M2/M2b/M4/M5: n=1 batch (deterministic byte accounting — bytes are exact, not sampled);
    M5b/5c: deterministic (bit-identical asserted). Timing in M1/M5 is wall-clock
    `perf_counter`; M3 uses CUDA events; B5/B8 use perf_counter bracketed by device sync
    (B7: R1 docstring wrongly said "CUDA events" — corrected).
