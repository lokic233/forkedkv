# Limitations — what we did NOT measure / what is mocked (v0.1)

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

2. **CoW copy uses a temporary VA window + cuMemcpyDtoD.**
   `_cow` reserves a 1-page scratch VA, maps the new physical handle, D2D-copies, then
   remaps. This adds 2 extra map ops per CoW vs an ideal in-place scheme. Not optimized.

## Workload caveats

3. **No real 7B model token generation in the end-to-end (Metric 5).**
   Metric 5 sizes KV prefixes from REAL SWE-bench-Verified instance text and runs the
   REAL fork/clone/CoW memory mechanism on the H100, but it does NOT run a Qwen/Llama
   forward pass to generate tokens. KV pages are filled synthetically. So Metric 5
   measures the *memory mechanism* end-to-end over real workload shapes, NOT
   end-to-end agent task accuracy or token throughput. The attention-kernel cost is
   measured separately and honestly in Metric 3.

4. **We did NOT run the SWE-bench-Verified test harness or solve any instances.**
   We use 7 of 500 instances purely to derive realistic prefix sizes. No patches were
   generated or evaluated. No pass@k. This is a systems prototype, not an agent eval.

5. **RNG and TOOL domains in the cross-domain demo (Priority 3) are host-simulated.**
   Only the KV domain numbers are GPU-measured. RNG (32 B) and TOOL (4 KB) state sizes
   are representative constants, copied in host memory. Clearly labeled `host-simulated`
   in `data/crossdomain_granularity.csv`.

6. **Divergence model is uniform.** In Metrics 2b/5 each branch writes the FIRST
   `divergence_frac * prefix_pages` pages. Real agent branches diverge in their *tail*,
   not the prefix head; the byte-count is identical but the page indices differ. This
   does not affect bytes-written or capacity numbers.

## Scale / coverage caveats

7. **Single H100, single "layer-equivalent" KV shape.** Metric 3 uses one attention op
   (32 heads, head_dim 128, fp16), not a full 32-layer model. Multi-GPU not attempted.

8. **Metric 4 CoW capacity hit our self-imposed cap (64), not a hardware OOM.**
   We stopped at MAX_BRANCHES=64; CoW did NOT OOM. The true CoW ceiling is higher
   (it only grew HBM by the divergent pages). We report "reaches 64 (cap), did not OOM"
   — NOT "infinite". Full-clone genuinely OOMed at 6.

9. **Fork latency is NOT flat (Metric 1).** Target was "flat vs prefix length." Measured:
   CoW fork latency grows ~LINEARLY with prefix pages (dominated by per-page cuMemMap +
   cuMemSetAccess driver calls). We do NOT claim flat. We claim CoW is ~24% faster than
   full-clone and avoids the byte copy; the map-op cost is real and reported.

10. **End-to-end wall-time is ~equal to full-clone (Metric 5), not faster.**
    At the tested prefix sizes (430-570 pages) CoW fork+CoW-writes take roughly the same
    wall time as full-clone (speedup 0.9-1.1x). CoW's win is MEMORY/CAPACITY, not
    latency. We state this plainly and do not claim a latency win end-to-end.

## Reproducibility caveats

11. **N replications vary by metric.** M1: n=10; M3: n=30; M2/M2b/M4/M5: n=1 batch
    (deterministic byte accounting — bytes are exact, not sampled). Timing in M1/M5 is
    wall-clock `perf_counter`; M3 uses CUDA events.
