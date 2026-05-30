# Prototype Status

Hardware: 1× H100 97 GiB, CUDA 12.8, driver 580.82, torch 2.11.0+cu128, cuda-python.

## Engineering decisions (committed early)

- **D1: Standalone KV manager, NOT a vLLM block-manager patch.** Forking at the virtual
  address level requires owning the allocation via CUDA VMM. Patching vLLM's pre-reserved
  torch pool would obscure the mechanism. `baseline_fullclone.py` mimics naive vLLM
  full-sequence cloning for a fair comparison.
- **D2: 1× H100 (device 0).** Multi-GPU not attempted (bonus).
- **D3: CoW unit = one VMM physical page = 2 MiB (CU_MEM_ALLOC_GRANULARITY_MINIMUM).**
- **D4: Page-fault-on-write is software-detected** (manager checks refcount>1 before a
  write), with a REAL GPU MMU remap (cuMemUnmap+cuMemMap to a new handle). CUDA VMM does
  not expose user-level write-protect faults on device mappings. See LIMITATIONS.md #1.
- **Model sizing:** KV shapes use Llama-3.1-8B / Qwen2.5-7B class params (32 layers,
  GQA-8, head_dim 128, fp16). We did NOT load model weights; no token generation.

## WORKS (measured on hardware)
- Snapshot, Fork (zero-copy alias), per-page CoW, refcounting, divergence detector —
  `src/test_primitives.py` passes; aliasing proven via cuMemRetainAllocationHandle.
- Replay with controlled nondeterminism (RNG/TOOL modifiers) — `src/test_replay.py` passes.
- All 5 metrics measured + cross-domain demo. CSVs in data/, figures in figures/.
- Full-clone baseline genuinely OOMs at 6 branches (CUDA_ERROR_OUT_OF_MEMORY).

## MOCKED / SIMULATED
- Metric 5 KV pages filled synthetically (sizes from real SWE-bench text; no fwd pass).
- Cross-domain RNG (32 B) + TOOL (4 KB) state host-simulated; only KV is GPU-measured.
- Divergence writes the prefix HEAD pages (real branches diverge in the tail; byte count
  identical, page indices differ).

## BROKEN / NOT DONE
- Fork latency is NOT flat (linear in prefix; map-op bound). Target missed, reported honestly.
- End-to-end wall-time speedup is ~parity (0.86–1.10×), not a win. Reported honestly.
- No vLLM integration. No SWE-bench harness run. No multi-GPU. No real model weights.
- CoW capacity ceiling unknown (hit self-imposed cap of 64, not hardware OOM).

## R1 status update
- WORKS (added R1): real single-layer decode over CoW KV (Metric 5b, P0-A) — bit-identical
  to clone; dynamic VA `append_page()` (P0-C, test_dynamic_va.py passes); CoW cost
  decomposition (B5); Metric 4 true OOM at 84 branches (P1-A); blake2b replay hash (B4).
- FIXED (R1): Metric 3 8192 overhead 7.7%->0.05% (B2 interleaved+median); Metric 2b exact
  integer divergence + tail writes (B3, P0-D); Metric 2 bytes formula consistency (B1).
- STILL MOCKED: Metric 5 (macro) synthetic fill; cross-domain RNG/TOOL host-sim.
- STILL NOT DONE: full multi-layer model; vLLM patch (Option A — sketch only, P1-B);
  multi-GPU; SWE-bench harness run; CoW in-place remap optimization (measured, not built).

## Metric status: 6 of 6 measured (Metric 5b decode added in R1). Memory/capacity claims
## strong; latency at parity (honestly reported); attention overhead ~0%.


---

## R1 revision decisions (committed early — round 1)

The v0.1 committee returned 4-of-4 YELLOW (ASPLOS accept ~15-30%). This round implements
the P0 + B1-B5 + at least one P1 revisions. Decisions made up front:

- **R1-D1 (model for P0-A real decode loop): Qwen2.5-7B-Instruct.** Open weights (not
  gated), fits on one H100. Real config: 28 layers, 28 attn heads, 4 KV heads (GQA),
  head_dim 128, fp16. We use ONE transformer layer (layer 0) for the autoregressive
  decode loop — this proves the CoW-backed KV pages support real attention compute with
  real model weights. We do NOT run the full 28-layer stack (out of R1 scope; layer-0 is
  representative for the memory-mechanism claim).
- **R1-D2 (decode layers): ONE layer.** Per brief. The contribution under test is that
  CoW pages back a real attention computation, not multi-layer throughput.
- **R1-D3 (vLLM, P1-B): Option B — design sketch.** We write a design sketch (in this
  file) of the vLLM block-manager / scheduler changes needed, citing vAttention's
  standalone-prototype path. Option A (real vLLM patch) is deferred to v0.3.
- **R1-D4 (which P1): we land P1-B (vLLM sketch, cheapest) AND P1-A (Metric 4 true OOM).**
  P1-C (expand SWE-bench 7->50+) is attempted if time permits.
- **R1-D5 (m5_config note):** m5 capacity sizing assumes a 32-layer GQA-8 7B-class model
  (Llama-3.1-8B-class KV shape = 128 KiB/token). The P0-A decode loop uses Qwen2.5-7B
  (GQA-4) weights. These are intentionally different: m5 measures the *memory mechanism*
  over a representative model-class KV footprint; P0-A proves *real attention compute* on
  CoW pages. Both are honestly labeled; we do not conflate them.

### Note on redacted config literals
The v0.1 working tree shipped with two integer literals in `bench/m5_config.json` and
`bench/bench_metric5_e2e_trajectories.py` replaced by the token `<redacted>`. R1
reconstructed them from intact structural fields and the committed v0.1 CSVs:
`kv_bytes_per_token = n_layers*n_kv_heads*head_dim*dt_bytes*kv_factor = 131072` (128 KiB),
`chars_per_token = 4` (back-solved from `prefix_tokens` in `data/metric5_e2e.csv`:
django-14500 has ps_chars=292, prefix_tokens=6873, sys+repo=6800 => 73 prob tokens =>
292/73 = 4). Both reproduce the v0.1 CSV exactly.

---

## P1-B: vLLM integration design sketch (Option B — not implemented, R1)

The committee asked "deployability?". We did NOT patch vLLM in R1 (R1-D3: Option B). We
follow vAttention's (ASPLOS'25) precedent of validating the VMM mechanism in a standalone
prototype first, then describing the integration path. Below is the concrete design.

### Where vLLM stands today
vLLM's `PagedAttention` already allocates KV in fixed-size **blocks** (default 16 tokens)
managed by `BlockManager` / `KVCacheManager`. Blocks are indexed by a per-sequence
**block table** (logical block -> physical block id). Prefix caching ("automatic prefix
caching", APC) already lets multiple sequences SHARE read-only prefix blocks via a hash
of block contents, with copy-on-write at block granularity when a shared block is written.

**Key observation:** vLLM APC is already a CoW system, but at the *block-table* level over
a pre-allocated torch KV pool — NOT at the GPU MMU level. It shares by pointing two block
tables at the same physical block id; it cannot share at sub-block granularity and it
cannot grow the physical pool past what was pre-reserved at startup.

### What our mechanism adds
Our VMM CoW shares at the **GPU virtual-address / page-table** level (2 MiB pages), which
(a) lets the attention kernel see one contiguous VA range (no block-table indirection in
the kernel — measured ~0% overhead, Metric 3 R1), and (b) lets the physical pool GROW on
demand via `cuMemMap` instead of being pre-reserved (P0-C dynamic VA). The forking unit is
an explicit Snapshot/Fork API at causal boundaries (agent branch points), not implicit
content-hash dedup.

### Integration path (engineering estimate)
Three components change:

1. **Block allocator -> VMM-backed pool (largest change, ~5-7 days).**
   Replace vLLM's `CacheEngine` torch `kv_cache` tensors with VA ranges from a `VMMPool`
   (our `src/vmm_pool.py`). Each KV block becomes a `cuMemMap` of a 2 MiB physical handle
   at a fixed offset in the sequence's reserved VA range. vLLM block_size (16 tokens) is
   much smaller than our 2 MiB page; either (a) raise vLLM block_size so 1 block = 1 page,
   or (b) sub-allocate N vLLM blocks per VMM page (page is the CoW unit; block is the
   scheduling unit). Recommend (b): keep vLLM's 16-token block for the scheduler, make
   the CoW/sharing unit the 2 MiB page underneath.

2. **BlockManager.fork_seq() Snapshot/Fork API (~3-4 days).**
   Add `fork(parent_seq_id, child_seq_id)` to `KVCacheManager` that calls our
   `KVBranchManager.snapshot()` + `fork()` to alias the parent's VMM pages into the child's
   VA range (refcount++), instead of allocating + memcpy. The scheduler already tracks
   per-seq block tables; the change is making "copy parent blocks to child" a VA alias
   rather than a deep copy. CoW-on-write reuses our `_cow` page-fault path. Exposes
   branch/replay to the serving API (a new `/fork` endpoint).

3. **Scheduler accounting (~2-3 days).**
   vLLM's scheduler budgets free blocks to admit requests. With VMM sharing, "free HBM"
   must be computed from live *physical* pages (refcounted), not block-table entries, so
   the scheduler can admit far more concurrent branches when they share a prefix (this is
   exactly the Metric 4 6->N_cow capacity jump, surfaced to the scheduler). Plus a true-OOM
   guard on `cuMemCreate`.

### Risks / open questions for the patch
- vLLM's CUDA graph capture pins KV tensor addresses; VA ranges must be stable across
  `cuMemMap`/`cuMemUnmap` (they are — VA is reserved once; only physical backing changes),
  but graph replay over a remapped page needs validation.
- FlashAttention kernels assume contiguous KV; our VA range IS contiguous (Metric 3
  confirms ~0% overhead), so this should hold, but multi-page-table TLB pressure at very
  long contexts is unmeasured.
- Interaction with vLLM's existing APC (content-hash CoW) — likely disable APC and let the
  explicit Snapshot/Fork API own sharing, to avoid double-CoW.

### Total estimate: ~10-14 engineering days for a working vLLM fork prototype (Option A).
This is the v0.3 / camera-ready target. R1 ships the standalone mechanism + this sketch.

---

## R2 revision decisions (committed early — round 2)

R1 returned 3 strong YELLOW + 1 GREEN (ASPLOS ~35-45% median, one reviewer 65-75%).
R2 targets 4-of-4 GREEN. Decisions made up front:

- **R2-D1 (multi-layer decode, P0-1): N=4 layers, FULL transformer block per layer.**
  Per brief recommendation N=4. We run the full Qwen2.5-7B block (input_layernorm →
  attention with per-layer GQA KV → residual → post_attention_layernorm → SwiGLU MLP →
  residual) for layers 0..3, then final norm + tied lm_head. Each layer keeps its OWN
  per-branch K/V CoW pages (one BranchKV per layer). Rationale: 4 real layers give a
  non-degenerate token sequence (real residual+MLP signal), proving the mechanism
  composes and isn't a single-layer toy. We chose full-block over attention-only because
  the reviewers' C1 ("single layer is a toy") is best answered by real language-modeling
  flow, and MLP is cheap relative to the systems question.
- **R2-D2 (VA pooling, P0-2): process-wide free-list keyed by size_in_pages.**
  On branch/snapshot destroy, return the reserved VA range to a free-list in VMMPool
  keyed by num_pages. New reservations prefer reuse from the pool (cuMemAddressFree is
  deferred; the VA stays reserved and is re-handed-out). Physical handles still freed at
  refcount 0 (unchanged). Re-run Metric 4 to measure the new ceiling. Add per-call-site
  error annotation so the OOM is forensic (which exact cuMem* call fails).
- **R2-D3 (scratch-VA pool, B8): pool size 8 reusable scratch VA pages.**
  KVBranchManager pre-reserves 8 one-page scratch VA ranges at init; _cow borrows one,
  maps the new handle, D2D-copies, unmaps, returns it to the pool — eliminating the
  reserve/free pair per CoW (the 47% scratch bookkeeping from B5). 8 supports 8
  concurrent CoW ops; documented. Re-run B5 / Metric 1 / Metric 5b post-optimization.
- **R2-D4 (unaligned prefix, B6): prefix_tokens chosen NON-page-aligned.**
  R1 used 4096 tokens = exactly 2 pages (toks_per_page=4096 for GQA-4) so decode never
  overwrote a shared page → 0 bytes copied was an artifact. R2 Metric 5b uses an
  unaligned prefix so the first decode token lands in a PARTIALLY-FILLED shared prefix
  page, forcing a real partial-page CoW. We report the partial-page CoW cost.
- **R2-D5 (correctness, P0-3): hard assert across ALL branches, not branch-0 print.**
  Metric 5b asserts decoded tokens bit-identical CoW-vs-clone for every branch, writes
  per-branch checksums + first-mismatch index to CSV.
- **R2-D6 (CoW-on-write stress, P0-4): new Metric 5c.** Deliberately overwrite a shared
  prefix page mid-decode (tree-of-thought rollback / speculative edit), verify _cow
  fires (refcount 2→1), exactly one page copied, parent uncorrupted, writer diverges.
- **R2-D7 (which P1): land P1-A (SWE-bench N=7→24) AND P1-B (vLLM APC analytic).**
  P1-A: pull 17 more SWE-Bench-Verified instances spanning the real size distribution
  (median 1185 chars, max 24770) → N=24. P1-B: analytic comparison vs vLLM APC /
  block-table prefix sharing in WRITEUP (analytic-only per brief recommendation).

---

## R2 status update (round 2 complete)

- WORKS (added R2):
  - P0-1: REAL MULTI-LAYER decode (4 full Qwen2.5-7B blocks, attn+SwiGLU+residuals), each
    layer's KV on CoW pages. `decode_layer.QwenLayerN` + `MultiLayerBranchKV`. N=16 branches,
    3000-tok UNALIGNED prefix, 128 decode: peak HBM CoW 288 vs clone 544 MiB (47%), tok/s 221
    vs 187, ALL 16 branches bit-identical, non-degenerate tokens.
  - P0-2: VA free-list in vmm_pool (process-wide, keyed by size) + per-call-site OOM
    annotation (`CudaCallError`). Metric 4: CoW OOM forensically = `cuMemSetAccess`; VA
    free-list recycles (120 fork/destroy cycles, reserved=10 reused=119, live HBM flat 12GiB).
  - P0-3: HARD correctness assert across ALL branches (was branch-0 print); per-branch
    blake2b token checksums + first-mismatch index in metric5b_decode.csv.
  - P0-4: Metric 5c CoW-on-write stress — overwrite shared prefix page mid-decode; all
    assertions pass (1 CoW event, 1 page copied, refcount 4->3, parent uncorrupted).
  - B6: unaligned prefix -> 128 real CoW events / 256 MiB copied (R1 "0 bytes" artifact gone).
  - B7: bench_cow_overhead docstring fixed (perf_counter, not CUDA events).
- FIXED / RETRACTED (R2):
  - B8: R1 "47% removable scratch-VA bookkeeping" claim RETRACTED — built it, only ~3% (the
    cost is SetAccess+Unmap, not reserve/free). Real win = VA-swap CoW (59%) but it breaks
    contiguous-VA view; not adopted in headline. Honest correction in WRITEUP + LIMITATIONS.
  - P1-A: Metric 5 N=7 -> N=24 instances spanning 143-24770 chars; 90%/80% reduction stable.
  - P1-B: vLLM APC analytic comparison section added to WRITEUP.
- STILL MOCKED: Metric 5 (macro) synthetic fill; cross-domain RNG/TOOL host-sim.
- STILL NOT DONE: full 28-layer model; real vLLM patch (analytic only); multi-GPU;
  SWE-bench harness run; concurrent-capacity beyond 84 (mapping-metadata ceiling is real).

## Metric status (R2): 7 of 7 measured (5c added). Memory/capacity strong; latency at
## parity (honest); attn overhead ~0%; correctness hard-asserted; one R1 overclaim (B8)
## retracted with measured evidence.
