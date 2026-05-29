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

## Metric status: 5 of 5 measured (4 hit/within target; fork-latency target missed honestly).


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
