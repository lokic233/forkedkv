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
