# BACKTEST / MVP EXPERIMENT REPORT

## E1 — T1 cross-vendor mapping ceiling on AMD MI350X (gfx950, ROCm 6.4/7.0) — PARTIAL, DECISIVE
Built hip_vmm_ceiling.cpp (HIP analog of forkedkv M4b: hipMemCreate/hipMemAddressReserve/
hipMemMap/hipMemSetAccess, sweep prefix_pages, map shared chunk into new VAs until failure).
RESULT (P=16, page granularity=4096 B on AMD vs 2 MiB on H100):
  max_branches hit our 400,000 cap with FAILING_CALL=none -> >= 6,400,000 live mappings, ZERO failures.
INTERPRETATION: On H100/CUDA, the analogous sweep OOMs at cuMemSetAccess with branches x pages
  ~= 520K. On AMD/ROCm the same construction sustained >=6.4M mappings (12x past the CUDA wall)
  with no failure at our cap. => The ~520K ceiling does NOT reproduce on ROCm.
VERDICT IMPACT: This CONFIRMS Muse Park's T1 RED: the 520K invariant is a CUDA/NVIDIA-driver
  characteristic, NOT a cross-vendor hardware law. T1 must be honestly demoted from "cross-vendor
  capacity law" to "NVIDIA-CUDA-VMM resource characterization" (still a real, schedulable,
  HBM-orthogonal resource ON NVIDIA — but not universal). The multiplicative FORM may still hold
  per-stack; AMD's ceiling (if any) is >=12x higher and was not reached at our cap.
CAVEATS: (1) AMD granularity is 4KB not 2MB, so "pages" are not size-comparable; the relevant
  comparison is raw mapping COUNT, where AMD >> CUDA. (2) cap not raised to true AMD ceiling
  (box went offline under cleanup churn). (3) Single MI350X, ROCm 6.4. Re-run with higher cap +
  binary-search to true ceiling when devgpu499 is back online.

## E2 — T2 cross-engine (SGLang tool-injection penalty) — NOT RUN
Deferred: requires standing up SGLang + Qwen2.5-7B on H100 (~1-2h). This is the remaining
tie-breaker for T2 (Muse Park RED: "SGLang won't show >5x"). High EV; queued.

## NET EFFECT ON CONSENSUS
- T1: E1 partial result SUPPORTS the holdout. Honest move = demote T1 to NVIDIA-scoped
  characterization. Under that scoping, all 6 likely converge (it stops over-claiming).
- T2: still 5G/1R; needs E2 to settle empirically. Do NOT declare GREEN on rhetoric.
- T3: 4G/2Y, no RED, no experiment run; remains YELLOW (consensus).
