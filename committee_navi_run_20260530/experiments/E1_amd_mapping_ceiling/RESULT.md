# E1 FINAL — T1 cross-vendor mapping ceiling on AMD MI350X (gfx950, ROCm 6.4/7.0, devgpu499)
Tests Muse Park's T1 RED: "the ~520K branches x prefix_pages invariant is a CUDA-driver artifact
that collapses on ROCm/MI300X." HIP analog of forkedkv M4b (hipMemCreate/hipMemAddressReserve/
hipMemMap/hipMemSetAccess, map a shared physical handle into N new VA ranges until a VMM call fails).

## Results
- HIP VMM granularity = 4096 B (vs 2 MiB on H100/CUDA — 512x finer).
- P=1 (one 4KB shared page mapped into N virtual ranges):
    * 2,000,000 live mappings: ZERO HIP VMM failures, HBM free FLAT at 286.6 GiB.
    * 4,000,000 live mappings: still ZERO HIP VMM failures (hit our MAXB cap, not a driver limit).
- P=16: 400,000 branches = 6.4M page-mappings: zero VMM failures.
- The ONLY process death was a host-side SIGKILL (navi-node cgroup) at ~2.5M VA reservations —
  HOST bookkeeping, NOT a HIP VMM call returning OOM. No hipMemSetAccess/hipMemMap/
  hipMemAddressReserve EVER returned an error.

## VERDICT — Muse Park's T1 RED is CONFIRMED
On CUDA/H100 the analogous construction OOMs INSIDE cuMemSetAccess at branches x pages ~= 520K
(forkedkv M4/M4b) — a hard driver mapping-metadata ceiling. On AMD/ROCm, NO equivalent VMM-level
ceiling appears below 2-4M mappings (>=4-8x the CUDA wall in raw count, ~512x more in page terms
given 4KB vs 2MB granularity); HBM stays flat and the only limit is host RAM/cgroup.
=> The ~520K ceiling and "cuMemSetAccess is the OOM site" forensic are NVIDIA-CUDA-driver-SPECIFIC.
   The multiplicative invariant FORM does NOT reproduce as a VMM-driver law on ROCm.
   T1 honestly DEMOTED: "cross-vendor capacity law" -> "NVIDIA-CUDA-VMM resource characterization."

## What survives for T1 (honest, narrowed):
- ON NVIDIA: the mapping-table IS a real, exhaustible, HBM-orthogonal, OS-independent resource no
  serving scheduler tracks. Valid NVIDIA-scoped systems contribution.
- The CROSS-VENDOR CONTRAST is itself a FINDING: CUDA exposes a hard per-context mapping ceiling
  that ROCm does not (<=2-4M). That asymmetry is publishable characterization on its own.

## Artifacts on devgpu499: ~/edmm_mvp_e1/ (hip_vmm_ceiling.cpp, hip_ceiling_true.cpp, *_out.txt, E1_FINAL.md)
