# E1 FINAL — AMD MI350X (gfx950, ROCm 6.4/7.0) VMM mapping-ceiling probe
Tests Muse Park's T1 RED: "the ~520K branches x prefix_pages invariant is a CUDA-driver
artifact that collapses on ROCm/MI300X." HIP analog of forkedkv M4b.

## Results (MI350X back from maintenance, re-run 2026-05-30)
- HIP VMM granularity = 4096 B (vs 2 MiB on H100/CUDA — 512x finer page).
- P=1 (one 4KB shared physical page mapped into N new VAs via
  hipMemAddressReserve + hipMemMap + hipMemSetAccess):
    2,000,000 live mappings: ZERO HIP VMM failures, HBM free FLAT at 286.6 GiB.
    (earlier push to 4,000,000: still zero VMM failures.)
- P=16 (earlier run): 400,000 branches = 6.4M page-mappings: zero VMM failures.
- The ONLY death observed = host-side SIGKILL (navi-node cgroup) at ~2.5M VA reservations =
  HOST bookkeeping, NOT a HIP VMM OOM. No hipMemSetAccess/hipMemMap/AddressReserve EVER errored.

## VERDICT — Muse Park's T1 RED CONFIRMED
CUDA/H100: OOMs INSIDE cuMemSetAccess at branches x pages ~= 520K (forkedkv M4/M4b) — a hard
DRIVER mapping-metadata ceiling. AMD/ROCm: NO equivalent VMM-level ceiling below 2-4M mappings;
HBM flat; only host RAM/cgroup limits. The ~520K ceiling + "cuMemSetAccess is the OOM site"
forensic are NVIDIA-CUDA-driver-SPECIFIC. The multiplicative invariant FORM does NOT reproduce
as a VMM-driver law on ROCm.
=> T1 DEMOTED: "cross-vendor capacity law" -> "NVIDIA-CUDA-VMM resource characterization."

## What survives (honest, narrowed):
- ON NVIDIA: mapping-table is a real exhaustible HBM-orthogonal OS-independent resource no
  scheduler tracks. Valid NVIDIA-scoped systems contribution.
- The CROSS-VENDOR CONTRAST is itself a finding: CUDA exposes a hard per-context mapping ceiling
  that ROCm does not (<=2-4M). That asymmetry is publishable characterization.
