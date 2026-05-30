# CROSS-VENDOR VMM CEILING — measured 2026-05-30 (committee_naviC E2)

## NVIDIA H100 (from forkedkv repo, Metric 4b / Lab 1)
- Granularity: 2 MiB. Ceiling: K = branches × prefix_pages ≈ 520,000 (±1%).
- Fails forensically at cuMemSetAccess. Independent of vm.max_map_count (392 VMAs vs 67M).

## AMD MI350X gfx950, ROCm/HIP (NEW, this committee — /tmp/hip_vmm_ceiling.cpp)
- Granularity: 4096 bytes (4 KB) — 512× finer than NVIDIA's 2 MiB.
- Mapped 4,000,000 shared-physical→distinct-VA pages (one phys handle, hipMemMap+
  hipMemSetAccess each) with ZERO driver failure. Hit our 15.26 GiB VA-reserve cap, NOT a
  driver ceiling. No cuMemSetAccess-equivalent refusal observed.
- => AMD does NOT enforce the ~520K per-context mapping-metadata ceiling NVIDIA does.

## HONEST IMPLICATION (corrects Thesis A framing)
The K≈520K ceiling is NVIDIA-DRIVER-SPECIFIC, not a universal property of GPU VMM.
The defensible characterization is therefore VENDOR DRIVER-ARCHITECTURE DIVERGENCE:
NVIDIA enforces a low per-context mapping-metadata ceiling (520K @ 2MB granule) that AMD
(4KB granule, ≥4M mappings) does not. This has direct portability consequences for any
VMM-based KV/branch/CoW system: a design that scales on AMD can hit a hard wall on NVIDIA.
This is a STRONGER, more falsifiable claim than "structural GPU limit" and survives the
round-1 "single-driver quirk" kill precisely BECAUSE we measured the divergence.

CAVEAT (state in paper): AMD not pushed to its absolute ceiling (VA-reserve capped at
15.26GiB/4M pages); claim is "≥4M, no ceiling at 512× NVIDIA's count," not a measured K_amd.
Repro: raise VA reserve + LIMIT to find AMD's true ceiling (cheap follow-up).

## UPDATE — AMD pushed toward true ceiling (2026-05-30, MI350X)
- ROCm 7.0.2.1, gfx950, 288 GiB HBM (309,220,868,096 B VRAM).
- Reserved 244.1 GiB VA (64,000,000 × 4KB pages). Mapped ALL 64,000,000
  shared-physical→distinct-VA pages via hipMemMap+hipMemSetAccess. ZERO driver failure.
- 64M mappings = 123× NVIDIA H100's 520K ceiling. This is STILL our VA-reserve cap, NOT an
  AMD driver ceiling — AMD shows no per-context mapping-metadata wall in the regime where
  NVIDIA hard-fails at cuMemSetAccess.
- Probe: /tmp/hip_vmm_ceiling_max.cpp on cli:devgpu499. Repro: raise VA reserve further.

## FINAL CROSS-VENDOR VERDICT (firm)
| Vendor | Driver | Granule | Max mappings observed | Driver mapping ceiling? |
|---|---|---|---|---|
| NVIDIA H100 | CUDA 12.8 / 580.82.07 | 2 MiB | ~520,000 (K=branches×prefix_pages, ±1%) | YES — hard fail at cuMemSetAccess |
| AMD MI350X | ROCm 7.0.2.1 / gfx950 | 4 KB | 64,000,000 (cap, not ceiling) | NO ceiling found at 123× NVIDIA |

The ~520K per-context VMM mapping ceiling is NVIDIA-DRIVER-SPECIFIC. AMD's HIP VMM does not
enforce it in this regime. Honest framing for any thesis: this is a VENDOR DRIVER-ARCHITECTURE
DIVERGENCE with portability consequences for VMM-based KV/branch systems — NOT a universal GPU law.

## E-A1 PARTIAL (2026-05-30, MI350X) — strong even though node crashed
Pushed AMD true-ceiling probe with chunked VA reservation. Last captured progress before the
node went unresponsive: **50,000,000 mappings at 191 GiB VA, ZERO driver failure** (96× NVIDIA's
520K). The probe's unbounded VA growth eventually thrashed devgpu499 and took its CLI agent
OFFLINE (same failure mode the H100 hit earlier).
OPERATIONAL LESSON: unbounded VMM mapping probes destabilize the host — future probes MUST cap
VA reservation well below host memory headroom and run nice/cgroup-limited.
NET FOR A*: across two independent runs AMD reached 4M (early) and 50M (this run) with NO
mapping-metadata ceiling, vs NVIDIA's hard 520K wall. The qualitative cross-vendor divergence
is firmly established at 8×–96× NVIDIA with zero AMD failures. A precise AMD K_ceiling remains
unmeasured (AMD may simply not enforce one in any reachable VA), but CC48's core ask —
"is the absence real or just an un-hit cap?" — is answered directionally: two runs, two scales,
no wall, while NVIDIA fails reproducibly at the SAME 520K. The honest claim stands: NVIDIA-
specific ceiling; AMD shows none in 191 GiB of reachable VA.
