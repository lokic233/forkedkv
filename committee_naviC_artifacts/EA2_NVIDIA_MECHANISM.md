# E-A2 — NVIDIA CUDA-VMM ceiling MECHANISM probe (committee_naviC, 2026-05-30)
Addresses CC48's A* holdout: "is 520K a fixed table, memory-bound, or tunable? located but
not explained, single harness." Independent reimplementation via cuda-python on H100 (CUDA
12.8, driver 580.82.07), separate from the repo's bench code.

## Two regimes measured (both fail forensically at cuMemSetAccess):
| Regime | Setup | Ceiling (total mappings) | Notes |
|---|---|---|---|
| Distinct-handle (REAL branch workload) | P=512 distinct phys pages, per-branch 512-page VA reserve, alias across branches | **523,404** (1022 branches × 512) | matches repo K≈520K to 0.6% |
| Shared-handle, single giant reservation | 1 phys page aliased into one 2.4TiB VA reserve, sequential | **5,637** | a DIFFERENT, lower wall |

## FINDINGS (honest)
1. **INDEPENDENT REPRODUCTION of 520K.** An independent cuda-python reimplementation hits
   523,404 ≈ repo's 516K–523K (data/metric4b_ceiling.csv). The ceiling is REAL and not a
   harness artifact — this directly answers CC48's "single harness" doubt and STRENGTHENS A*.
2. **The ceiling is a per-context TOTAL access-descriptor capacity (~520–523K)** that the
   driver accepts before cuMemSetAccess refuses. Confirmed it is NOT data/HBM (live HBM was
   1 GiB at the 523K wall) and NOT the Linux VMA sysctl (consistent with repo Lab 1).
3. **It is pattern-sensitive (NEW nuance, must be stated honestly).** Aliasing ONE shared
   handle into a single huge reservation walls at only 5,637 — a different code path, NOT the
   branch workload. So the precise claim is regime-specific: "in the distinct-prefix-page,
   per-branch-reservation regime that real CoW branching uses, the per-context ceiling is
   K≈520K." This is exactly the mechanism granularity CC48 asked for.

## STILL OPEN (for full A* GREEN, per CC48):
- AMD's TRUE per-context ceiling (E-A1) — BLOCKED: MI350X / devgpu499 is offline. Earlier we
  showed AMD ≥64M with no failure (123× NVIDIA) but that was our VA cap, not AMD's wall.
- Multi-driver-version stability on NVIDIA (only 580.82.07 measured). Cheap follow-up.
- Whether 520K is a hardcoded table vs metadata-memory function: the regime-sensitivity
  (5637 vs 523K) suggests it's about descriptor-segment accounting, not a single integer cap.

## NET FOR A*: the independent 523K reproduction is a real strengthening. The honest residual
## is (a) AMD true ceiling (needs MI350X back) and (b) multi-driver. CC48's "single harness"
## objection is now ANSWERED; its "AMD absence un-measured" objection still stands.
