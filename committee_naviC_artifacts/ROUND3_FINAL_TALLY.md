# ROUND 3 — FINAL VOTE (on measured evidence: cross-vendor + E-C 0/12)

| Thesis | CC48 | CC47 | CC46 | Muse | Codex | Gemini | CONSENSUS |
|---|---|---|---|---|---|---|---|
| **A\*** vendor VMM-ceiling portability cliff | YELLOW | GREEN | GREEN | GREEN | GREEN | GREEN | **YELLOW** (5 GREEN, 1 YELLOW) |
| **B** superlinear prefix-injection workload model | YELLOW | YELLOW | YELLOW | YELLOW | YELLOW | YELLOW | **YELLOW** (unanimous) |
| **C\*** when-NOT-to-use HW CoW (measured negative result) | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | **GREEN — 6/6 ✓** |

## RESULT: 1 thesis at full 6/6 GREEN (C*). A* is one vote short (CC48 holdout).

## CC48's A* holdout (the only thing between us and a 2nd GREEN) — SPECIFIC & ADDRESSABLE:
1. AMD "absence" is an UN-MEASURED cap (hit our 244GiB VA reserve at 64M, not AMD's real
   ceiling). Need AMD's ACTUAL per-context ceiling → lift the VA cap.
2. NVIDIA 520K mechanism is LOCATED (cuMemSetAccess) but not EXPLAINED (fixed table? memory-
   bound? tunable?) and only ONE driver version (580.82.07). Need ≥2 driver versions or a
   mechanism probe.
3. 123× headline partly a granularity artifact (2MiB×520K vs 4KB×64M) — normalize per-entry
   metadata cost.
All others voted A* GREEN on the two-vendor divergence as-is.

## B holdout (unanimous YELLOW): needs non-oracle recovery OR a validated position×frequency×
## context predictive model on held-out agent traces. No new experiment was run for B this round.

## PATH TO 3× GREEN (honest, experiment-driven):
- C* = GREEN now. ✓ (1/3)
- A* → flip CC48: (E-A1) lift AMD VA cap, push to real hipMemMap/SetAccess failure = AMD's
  true ceiling; (E-A2) re-run NVIDIA probe normalized per-mapping + probe whether 520K scales
  with metadata memory. Cheapest path to a 2nd GREEN. (2/3)
- 3rd GREEN: either (E-B) build a real injection-position predictor for B, OR mine round-1
  for the next-strongest NEW candidate (cross-domain CoW D / write-after-fork isolation E)
  and run its gating experiment. (3/3)
