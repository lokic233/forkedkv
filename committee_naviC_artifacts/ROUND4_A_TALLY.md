# ROUND 4 — A* FINAL VOTE (after E-A2 mechanism + E-A1 two-run AMD divergence)

| Thesis | CC48 | CC47 | CC46 | Muse | Codex | Gemini | CONSENSUS |
|---|---|---|---|---|---|---|---|
| **A\*** NVIDIA VMM 520K vendor portability cliff | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | **6/6 GREEN ✓** |

CC48 (round-3 holdout) FLIPPED to GREEN. Its reasoning: a portability cliff needs (a) one
side's wall pinned [NVIDIA 520K ±1% across 4 prefix sizes, independently reproduced at 523,404]
and (b) the other side not hitting it in reachable range [AMD 96× headroom, 4M & 50M, no wall].
Both discharged. The predictive model max_branches≈K/prefix_pages is NVIDIA-validated to ±1%
regardless of AMD's exact ceiling.

UNANIMOUS CAVEAT (for the paper, not a validity threat): soften "ABSENT on AMD" to the measured
"no ceiling reached in 191 GiB across 4M/50M mappings." Future nice-to-have: a true
hipMemSetAccess failure on a big-memory host, or driver-instrumented NVIDIA tunability probe.

## SCOREBOARD: 2 of 3 GREEN (C*, A*). Need 1 more.
