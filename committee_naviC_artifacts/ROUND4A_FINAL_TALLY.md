# ROUND 4 — A* FINAL VOTE (on E-A2 mechanism + E-A1 two-run AMD divergence)

| Thesis | CC48 | CC47 | CC46 | Muse | Codex | Gemini | CONSENSUS |
|---|---|---|---|---|---|---|---|
| **A\*** NVIDIA VMM-ceiling vendor portability cliff | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | **GREEN — 6/6 ✓** |

## CC48 (round-3 holdout) FLIPPED → GREEN. Both objections resolved:
- Obj1 (mechanism): E-A2 independent cuda-python reimpl reproduced 523,404 ≈ repo 520K (±0.6%),
  forensically at cuMemSetAccess, regime-characterized (520K distinct-handle vs 5,637 shared).
- Obj2 (AMD absence): two independent AMD runs (4M, 50M maps) zero failure = 96× NVIDIA wall.
- CC48 verdict: "A cliff that one vendor hits at 5.2e5 and the other does not hit at 5.0e7 is a
  portability cliff regardless of whether AMD's wall sits at 1e8 or infinity. The 100x gap is
  the artifact, and it's measured." All 6 agree AMD's exact ceiling is a limitations paragraph,
  not a validity threat.

## A* = GREEN #2 of 3. Venue: ATC/EuroSys/OSDI. Honest residual: AMD exact K unmeasured
## (host limits hit first), multi-driver-version NVIDIA stability (cheap follow-up).
