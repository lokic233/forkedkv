# ROUND 2 — Convergence vote on sharpened A/B/C (6 agents)

| Thesis | CC48 | CC47 | CC46 | Muse | Codex | Gemini | CONSENSUS |
|---|---|---|---|---|---|---|---|
| A (VMM ceiling/vendor divergence) | RED | RED-lean | (see file) | RED | YELLOW | YELLOW | **RED** (AMD non-repro sank headline) |
| B (superlinear prefix-injection) | YELLOW-cond | YELLOW | (see file) | RED | YELLOW | YELLOW | **YELLOW** |
| C (when-NOT-to-use HW CoW; negative result) | YELLOW(closest) | YELLOW | (see file) | YELLOW | **GREEN** | **GREEN** | **YELLOW** (2 GREEN, rest cond-YELLOW) |

## Key finding: the committee got MORE brutal (working as designed).
- My AMD MI350X result CORRECTLY killed A's headline: AMD didn't reproduce the 520K ceiling
  (4KB granule, 4M+ mappings, no failure) → "vendor quirk, not structural law." Agents flagged
  the granularity+cap confound. Honest hostile review, not consensus theater.
- C is the survivor everyone points to. The UNANIMOUS flip-to-GREEN condition:
  ONE experiment (E-C): end-to-end rollback-heavy trace, HW VMM CoW vs vLLM-APC software
  vs FlashInfer-paged. If HW CoW wins even ONE regime → C GREEN (and drags A's contiguous-VA
  survivor-claim). If it wins nothing → clean citable negative result (still C, still publishable).
- B flip condition: replace EDMM oracle with a REAL heuristic predictor + position-aware model
  (R²≥0.9 across ≥3 engines) showing live latency reduction.

## ACTION TAKEN (run experiments, don't re-word):
- E-C launched as sub-agent on devgpu014 (the decisive vote-flipping experiment).
- AMD true-ceiling probe launched on devgpu499 (firms up cross-vendor data for A/divergence).
- Next: E-B (real predictor for EDMM) if C lands and we need a 3rd GREEN.
