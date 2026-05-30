# AGENT_VOTES — 6-agent committee, 3 rounds (Navi run, 2026-05-30)
Engines: claude-opus-4-8 / 4-7 / 4-6, codex (gpt-5.5), metacode (Muse Park/Avocado), gemini-3.5
All run via ~/.navi/bin/cleanenv on devgpu014 (H100). GitHub artifacts = source of truth.

## ROUND 1 — open ideation (each agent proposed 4-5 theses). Clustered into 5 themes.
## ROUND 2 — vote on 3 refined survivors (T1 mapping-wall, T2 invalidation, T3 spec-prefill).
   T1: 2G/4Y | T2: 4G/2Y | T3: DEAD (3 RED — infeasible exact-token predictor; request-idle != GPU-idle).
## ROUND 3 — sharpened T1/T2, replaced dead T3 with divergence-workload-model.

| Thesis | 4.8 | 4.7 | 4.6 | codex | muse | gemini | Tally | Status |
|--------|-----|-----|-----|-------|------|--------|-------|--------|
| T1 Mapping-table resource | YELLOW | GREEN | GREEN | GREEN | RED | GREEN | 4G/1Y/1R | YELLOW (consensus) |
| T2 Mid-prompt invalidation | GREEN | GREEN | GREEN | GREEN | RED | GREEN | 5G/1R | YELLOW (1 RED blocks) |
| T3 Divergence workload-model | YELLOW | GREEN | GREEN | GREEN | YELLOW | GREEN | 4G/2Y | YELLOW (consensus) |

CONSENSUS RULE: GREEN=6G, YELLOW=any Y & no R, RED=any R.
=> No thesis is 6/6 GREEN. T2 is strongest (5G). Muse Park is the sole holdout on T1+T2 with
   EMPIRICAL (testable) objections, and YELLOW on T3.

## MUSE PARK (Muse Park) BLOCKING OBJECTIONS — all empirically testable:
- T1: "invariant collapses on ROCm/MI300X (CUDA-driver artifact, not law)"; "needs reserve/query API, not just a predictive microbench."
- T2: "8.21x is a vLLM 0.6.6 bug, not fundamental — SGLang will NOT show >5x"; "RoPE-invariant repairable subclass is vacuous for variable-length tool outputs."
- T3: YELLOW — "divergence model is trivial 1-(tail/total); 3 coding harnesses != general 'agentic'."

## DECISION: tie-break by EXPERIMENT, not more rhetoric.
The committee converged to 4-5 GREEN on all three; the lone holdout's REDs are bets about data.
We have the hardware to settle two of them (AMD MI350X for T1 cross-vendor; H100 SGLang for T2).
=> Proceed to MVP validation experiments E1 (T1 AMD) and E2 (T2 SGLang).

## ROUND 4 — re-vote on T2 ONLY, after E2 experiment (SGLang cross-engine evidence)
| Thesis | 4.8 | 4.7 | 4.6 | codex | muse | gemini | Tally | Status |
|--------|-----|-----|-----|-------|------|--------|-------|--------|
| T2 (post-E2) | YELLOW | GREEN | GREEN | GREEN | RED | YELLOW | 3G/2Y/1R | YELLOW |

WHAT E2 RESOLVED: Muse Park's primary RED ("8.21x is a vLLM bug; SGLang won't show >5x") is
EMPIRICALLY FALSIFIED and CONCEDED by Muse Park itself ("E2 torpedoes the >5x fairy tale").
SGLang-7B reaches 12.77x @32K; the pathology is architectural & cross-engine. The WORKLOAD-MODEL
half of T2 is now settled.

WHAT SURVIVED (the committee converged on a NEW, precise, shared flaw):
The REPAIR PRIMITIVE is unvalidated. 3 agents independently (claude48, gemini, metacode) raised
the SAME objection: the mid-prompt injection E2 measured is a POSITION-SHIFTING edit = exactly the
RoPE-NON-invariant subclass that the VMM pointer-swap CANNOT repair, and nobody has measured what
fraction of REAL agent tool-call edits fall in the repairable (RoPE-invariant) subclass. The
repair is vLLM-only, not ported to SGLang.

=> T2 is YELLOW (1 RED blocks GREEN, but the RED narrowed from "whole thesis is a bug" to
   "repair half unvalidated"). The workload-model contribution alone is now defensible.
   Next experiment E3 (defined below) directly tests the surviving flaw.

## ROUND 5 — re-vote on T2 after E3 (edit-type taxonomy experiment)
| Thesis | 4.8 | 4.7 | 4.6 | codex | muse | gemini | Tally | Status |
|--------|-----|-----|-----|-------|------|--------|-------|--------|
| T2 (post-E3) | YELLOW | GREEN | GREEN | YELLOW | RED | YELLOW | 2G/3Y/1R | YELLOW |

WHAT E3 RESOLVED: the Round-4 "repairable subclass is vacuous" objection is DEAD. E3 showed
RoPE-invariant edits (append 0.9-0.96x, fixed-width 1.3-2.4x) are 3-14x cheaper than
position-shifting edits (3-13.6x), cross-model. Muse Park conceded vacuity.

SURVIVING FLAW (committee converged again, even tighter): TWO build-required items remain:
  (1) an ACTUALLY-IMPLEMENTED VMM pointer-remap repair, measured as an intervention (not just the
      recompute cost it avoids) -- raised by Muse Park, claude48, codex.
  (2) real-trace edit-class FREQUENCY census -- raised by all.
Both require BUILDING, not arguing. The thesis framing now survives; what's left is engineering
+ measurement, defined precisely by E4/E5 below.

## EVOLUTION OF THE HOLDOUT'S OBJECTION (the loop working):
  R3: "8.21x is a vLLM bug, SGLang won't show >5x"   -> killed by E2 (SGLang 12.77x)
  R4: "repairable RoPE-invariant subclass is vacuous" -> killed by E3 (0.9x vs 8-13x, tracks RoPE)
  R5: "repair not implemented + real-trace incidence unknown" -> E4/E5 (build + census)
Each experiment killed one objection and surfaced the next-deepest. T2 is now a defined build spec,
not a framing dispute. Workload-model half: SETTLED. Runtime-primitive half: spec'd, not built.
