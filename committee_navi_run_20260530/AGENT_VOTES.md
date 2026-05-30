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
