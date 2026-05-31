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

## ROUND 6 — re-vote on T2 after E4 (single-layer repair implementation)
| T2 (post-E4) | 4.8 | 4.7 | 4.6 | codex | muse | gemini | Tally |
|--------------|-----|-----|-----|-------|------|--------|-------|
|              | RED | GREEN | GREEN | RED | GREEN | GREEN | 4G/2R |
KEY EVENT: Muse Park FLIPPED to GREEN (holdout satisfied by built+exact+18.8x repair). BUT
claude48 + codex independently caught a REAL BUG: E4 proved equality only at LAYER 0; for an
INTERIOR edit the suffix attends to the changed slot so suffix K/V change at L>0 -> pointer-stable
reuse is NOT exact. The single-layer E4 masked it.

## E4b — multi-layer test of the claude48/codex RED. CONFIRMED THE BUG.
Interior edit: suffix max|dK| L1..L5 = 0.64,2.54,1.47,2.78,4.04 (diverges). Terminal edit: 0.0 at
all layers. => E4's interior 18.8x was INVALID (skipped required suffix recompute). T2 repair
rescoped to TERMINAL/APPEND edits only.

## E4c — rescoped terminal repair, multi-layer, exact+timed. VALIDATED.
argmax-exact at every S; 5.4x@2K -> 81.6x@16K vs full recompute. Correct + fast at full depth.

## ROUND 7 — re-vote on CORRECTED + RESCOPED T2
| T2 (R7) | 4.8 | 4.7 | 4.6 | codex | muse | gemini | Tally | Status |
|---------|-----|-----|-----|-------|------|--------|-------|--------|
|         | YELLOW | GREEN | GREEN | GREEN | RED | GREEN | 4G/1Y/1R | YELLOW |
codex FLIPPED to GREEN (its bug fixed). NEW SHARED OBJECTION (claude48 YELLOW + Muse Park RED):
"Terminal/append repair == standard prefix-KV incremental prefill that vLLM/SGLang ALREADY do; the
5-82x is vs a full-recompute STRAWMAN. Prove it beats a prefix-caching engine, else only the
E2/E3 workload characterization survives as the contribution."

CRITICAL SELF-ASSESSMENT: this objection is CORRECT and partly self-defeating for T2's primitive.
Our OWN E3 measured E_append on live SGLang RadixAttention at 0.90-0.96x (FREE) — i.e. a
prefix-caching engine ALREADY handles pure terminal append cheaply. So the terminal "repair" is
NOT a novel primitive over RadixAttention. The honest consequence: T2's RUNTIME-PRIMITIVE half
collapses to "what RadixAttention already does"; only the WORKLOAD-MODEL half (E2/E3 pathology +
edit-cost taxonomy) is a defensible novel contribution.

## ROUND 8 — T2 NARROWED (runtime-primitive WITHDRAWN, full-trace disclosure incl. self-defeating data)
| T2-narrowed | 4.8 | 4.7 | 4.6 | codex | muse | gemini | Tally |
|-------------|-----|-----|-----|-------|------|--------|-------|
|             | YELLOW | GREEN | GREEN | YELLOW | GREEN | GREEN | 4G/2Y |
Both YELLOWs (4.8, codex) gated on TWO precise copy-fixes: (1) 13.63x is E_prepend, not interior
(conflated); (2) "impossibility" overclaims -> "empirical quantification of known constraint."

## ROUND 9 — FINAL: both corrections applied verbatim.
| T2-FINAL | 4.8 | 4.7 | 4.6 | codex | muse | gemini | Tally | STATUS |
|----------|-----|-----|-----|-------|------|--------|-------|--------|
|          | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | **6G/0Y/0R** | **GREEN (UNANIMOUS)** |

=== FIRST UNANIMOUS GREEN THESIS REACHED ===

## ROUND 10 — E5 self-hostile test of the GREEN (real-trace incidence)
| T2 + E5 | 4.8 | 4.7 | 4.6 | codex | muse | gemini | Tally |
|---------|-----|-----|-----|-------|------|--------|-------|
|         | GREEN | YELLOW | GREEN | YELLOW | GREEN | GREEN | 4G/2Y |
E5 (325 real Claude Code sessions): ~99.9% APPEND (free), ~0.1% RESET, 0% INTERIOR -> the 8-13x
pathology has ~0% incidence in mainstream append-only harnesses. The GREEN BROKE. YELLOWs (4.7,
codex): must reframe from "workload-model thesis" to "conditional cost-map / design-constraint"
with ~0% incidence stated UP FRONT, no broad-impact claim.

## ROUND 11 — FINAL: reframed to conditional cost-map, ~0% incidence as headline sentence.
| T2-FINAL | 4.8 | 4.7 | 4.6 | codex | muse | gemini | Tally | STATUS |
|----------|-----|-----|-----|-------|------|--------|-------|--------|
|          | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | **6G/0Y/0R** | **GREEN (UNANIMOUS, E5-SURVIVED)** |

=== T2 is GREEN as a CONDITIONAL ARCHITECTURAL COST-MAP + DESIGN CONSTRAINT (negative-result style) ===
The thesis survived its own incidence audit by narrowing to exactly what 5 experiments measured and
leading with the honest ~0% current-harness incidence. No overclaim survives.

## ROUND 12 — T3 disposition after E6 (which FALSIFIED T3's intended angle)
| T3 + E6 | 4.8 | 4.7 | 4.6 | codex | muse | gemini | Tally | STATUS |
|---------|-----|-----|-----|-------|------|--------|-------|--------|
|         | KILL | KILL | KILL | KILL | KILL | KILL | **6/6 KILL** | **RETIRED** |
E6 built to answer T3's YELLOWs (does token-share overstate KV-share?). A discipline CONTROL
(identical 207-tok prefix, prefix-only vs prefix+800) showed max|dK| up to 6.25e-2 >> 5e-3 tol =
fp16 SDPA kernel nondeterminism, NOT semantic divergence. With the artifact controlled, token-prefix
sharing PREDICTS KV sharing -> RadixAttention already captures the opportunity -> T3's only
differentiator is dead. Committee unanimously retired T3 rather than launder the trivial
1-(tail/total) form into a YELLOW. NEGATIVE RESULT, honestly reported.

## ROUND 13 — NEW thesis T4 "layer-stratified positional KV reusability" (measured backbone E8)
| T4 | 4.8 | 4.7 | 4.6 | codex | muse | gemini | Tally | STATUS |
|----|-----|-----|-----|-------|------|--------|-------|--------|
|    | RED | YELLOW | YELLOW | YELLOW | YELLOW | KILL | 0G/4Y/1R/1K | NOT GREEN |
Measurement (E8) is clean + acknowledged real by all. But weaknesses compound fatally:
~1.5% real exact-repeat incidence (E5) x exact-reuse confined to ~1-2 of 28 shallow layers (least
FLOPs) => savings ceiling near-zero (gemini). 4.8: L0 exactness is partly a RoPE tautology
(pre-RoPE projections position-independent by construction). Two agents (47,46) gave the SAME
GREEN-gate experiment: measure end-to-end prefill FLOPs/latency saved by shallow-layer KV reuse for
real repeats vs recompute -> GREEN if >=10% saved at matched quality, KILL if <2%.
DECISION: run the gate experiment (E9) before any GREEN claim. Predicted to fail (incidence x
shallow-only bound), but measure don't assume.

## ROUND 14 — candidate T5 "error-class predicts agent recovery" (failure-attribution, E10)
| T5 | 4.8 | 4.7 | 4.6 | codex | muse | gemini | Tally | STATUS |
|----|-----|-----|-----|-------|------|--------|-------|--------|
|    | YELLOW | YELLOW | YELLOW | YELLOW | YELLOW | KILL | 0G/5Y/1K | NOT GREEN |
Honest ~3x error-class repeat-rate spread (path_missing 18.6% vs other 6.1%) on 6,510 real calls.
All agents: real but THIN bounded characterization; single harness/family, ~200 well-classified
errors, no intervention, crowded reflexion/self-debug prior art. Self-disclosed 2 parsing bugs
(46.8%->18.6% classification artifact; "178 spirals"->10 redundant = 0.2%, negligible) earned trust
but the signal shrank under scrutiny. GREEN-gate (47+46, identical): (1) cross-harness replication
(OpenHands/SWE-agent/Cursor) + (2) error-class-aware reformatting A/B cutting repeat-rate >=50%.
BLOCKED today: needs other-harness traces (not on disk) + live A/B runs. Not closeable H100-only.

## T5 GREEN-GATE ATTEMPTED (E11 cross-harness + E12 intervention A/B) — BOTH FELL SHORT.
- E11 (codex traces = 2nd harness/family): only 9 errors/208 calls (codex error rate 4.3% vs CC
  13.2%) -> UNDERPOWERED, can't confirm class->repeat on a 2nd harness from on-disk traces.
- E12 (intervention A/B, live claude agent): first cut showed control 0/20 vs treatment 20/20
  (+100pts) -> looked like a GREEN-clinching intervention. E12b HARDENED CONTROL (added cwd +
  realistic layout) showed control recovers 8/8 ON ITS OWN -> the win was a TOY-SETUP ARTIFACT
  (impoverished control). Effect collapses under a fair baseline.
=> Neither gate half holds. T5 remains YELLOW, NOT promoted. Discipline: the impressive 0->100%
   was self-audited and retracted before any GREEN claim. (Self-caught overclaim #3 this session.)
