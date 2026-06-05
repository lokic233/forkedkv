# ROUND 11 — FINAL. T2 reframed per the R10 YELLOWs (claude47, codex): from "workload-model thesis"
# to "conditional cost-map / design-constraint", with ~0% current incidence stated UP FRONT.
# Source of truth = measured data. GREEN only if positioning now matches evidence with zero overclaim.

## T2 — FINAL FORM (conditional architectural cost-map + design constraint):
HEADLINE (stated up front, not a footnote): "In mainstream append-only agent harnesses (measured:
325 Claude Code sessions, 12,380 turns), mid-prompt cache-invalidation has ~0% incidence — tool
results are appended at the tail and prefix caching handles them for FREE. The pathology below is
therefore a LATENT architectural cost, not an active problem for today's dominant agents."

CONTRIBUTION (what we DO claim, all measured):
1. An architectural COST MAP of KV-cache behavior under context mutation, per edit class, on live
   engines: APPEND = free (0.90-0.96x, E3); interior fixed-width = 1.3-2.4x; interior variable =
   3.0-8.2x; prepend up to 13.6x (separate class); cross-engine 8-13x (vLLM 8.21x, SGLang 12.77x, E2).
2. A structural EXPLANATION + quantification (tested stacks) of WHY interior edits aren't cheaply
   KV-repairable: an interior edit changes suffix K/V at every layer >=1, forcing suffix recompute
   (E4b: suffix |dK| up to 4.04 at L>=1); pointer-swap repair is exact ONLY for terminal/append,
   which prefix caching already does (so NO novel primitive).
3. An ACTIVATION CONDITION (design constraint): the latent cost becomes active IFF a harness performs
   interior context mutation. We ENUMERATE such patterns (editable scratchpads, RAG re-ranking that
   reorders docs mid-context, structured-memory rewriting, multi-agent shared-context editing,
   speculative rollback) as DESIGN-TIME WARNINGS — we do NOT claim these are emerging or prevalent
   (unsupported by our data); we claim only that IF adopted, they hit this measured cliff.

EXPLICITLY NOT CLAIMED: broad active real-world impact; that interior-mutating harnesses are coming;
any runtime primitive. Incidence in current harnesses is ~0% and reported as the first sentence.
- Contribution type: characterization / conditional cost-model / negative-result. Venue: MLSys.

## VOTE on the reframed T2. One line. GREEN iff the positioning (latent cost-map + activation
## condition, ~0% current incidence up front, no impact overclaim) matches the measured evidence.
T2: <GREEN|YELLOW|RED> — <=2 lines.
