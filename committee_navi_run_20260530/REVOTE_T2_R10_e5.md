# ROUND 10 — does T2 (the 6/6 GREEN workload-model) SURVIVE the E5 incidence finding?
# E5 is a self-administered hostile test of our OWN green thesis. Vote honestly; demote if E5
# undermines it. Source of truth = measured data.

## RECAP: T2-GREEN (R9, 6/6) claims:
mid-prompt tool injection is an architectural prefix-hash cache-invalidation pathology, 8-13x
TTFT penalty cross-engine (E2), per-edit-class cost taxonomy (E3: append free / interior 3-8.2x),
interior edits not cheaply KV-repairable (E4b). Workload-model + characterization. No primitive.

## NEW EVIDENCE — E5 (real agent-trace census, 325 Claude Code sessions, 198MB, 12,380 turns):
- ~99.9% of real context mutations are APPEND (tool result/attachment/reminder at TAIL) -> FREE
  under existing prefix caching (E3 0.90-0.96x). Verified: 6510 tool_use ~ 6508 tool_result, all
  tail-adjacent; no tool result is spliced before cached tokens.
- ~0.1% RESET (compaction: full prefix replaced by summary — new cache, not interior edit).
- 0% INTERIOR splice observed. => The 8-13x pathology T2 measured is NOT triggered by today's
  dominant append-only coding-agent loop. The penalty is REAL but its real-world INCIDENCE in
  current harnesses is ~zero (LATENT, not active).
- Where it WOULD bite (not measured): interior-mutating harnesses — editable scratchpads, RAG
  re-ranking that reorders docs mid-context, structured-memory rewriting, multi-agent shared-context
  editing, speculative rollback. EDMM's original 8.21x came from a vLLM path that re-injected tool
  output mid-prompt; the Claude Code harness avoids this by appending.

## THE HONEST QUESTION FOR YOU:
Does T2 survive as a GREEN workload-model+characterization thesis given that its headline pathology
has ~0% incidence in mainstream append-only agents? Options you may vote:
- GREEN if: the characterization + the "latent cost that activates iff harness interior-mutates"
  framing is a legitimate, correctly-scoped contribution (an architectural cost map + a condition
  for when it bites), AND we report the ~0% current-harness incidence honestly rather than implying
  broad impact.
- YELLOW/RED if: with ~0% incidence in real agents, T2 is a characterization of a NON-PROBLEM (a
  penalty nobody hits), or it now requires a NEW claim (that interior-mutating harnesses are coming/
  valuable) that is unsupported by measured data.

## VOTE. One line. Judge whether the GREEN survives E5 with the incidence caveat REPORTED, or must
## be demoted for characterizing a latent/non-active pathology.
T2: <GREEN|YELLOW|RED> — <=2 lines.
