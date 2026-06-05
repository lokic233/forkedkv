# ROUND 9 — FINAL vote on T2-NARROWED, with the EXACT two corrections the R8 YELLOWs (claude48,
# codex) required. No other changes. Source of truth = measured experiments. Reject any overclaim.

## TWO CORRECTIONS APPLIED (verbatim to the R8 asks):
1. (claude48) The 13.63x figure is E_PREPEND (prepend at position 0) — a NON-interior edit. It was
   wrongly folded into the interior-edit range. CORRECTED: interior var-len edit (E_varins) max =
   8.21x (7B). Prepend (13.63x) is reported SEPARATELY as a distinct, more-extreme class.
2. (claude48 + codex) "IMPOSSIBILITY result" is withdrawn as overclaim. RELABELED as: "an empirical
   quantification, on the tested transformer stacks (Qwen2.5 1.5B/7B, vLLM 0.6.6 + SGLang 0.5.12),
   of the known causal-attention constraint that makes prefix caching append-only — i.e. why an
   interior edit forces suffix recompute at layers >0 (E4b: suffix max|dK| up to 4.04 at L>=1)."
   We claim quantification of a known KV constraint, NOT a new impossibility theorem.

## T2-NARROWED (final wording):
"Agentic tool-call mid-prompt injection is an architectural cache-invalidation pathology of
prefix-hash KV caches, inflicting an 8-13x live-engine TTFT penalty that reproduces across vLLM
(8.21x) and SGLang (12.77x @32K/7B) and grows with context (E2). We give a per-edit-class cost
taxonomy on live engines (E3): terminal append = FREE (0.90-0.96x, already handled by prefix
caching); interior fixed-width = 1.3-2.4x; interior variable-length = 3.0-8.2x; whole-context
prepend = up to 13.6x (reported as a separate, most-extreme class). We empirically quantify, on the
tested stacks, WHY interior edits cannot be cheaply repaired at the KV level (E4b: an interior edit
changes suffix K/V at every layer >=1, so pointer-swap repair is not exact; multi-layer confirmed).
NO runtime-primitive is claimed — terminal append is already free under existing prefix caching,
and interior repair is shown not cheaply achievable. Contribution = workload-model + characterization."
- Closest prior work + delta: RadixAttention/vLLM handle append; NO prior work quantifies the
  interior-edit agent-workload penalty or its per-class cost structure. Anti-FlashInfer: YES.
- Honest scope: real-trace FREQUENCY of edit classes (E5) = future work; cost taxonomy + the
  layer>0 suffix-divergence quantification stand on measured data NOW. Venue: MLSys / NSDI.

## VOTE. One line. GREEN only if the claim now matches evidence with zero overreach.
T2: <GREEN|YELLOW|RED> — <=2 lines.
