# ROUND 13 — NEW thesis candidate T4, with MEASURED backbone (E8) + self-identified weaknesses.
# Do NOT reward overclaim. Vote GREEN/YELLOW/RED/KILL. Source of truth = measured E8 + E5 incidence.

## T4 — "Layer-stratified positional reusability of repeated KV"
One-sentence: The KV of identical repeated content placed at different absolute positions is
REUSABLE in a LAYER-STRATIFIED way — at shallow layers the only position-dependence is a
deterministic RoPE rotation (pre-RoPE K and V are position-INDEPENDENT; measured bit-identical at
layer 0), so cross-position KV reuse is exact there; at depth, attention mixes in the differing
surrounding context, making KV divergence irrecoverable (grows monotonically to large values by the
last layer) — a quantified decomposition that bounds any position-shifted KV-sharing scheme and
that contiguous-prefix sharing (RadixAttention) structurally cannot exploit.
- Contribution type: characterization (KV structure) + a bounded mechanism opportunity.

## MEASURED EVIDENCE (E8, Qwen2.5-7B, 8 trials, same 48-tok chunk at 2 positions, median):
- pre-RoPE K and V at LAYER 0: max diff = 0.0000 (bit-identical regardless of position).
- post-RoPE K at L0: ~11.6 (pure recoverable rotation).
- V divergence by layer: L1 0.26 -> L5 2.4 -> L15 4.0 -> L27 15.6 (monotone; context-mixing).
- pre-RoPE K divergence by layer: L0 0 -> L2 3.1 -> L27 6.4.

## CLOSEST PRIOR WORK + DELTA:
RadixAttention/vLLM/ChunkAttention share KV for CONTIGUOUS identical prefixes from position 0 only;
they never reuse repeated INTERIOR content at a shifted position. CacheBlend/position-independent
caching (if reviewers cite it) reuses chunk KV but reports accuracy loss; T4's delta is the
LAYER-RESOLVED, CAUSE-SEPARATED quantification (recoverable RoPE vs irrecoverable context-mix) that
says EXACTLY which layers a positional-reuse scheme can be exact at and where it must recompute.

## SELF-IDENTIFIED WEAKNESSES (judge these honestly — they may be fatal):
1. INCIDENCE: E5 found exact-repeat in real coding-agent traces is LOW (line-dup median 1.5%, tail
   30%). Real-world impact may be marginal (same latent-vs-active risk that capped T2).
2. SHALLOW-ONLY EXACTNESS: reuse is exact only at L0/very shallow layers; most FLOPs are in deep
   layers where reuse is NOT exact. Whether shallow reuse saves meaningful compute is UNMEASURED.
3. PRIOR ART RISK: position-independent / blended KV caching may already cover the mechanism;
   T4 may be only a characterization on top of it.
4. Single model, single chunk width so far.

## ANTI-FLASHINFER: YES (about KV reuse/memory structure, not attention-kernel speed).

## VOTE T4. One line. KILL if a weakness is fatal; YELLOW if only a bounded characterization;
## GREEN only if you see a defensible non-trivial contribution that survives the weaknesses + needs
## a NAMED next experiment to confirm (state it).
T4: <GREEN|YELLOW|RED|KILL> — <=2 lines (+ if GREEN, the one experiment that would confirm it).
