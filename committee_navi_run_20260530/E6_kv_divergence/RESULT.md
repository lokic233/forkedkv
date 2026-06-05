# E6 — T3 probe: does token-prefix sharing OVERSTATE true KV-sharing in real agent branches?
# RESULT: intended T3 angle FALSIFIED by a discipline check. Reported honestly.

## Intended claim (to beat RadixAttention + answer the "trivial" YELLOWs):
"Token-prefix match (what RadixAttention shares on) overstates the truly-reusable KV fraction by a
layer-dependent amount -> an a-priori KV-divergence model beats reactive token matching."

## Data: 12 REAL agent branch pairs (sibling Claude Code sessions sharing a 200-char opening prefix,
from 33 real branch families incl. 137-sibling and sub-agent forks). Qwen2.5-7B, 6 layers, tol 5e-3.

## Raw result looked promising then SUSPICIOUS:
8/12 pairs: KVshare/tokshare = 1.000 (token-share fully predicts KV-share).
3/12 pairs (8,9,10): ratio collapsed to 0.005-0.05 (KV diverged far earlier than tokens).
A 0.05 ratio would have been the headline "non-trivial divergence" result.

## DISCIPLINE CHECK (control experiment) — the collapse is a NUMERICAL ARTIFACT, not real:
Ran an IDENTICAL 207-token prefix through the model as (A) prefix-only vs (B) prefix+800 tokens.
For a truly identical causal prefix, KV MUST be bit-identical. Measured max|dK| over the shared
prefix: L0=6.25e-2, L3=6.25e-2, L5=1.2e-2 — ALL far above the 5e-3 tol. Cause: fp16 SDPA uses
different kernels/reduction order for different total sequence lengths, so the SAME prefix yields
slightly different KV when the sequence around it differs in length. The E6 "divergence" in pairs
8/9/10 is this kernel nondeterminism, NOT semantic KV divergence.

## HONEST CONCLUSION — this experiment FALSIFIED the intended T3 angle:
1. With the artifact controlled, token-prefix sharing DOES predict KV sharing (ratio ~1.0 where
   tokens match). RadixAttention's token-based sharing is SOUND. The "token-share overstates
   KV-share" delta does NOT exist at fp16-meaningful tolerance.
2. This STRENGTHENS the R3 YELLOW objections (claude48: "RadixAttention already does dynamic
   sharing"; Muse Park: "model is trivial"). T3's KV-vs-token-divergence angle is DEAD.
3. T3's surviving (weaker) form: the DISTRIBUTION of token-prefix-shared fraction across real agent
   branches (a workload characterization) — but that IS essentially 1-(divergence), exactly the
   "trivial" thing Muse Park flagged. No non-trivial KV structure was found.

## RECOMMENDATION: do NOT push T3 to GREEN on this evidence. The honest finding is a NEGATIVE result
## (token-sharing suffices; no KV-level a-priori model needed). Report it as such, or pivot T3.
