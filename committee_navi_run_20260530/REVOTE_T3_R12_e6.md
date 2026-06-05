# ROUND 12 — T3 verdict after E6 (which FALSIFIED the intended T3 angle). Honest disposition vote.
# We are NOT trying to save T3. We want the correct call: is there a surviving GREEN-able T3, a
# YELLOW characterization, or should T3 be KILLED? Source of truth = measured data incl. the control.

## T3 (R3 form): "Agentic execution has a characterizable KV-divergence structure that determines
## memory-sharing opportunity; this divergence-distribution model should parameterize KV-sharing
## system design — distinct from RadixAttention's reactive token-prefix sharing."
## R3 votes were 4G/2Y (never RED). The 2 YELLOWs:
##  - claude48: "prior-assumes-static-prefixes is shaky; RadixAttention ALREADY does dynamic sharing"
##  - Muse Park: "model is a trivial 1-(tail/total), not a predictive law"

## E6 — the experiment built to ANSWER those YELLOWs (token-share OVERSTATES KV-share?):
Data: 12 REAL Claude Code agent branch pairs (from 33 real branch families), Qwen2.5-7B, 6 layers.
Measured per-layer KV common-prefix length vs token common-prefix length.
- 8/12 pairs: KVshare/tokshare = 1.000 (token match => KV match at all layers).
- 3/12 pairs APPEARED to collapse to ~0.05 -> looked like the desired "non-trivial divergence".
- *** DISCIPLINE CONTROL: ran an IDENTICAL 207-token prefix as (prefix-only) vs (prefix+800). For a
  truly identical causal prefix KV MUST be bit-identical, yet measured max|dK| up to 6.25e-2 >> 5e-3
  tol. The "collapse" was fp16 SDPA kernel nondeterminism (different reduction order at different
  total seq length), NOT semantic divergence. ***
=> With the artifact controlled: TOKEN-PREFIX SHARING PREDICTS KV SHARING. The "token-share
   overstates KV-share" delta does NOT exist at meaningful tolerance. RadixAttention is sound.

## HONEST IMPLICATION: E6 FALSIFIED T3's intended KV-vs-token angle and STRENGTHENED both YELLOWs.
The only surviving T3 form is "the distribution of token-prefix-shared fraction across real agent
branches" — which IS ~1-(divergence), exactly the "trivial" thing Muse Park flagged. No non-trivial
KV-level structure was found.

## VOTE the disposition of T3. One line:
T3: <GREEN|YELLOW|RED|KILL> — <=2 lines. (KILL = retire it; YELLOW = survives only as trivial
characterization not worth a top venue; GREEN only if you see a defensible non-trivial claim that
SURVIVES E6's negative result.)
