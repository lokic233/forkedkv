# E8 — Layer-stratified positional KV divergence of REPEATED content (new angle, H100)

## Phenomenon: agents re-encounter identical content (re-read files, repeated scaffolding) at
## DIFFERENT absolute positions. RadixAttention can't reuse that KV (position differs). We measure
## WHY, decomposed by cause, to bound any position-shifted reuse scheme.

## Method: same W=48-token chunk placed at two positions p1, p2 in one Qwen2.5-7B context. Measure
## per-layer divergence of that chunk's KV between p1 and p2. Separate PRE-RoPE K (token+context
## only, no position rotation) from post-RoPE; V is never RoPE'd. 8 trials, median.

## RESULT (median over 8 trials)
| layer | max|dK_preRoPE| | max|dV| |
|-------|-----------------|---------|
| 0     | 0.0000          | 0.0000  |   <- bit-identical: token-only, no context mixing yet
| 1     | 0.93            | 0.26    |
| 2     | 3.14            | 0.91    |
| 5     | 6.15            | 2.42    |
| 9     | 4.13            | 3.58    |
| 15    | 5.32            | 4.01    |
| 21    | 5.99            | 6.25    |
| 27    | 6.44            | 15.60   |   <- large: deep KV is dominated by surrounding context
post-RoPE K at layer 0 differs by ~11.6 (pure RoPE rotation) but PRE-RoPE K = 0 -> that component
is exactly recoverable by de-rotate/re-rotate.

## FINDING (measured, decomposed — the candidate thesis backbone):
The cross-position reusability of repeated content's KV is LAYER-STRATIFIED into two causes:
1. RECOVERABLE positional component: at shallow layers (esp. L0), same-token KV is position-
   independent BEFORE RoPE (pre-RoPE K and V bit-identical at L0). The post-RoPE difference is a
   deterministic rotation -> a position-shifted reuse (de-RoPE cached K, re-RoPE at new position;
   reuse V directly) is EXACT at L0 and near-exact at very shallow layers.
2. IRRECOVERABLE context component: at depth, KV mixes in the (different) surrounding context via
   attention, so divergence grows monotonically (V 0 -> 15.6 by L27). No positional transform
   recovers this; the content's deep KV genuinely depends on its neighbors.

## WHY IT'S NOT TRIVIAL / NOT ALREADY DONE:
- RadixAttention/prefix caching share KV ONLY for contiguous identical prefixes from position 0;
  they NEVER attempt cross-position reuse of repeated interior content. This decomposition shows
  the OPPORTUNITY (shallow layers) and the HARD LIMIT (deep layers) quantitatively.
- It is NOT "1-(tail/total)": it is a layer-resolved, cause-separated measurement.

## HONEST CAVEATS (anti-overclaim, to test next before any GREEN):
- E5 showed exact-repeat incidence in real coding-agent traces is LOW (line-dup median 1.5%, tail
  30%). So real-world IMPACT is uncertain (same latent-vs-active issue as T2). Must measure incidence.
- "Exact at L0 only" may be too shallow to yield end-to-end savings (most compute is deep layers).
  Whether shallow-layer reuse saves meaningful FLOPs is UNMEASURED.
- Single model (Qwen2.5-7B). Single chunk width. Needs robustness sweep.

## E9 — GREEN-gate experiment (defined by committee R13: claude47+claude46)
Measure max prefill-FLOP saving = real-incidence x (contiguous shallow layers reusable within eps).
RESULT (8 trials, median): at eps in {0.01, 0.05, 0.10}, reusable contiguous shallow layers = 1/28
(4%). At E5's 1.5% real exact-repeat incidence -> max prefill-FLOP saving CEILING = 0.054%.
GREEN bar was >=10%, KILL bar <2%. 0.054% << 2%.
=> T4 mechanism is KILLED by its own gate experiment. Only layer 0 is reusably-shared (and that is
partly a RoPE tautology per claude48). The characterization is real but has ~zero actionable impact.
HONEST OUTCOME: T4 is a clean negative-result / bounded-characterization, NOT a GREEN thesis.
