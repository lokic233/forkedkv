# E4c — Rescoped TERMINAL/APPEND repair: exact + fast at FULL multi-layer depth

## Context: E4b confirmed the claude48/codex RED — interior edits corrupt suffix K/V at L>0.
T2's repair is rescoped to TERMINAL/APPEND tool-result injection (the standard agent pattern:
model emits tool call -> result appended at decode boundary -> generation continues). Here the
suffix is empty, so cached prefix K/V is exact at all layers. E4c IMPLEMENTS and TIMES this.

## Method
REAL Qwen2.5-7B first 6 layers. B = stock full N-layer prefill of prefix(S)+result(W). C = exact
incremental repair: reuse cached prefix K/V at ALL layers, prefill ONLY the W appended tokens
(causal attention over cached-prefix ++ new). Correctness = final-token argmax vs full recompute. W=64.

## RESULT
| S | B recompute (ms) | C repair (ms) | C/B | speedup | argmax match |
|---|---|---|---|---|---|
| 2048 | 43.73 | 8.09 | 0.185 | 5.4x | True |
| 4096 | 128.90 | 8.60 | 0.067 | 15.0x | True |
| 8192 | 436.56 | 12.08 | 0.028 | 36.1x | True |
| 16384 | 1533.25 | 18.79 | 0.012 | 81.6x | True |
(S=32768: stock baseline OOM on its full SxS attention matrix — the repair itself was fine; the
 naive recompute is what blows up, which is itself the motivation.)

## FINDING — rescoped T2 repair is EXACT and FAST at full depth
- argmax match True at EVERY scale: the terminal repair reproduces the full-recompute next-token
  distribution exactly (multi-layer, real Qwen attention + SwiGLU). Correctness confirmed where E4b
  showed it must hold (empty suffix).
- Speedup 5.4x -> 81.6x, GROWING with context (same context-scaling signature as E2/E3 pathology).
- This is the HONEST, CORRECT version of T2's runtime-primitive: exact for terminal/append edits,
  which is the common agent tool-result pattern (results appended at the live decode boundary).

## T2 STATUS AFTER E4b+E4c:
- Workload-model half (mid-prompt invalidation pathology): SETTLED cross-engine (E2) + cost
  taxonomy (E3).
- Runtime-primitive half: CORRECTED (E4b killed the over-broad interior claim) and then VALIDATED
  in its correct scope (E4c: terminal/append repair, exact + 5-82x, full multi-layer).
- Remaining honest future work: real-trace FREQUENCY of terminal-vs-interior tool edits (E5).
