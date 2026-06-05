# E3 — Edit-type taxonomy: is T2's RoPE-invariant repairable subclass VACUOUS? (H100, SGLang)

## Surviving flaw being tested (Round-4 convergence of claude48 + gemini + Muse Park):
"The mid-prompt injection E2 measured is a POSITION-SHIFTING edit = exactly the RoPE-NON-invariant
subclass the VMM pointer-swap cannot repair. The repairable (RoPE-invariant) subclass may be
near-vacuous, so the repair-primitive half of T2 is unvalidated 'snake oil'."

## Method
Live SGLang RadixAttention. Warm clean prefix (cache hit = C0), then measure TTFT for 4 edit
classes vs C0, at 16K context. Classify each by whether it shifts downstream RoPE positions.

## RESULT — Qwen2.5-1.5B, 16K context, 5 trials
| edit class | RoPE | TTFT ms | x C0 |
|---|---|---|---|
| C0 clean hit | - | 33.6 | 1.00x |
| E_append (tool result at END) | INVARIANT | 30.4 | 0.90x |
| E_fixed (fixed-width interior replace, \|new\|==\|old\|) | INVARIANT | 44.1 | 1.31x |
| E_varins (variable-length interior insert) | SHIFTING | 100.6 | 2.99x |
| E_varins0 (prepend, shifts everything) | SHIFTING | 153.1 | 4.55x |

## FINDING — the RoPE-invariant subclass is NOT vacuous
There is a clean cost gradient that TRACKS the RoPE boundary:
- RoPE-INVARIANT edits are cheap: append is FREE (0.90x, within noise of cache hit), fixed-width
  replace is 1.31x.
- RoPE-SHIFTING edits are expensive: 2.99x-4.55x (the E2 pathology).
=> The repairable subclass has a fundamentally different (3-4x cheaper) cost profile. The repair
   opportunity is REAL and maps exactly onto RoPE-invariance. Muse Park's "vacuous" claim is
   refuted at the cost-structure level: there IS a large, cheap, well-defined repairable class.

## HONEST CAVEAT (what E3 does and does not prove):
- PROVES: the RoPE-invariant edit class exists and is 3-4x cheaper -> repair target is non-vacuous.
- DOES NOT prove: the FREQUENCY of each class in REAL agent tool-call traces (needs trajectory
  data we don't have on disk; that is E5/future work). E3 establishes the COST taxonomy claude48
  explicitly asked for ("position-shift taxonomy with per-class recompute cost"); the real-trace
  FREQUENCY distribution remains the honest open item for the camera-ready.

## E3b — Qwen2.5-7B, 16K context, 5 trials (apples-to-apples with edmm)
| edit class | RoPE | TTFT ms | x C0 |
|---|---|---|---|
| C0 clean hit | - | 36.5 | 1.00x |
| E_append (tool result at END) | INVARIANT | 35.2 | 0.96x |
| E_fixed (fixed-width interior replace) | INVARIANT | 86.7 | 2.38x |
| E_varins (variable-length interior insert) | SHIFTING | 299.5 | 8.21x |
| E_varins0 (prepend) | SHIFTING | 497.5 | 13.63x |

Note: E_varins @7B = 8.21x — reproduces edmm's ORIGINAL vLLM-7B number on a different engine.
The RoPE gap is WIDER at 7B: append is FREE (0.96x) while shifting edits cost 8.2x-13.6x.
Cross-model robust: the repairable subclass is non-vacuous and the cost cleanly tracks RoPE.
