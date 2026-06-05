# ROUND 5 RE-VOTE on T2 — new experiment E3 directly tests the surviving Round-4 flaw.
# Measured experiment = source of truth. Vote hostile, one line.

## THESIS T2 (final form)
Tool-call mid-prompt injection is an ARCHITECTURAL prefix-hash cache-invalidation pathology (NOT
an engine bug), reproducing cross-engine and growing with context; a microsecond VMM pointer-remap
repairs the RoPE-INVARIANT edit subclass. Contribution: workload-model + runtime-primitive. Venue MLSys/NSDI.

## ROUND-4 SURVIVING FLAW (Muse Park RED; claude48 + gemini YELLOW, all converged):
"The injection E2 measured is POSITION-SHIFTING = the RoPE-NON-invariant subclass the pointer-swap
can't repair. The repairable RoPE-INVARIANT subclass may be VACUOUS, so the repair half is
unvalidated 'snake oil' for variable-length tool outputs."

## NEW EVIDENCE — E3 edit-type taxonomy (live SGLang RadixAttention, per-class TTFT vs clean hit C0):
Qwen2.5-1.5B @16K:  E_append(invariant) 0.90x | E_fixed(invariant) 1.31x | E_varins(shifting) 2.99x | E_prepend(shifting) 4.55x
Qwen2.5-7B  @16K:  E_append(invariant) 0.96x | E_fixed(invariant) 2.38x | E_varins(shifting) 8.21x | E_prepend(shifting) 13.63x
(E_varins@7B = 8.21x reproduces edmm's original vLLM number on a different engine.)

INTERPRETATION:
- The RoPE-INVARIANT subclass is NOT vacuous: appended tool results are FREE (0.90-0.96x, within
  noise of a cache hit) and fixed-width interior replaces are 1.3-2.4x — vs 3-13.6x for
  position-shifting edits. The cost cleanly TRACKS the RoPE boundary, cross-model.
- This is the "position-shift taxonomy with per-class recompute cost" claude48 explicitly asked for.

WHAT E3 PROVES vs DOES NOT:
- PROVES: the repairable class exists, is well-defined, and is 3-13x cheaper -> repair target is real.
- DOES NOT PROVE: the FREQUENCY of each edit class in REAL agent tool-call traces (no trajectory
  data on disk; honest future work for camera-ready). E3 settles the COST structure, not the
  real-world incidence.

## VOTE: output EXACTLY one line, hostile. If not GREEN, name the precise UNFIXED flaw that survives
## BOTH E2 (pathology cross-engine) AND E3 (repairable subclass non-vacuous, cost tracks RoPE).
T2: <GREEN|YELLOW|RED> — <=2 lines.
