# E4b — Multi-layer suffix-divergence test: the claude48/codex Round-6 RED is CONFIRMED

## The objection (claude48 + codex, R6, independently):
"E4 proved KV equality only at LAYER 0. In a full causal transformer, a changed mid-prefix slot
changes suffix hidden states -> suffix K/V at layers >0 also change. So 'recompute only the W slot
tokens, reuse suffix K/V pointer-stably' is NOT exact for an INTERIOR edit — only for a TERMINAL
edit (slot at end, no suffix)."

## Method
REAL Qwen2.5-7B first 6 layers (QwenLayerN), full multi-layer prefill. Fixed-width W=64 slot,
same-width value change (RoPE positions unchanged). Compare per-layer SUFFIX K/V (tokens after the
slot) between original and edited, for INTERIOR (OFF=S/2) vs TERMINAL (OFF=S-W) slot. S=2048.

## RESULT
INTERIOR EDIT (OFF=1024, suffix=[1088..2048]) — suffix max|dK| per layer:
  L0: 0.0  | L1: 0.64 | L2: 2.54 | L3: 1.47 | L4: 2.78 | L5: 4.04   <- DIVERGES at every L>=1
TERMINAL EDIT (OFF=1984, suffix=[2048..2048]) — suffix max|dK| per layer:
  L0..L5: ALL 0.0   <- bit-identical at every layer (no suffix to corrupt)

## VERDICT — the RED is CORRECT. E4's "pointer-stable suffix reuse" claim was WRONG for interior edits.
- Layer 0 suffix is unchanged (the slot's value doesn't enter the suffix's K/V at L0) — which is
  EXACTLY why E4's layer-0-only test falsely showed correctness.
- At L>=1 the suffix attends to the edited slot, so suffix hidden states (and thus suffix K/V)
  change. Reusing cached suffix K/V after an INTERIOR edit is NOT exact. claude48 + codex caught a
  genuine core-mechanism error that the single-layer E4 masked. The hostile committee did its job.

## CORRECTED SCOPE OF T2's REPAIR PRIMITIVE:
- TERMINAL / APPEND edits (slot at sequence end, no suffix): repair is EXACT and pointer-stable at
  full depth. Suffix K/V trivially valid (there is none). This is the E_append class from E3
  (measured at 0.90-0.96x = free).
- INTERIOR fixed-width edits: NOT repairable by pure pointer-swap; the suffix MUST be recomputed at
  L>=1. The cheap-repair claim does NOT hold here. E4's 18.8x@32K interior-edit number is INVALID
  as stated (it skipped the suffix recompute that correctness requires).

## NET: T2's runtime-primitive must be RESCOPED to terminal/append tool-result injection (e.g.
## results appended at the live decode boundary — the common agent pattern), where it is exact.
## For interior edits the contribution is the workload characterization, not a free repair.
