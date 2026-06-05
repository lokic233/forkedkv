# ROUND 7 RE-VOTE on T2 — your Round-6 RED was CORRECT; the thesis is now corrected + rescoped.
# Measured experiments = source of truth. Vote hostile, one line.

## WHAT HAPPENED: In Round 6, claude48 + codex voted RED with an identical, correct objection:
"E4 proved KV equality only at LAYER 0; at L>0 the suffix attends to the edited interior slot, so
suffix K/V change -> pointer-stable suffix reuse is NOT exact for an interior edit, only terminal."
We BUILT the test (E4b) and YOU WERE RIGHT. The committee caught a real core-mechanism error.

## E4b (multi-layer, real Qwen2.5-7B, 6 layers) — confirms the RED:
INTERIOR fixed-width edit (slot at S/2): suffix max|dK| per layer = L0:0.0, L1:0.64, L2:2.54,
  L3:1.47, L4:2.78, L5:4.04  -> suffix K/V DIVERGE at every layer >=1 (L0=0 is why E4's
  layer-0-only test falsely passed).
TERMINAL edit (slot at end, no suffix): suffix max|dK| = 0.0 at ALL layers (nothing to corrupt).
=> Interior repair is NOT exact. The cheap-pointer-swap claim holds ONLY for terminal/append edits.

## T2 CORRECTED + RESCOPED:
Runtime-primitive claim is now: "a pointer-stable repair is EXACT for TERMINAL/APPEND tool-result
injection (model emits tool call -> result appended at the decode boundary -> generation continues
— the standard agent pattern), reusing cached prefix K/V at all layers and prefilling only the
appended tokens. Interior edits are NOT cheaply repairable (suffix recompute required); for those,
T2's contribution is the workload characterization only."

## E4c (multi-layer, real Qwen2.5-7B, 6 layers) — validates the RESCOPED repair:
| S | stock recompute (ms) | terminal repair (ms) | speedup | argmax-exact vs full recompute |
| 2048 | 43.7 | 8.1 | 5.4x | True |
| 4096 | 128.9 | 8.6 | 15.0x | True |
| 8192 | 436.6 | 12.1 | 36.1x | True |
| 16384 | 1533.3 | 18.8 | 81.6x | True |
=> Terminal/append repair is EXACT (argmax match at every scale, full multi-layer) and 5.4-81.6x
   faster, advantage growing with context (same signature as the E2/E3 pathology).

## FULL EVIDENCE CHAIN for T2:
- E2: pathology cross-engine (SGLang 12.77x @32K/7B ~ vLLM 8.21x) -> architectural, not a bug.
- E3: RoPE-invariant edit class is non-vacuous (append 0.9-0.96x vs shifting 3-13.6x).
- E4b: interior-edit repair is NOT exact (suffix corrupts at L>0) -> claim corrected.
- E4c: terminal/append repair IS exact + 5-82x at full depth.

## VOTE on the CORRECTED + RESCOPED T2. Output EXACTLY one line, hostile. If not GREEN, name the
## precise UNFIXED flaw that survives E2+E3+E4b+E4c (note: real-trace edit-frequency = E5 = scoped
## future work; judge whether the corrected mechanism + workload model clears your bar).
T2: <GREEN|YELLOW|RED> — <=2 lines.
