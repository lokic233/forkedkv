# ROUND 8 — vote on the HONESTLY NARROWED T2. Full experiment trace below. Source of truth =
# measured experiments + repo artifacts. DO NOT reward overclaim. If the narrowed claim still
# over-reaches, RED/YELLOW it. We want the claim to match EXACTLY what the evidence supports.

## FULL EXPERIMENT TRACE (every result, including the self-defeating ones):
- E2 (SGLang live, RadixAttention): mid-prompt injection TTFT penalty B2/C0:
    1.5B: 1.30/2.17/3.29/5.40x @ 4K/8K/16K/32K ; 7B: 5.51/8.34/12.77x @ 8K/16K/32K.
    vLLM-7B (edmm prior) 8.21x. -> pathology is ARCHITECTURAL + cross-engine + context-scaling.
- E3 (SGLang live, per edit class vs clean cache hit C0):
    E_append (terminal) 0.90x(1.5B)/0.96x(7B) = FREE ;  E_fixed (interior, fixed-width) 1.31/2.38x ;
    E_varins (interior, var-len) 2.99/8.21x ;  E_prepend 4.55/13.63x.
    *** SELF-DEFEATING DATUM: E_append is ALREADY FREE on a stock prefix-caching engine. ***
- E4 (single-layer Qwen-7B): interior fixed-width repair looked exact + 18.8x. WRONG — see E4b.
- E4b (MULTI-layer Qwen-7B): interior edit corrupts SUFFIX K/V at every layer >=1 (max|dK| up to
    4.04); terminal edit leaves suffix bit-identical (0.0) at all layers. => interior repair NOT
    exact; E4's 18.8x interior number was INVALID. (caught by claude48+codex, confirmed by build.)
- E4c (MULTI-layer): terminal/append repair IS exact (argmax-match all S) + 5.4-81.6x vs FULL
    recompute. BUT R7 objection (claude48+Muse Park, CORRECT): a full-recompute baseline is a
    STRAWMAN — a prefix-caching engine already does incremental append cheaply (cf E3 E_append=0.90x).
    => the terminal "repair" is NOT a novel primitive beyond RadixAttention.

## HONEST CONCLUSION WE ARE ASKING YOU TO RATIFY (anti-overclaim):
T2's RUNTIME-PRIMITIVE claim is WITHDRAWN. Evidence shows: (a) terminal/append edits are already
handled for free by existing prefix caching (E3), and (b) interior edits are NOT cheaply+exactly
repairable by pointer-swap (E4b). There is no novel repair mechanism. We do NOT claim one.

## T2-NARROWED (workload-characterization ONLY):
One-sentence: "Agentic tool-call mid-prompt injection is an architectural cache-invalidation
pathology of prefix-hash KV caches that inflicts an 8-13x live-engine TTFT penalty reproducing
across vLLM and SGLang and growing with context; we provide a per-edit-class cost taxonomy
(append=free, fixed-width-interior=1.3-2.4x, variable-interior=3-13.6x) that quantifies which
agent edit patterns are cheap vs catastrophic, and we show the catastrophic class (interior
position-shifting edits) is NOT cheaply repairable at the KV level (suffix recompute is forced at
all layers >0) — i.e. a CHARACTERIZATION + IMPOSSIBILITY result, not a mechanism."
- Contribution type: workload-model + characterization (NO runtime-primitive claim).
- Closest prior work + delta: RadixAttention/vLLM prefix caching handles append; NO prior work
  quantifies the interior-edit penalty as an agent-workload pathology nor shows its KV-level
  irreparability. Anti-FlashInfer: YES (prefill recompute, not attention-kernel speed).
- Honest scope: real-trace FREQUENCY of each edit class (E5) is future work; the COST taxonomy and
  the impossibility result stand on measured data now.
- Venue: MLSys (workload/characterization track) or NSDI.

## VOTE on T2-NARROWED. Output EXACTLY one line. RED/YELLOW if it STILL overclaims or a real flaw
## survives; GREEN only if the claim now matches the evidence with no overreach.
T2: <GREEN|YELLOW|RED> — <=2 lines.
