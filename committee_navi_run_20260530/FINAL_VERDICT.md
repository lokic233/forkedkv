# FINAL VERDICT — Navi 6-agent hostile committee (2026-05-30) — UNANIMOUS GREEN REACHED

## T2 (NARROWED + CORRECTED): 6/6 GREEN, Round 9. The first and only thesis to clear unanimous
## hostile review — and it cleared it by WITHDRAWING its overclaims, not by inflating evidence.

### The GREEN thesis (exact, evidence-bounded):
"Agentic tool-call mid-prompt injection is an architectural cache-invalidation pathology of
prefix-hash KV caches, inflicting an 8-13x live-engine TTFT penalty that reproduces across vLLM
(8.21x) and SGLang (12.77x @32K/7B) and grows with context. Per-edit-class cost taxonomy on live
engines: terminal append = FREE (0.90-0.96x, already handled by prefix caching); interior
fixed-width = 1.3-2.4x; interior variable-length = 3.0-8.2x; whole-context prepend = up to 13.6x
(separate most-extreme class). We empirically quantify, on the tested stacks (Qwen2.5 1.5B/7B,
vLLM 0.6.6 + SGLang 0.5.12), WHY interior edits cannot be cheaply repaired at the KV level (an
interior edit changes suffix K/V at every layer >=1; pointer-swap repair is not exact). NO
runtime-primitive is claimed. Contribution = workload-model + characterization. Venue: MLSys/NSDI."

### How it got to GREEN (9 rounds, 5 experiments — the loop working as designed):
R1-R3 ideation+refinement -> T2 emerged as strongest (5G/1R).
E2  (SGLang)  killed R3 RED "it's a vLLM bug"           -> cross-engine 12.77x.
E3  (taxonomy) killed R4 RED "repairable class vacuous" -> append free vs shifting 3-13.6x.
E4  (1-layer)  looked exact+18.8x -> Muse Park flipped GREEN R6.
E4b (multi-layer) CONFIRMED claude48+codex bug: interior repair NOT exact (suffix corrupts L>0).
E4c (multi-layer) terminal repair exact+5-82x BUT R7: "that's just RadixAttention append (free)".
=> Runtime-primitive WITHDRAWN (our own E3 E_append=0.90x proved it adds nothing over prefix cache).
R8 narrowed -> 4G/2Y on 2 copy-fixes. R9 fixes applied -> 6G/0Y/0R UNANIMOUS.

### DISCIPLINE NOTE: every RED was resolved by EXPERIMENT or by NARROWING THE CLAIM, never by the
### agents lowering their bar. The committee caught (a) a cross-engine generality gap, (b) a vacuity
### question, (c) a multi-layer CORRECTNESS BUG, (d) a novelty-vs-prior-art collapse, and (e) two
### specific overclaims. T2 is GREEN because it now claims EXACTLY what 5 experiments measured.

## OTHER THESES (honest status, NOT forced):
- T1 (mapping-table resource): YELLOW. E1 showed CUDA ~520K ceiling does NOT reproduce on AMD
  (4M+ mappings) -> demoted to NVIDIA-scoped characterization. Real but narrower.
- T3 (agentic divergence workload-model): YELLOW. No RED, but core empirical claim needs real
  agent-trace divergence distributions (E5) that don't exist on disk. Forcing GREEN now = overclaim.
  Honestly held at YELLOW pending traces.

## NEXT (real future work, not vote-rounds):
E5: generate real agent trajectories (SWE-agent/OpenHands) -> measure edit-class FREQUENCY (sizes
T2's real-world impact) AND per-step KV divergence distribution (could move T3 toward GREEN).

## UPDATE (post-E5, Rounds 10-11): GREEN re-confirmed after self-administered incidence audit.
E5 (325 real Claude Code agent sessions, 12,380 turns): mid-prompt invalidation has ~0% incidence
in mainstream APPEND-ONLY harnesses (99.9% append=free, 0.1% compaction-reset, 0% interior splice).
This BROKE the R9 GREEN (R10: 4G/2Y). T2 was reframed (R11) to a CONDITIONAL ARCHITECTURAL COST-MAP
+ DESIGN CONSTRAINT, with the ~0% current incidence as the HEADLINE sentence and explicit non-claims
on prevalence/impact/emergence. R11: 6/6 GREEN.

FINAL HONEST T2 = a measured cost map (append free / interior 1.3-8.2x / prepend 13.6x / cross-engine
8-13x) + structural impossibility-of-cheap-interior-repair (E4b) + an ACTIVATION CONDITION (binds iff
a harness interior-mutates: scratchpads, RAG reorder, shared-context editing, rollback). It is a
negative-result / design-constraint contribution, NOT a "widespread crisis" claim. Venue: MLSys.

This is the strongest possible HONEST outcome: a thesis that survived 11 rounds, 5 experiments, a
caught correctness bug, a primitive collapse, AND its own real-world incidence audit — by claiming
exactly, and only, what the data supports.
