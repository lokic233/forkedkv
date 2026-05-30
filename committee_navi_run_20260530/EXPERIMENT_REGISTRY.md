# EXPERIMENT REGISTRY (updated after E1, E2)

## E1 — T1 cross-vendor mapping ceiling (AMD MI350X). DONE (partial). VINDICATED holdout.
forkedkv CUDA ~520K mapping wall does NOT reproduce on ROCm (>=6.4M mappings, 0 fail).
=> T1 demoted to NVIDIA-scoped characterization. (MI350X now in maintenance; finish later.)

## E2 — T2 cross-engine invalidation penalty (SGLang, H100). DONE. FALSIFIED holdout's primary RED.
SGLang-1.5B: 1.30x->5.40x (4K->32K). SGLang-7B: 5.51x->8.34x->12.77x (8K->16K->32K).
vLLM-7B (edmm prior): 8.21x. At 16K, SGLang-7B (8.34x) ~ vLLM-7B (8.21x).
=> Pathology is ARCHITECTURAL, cross-engine, context-scaling. T2 workload-model half SETTLED.

## E3 (NEXT, HIGH EV) — T2 repair-subclass fraction on REAL agent traces. DEFINED, NOT RUN.
- Hypothesis: a non-trivial fraction of real agent tool-call edits are RoPE-INVARIANT (fixed-width
  / boundary-appended results) and thus repairable by VMM pointer-swap WITHOUT recompute.
- Why: this is the EXACT surviving flaw 3 agents converged on in Round 4. E2 measured the
  WORST case (position-shifting insertion). E3 measures the ADDRESSABLE fraction.
- What to run: parse real SWE-bench / tool-call traces; classify each KV-mutating edit as
  (a) append-at-boundary, (b) fixed-width interior replace (RoPE-invariant), (c) variable-length
  interior insert (RoPE-shifting). Report the distribution. Then measure repair TTFT on class (a)+(b)
  through a SGLang/vLLM hook.
- Min success (-> T2 GREEN candidate): >=20% of real edits in RoPE-invariant classes AND repair
  holds <=1.3x baseline on them. If <5%, the repair primitive is honestly near-vacuous (Muse Park
  right) and T2 ships as a workload-model-only paper.
- EV: HIGH — directly resolves the only surviving RED.

## E4 (deferred) — T1 AMD true ceiling (binary search) once MI350X out of maintenance.
## E5 (deferred) — T3 multi-harness divergence distributions.

## E3 — T2 edit-type taxonomy (RoPE-invariant subclass vacuous?). DONE. Killed R4 flaw.
SGLang live, per-class TTFT vs cache hit @16K: append 0.9-0.96x, fixed-width 1.3-2.4x (INVARIANT)
vs var-insert 2.99-8.21x, prepend 4.55-13.63x (SHIFTING). Cross-model. Repairable class NON-vacuous.

## E4 (NEXT) — IMPLEMENT the VMM pointer-remap repair + measure it as an intervention.
- Hypothesis: a real pointer-remap on the RoPE-invariant subclass recovers TTFT to <=1.3x C0 (vs
  the 2.4-8.2x recompute it replaces), measured end-to-end on a live engine, not inferred.
- Why: Muse Park + claude48 + codex all demand the repair be BUILT and measured as intervention.
  edmm has a vLLM-side pointer-swap (58us) but it's not wired to the RoPE-invariant edit classes
  from E3. E4 = port edmm's swap to the E_fixed/E_append classes and measure recovered TTFT.
- Min success (-> T2 GREEN candidate): measured repaired-TTFT <=1.3x C0 on E_fixed @7B, vs 2.38x
  unrepaired. If repair doesn't beat recompute, the primitive is honestly dead and T2 ships
  workload-model-only.

## E5 (NEXT) — real-trace edit-class FREQUENCY census.
- Needs real agent tool-call trajectories (SWE-agent/OpenHands runs). Classify each KV-mutating
  edit as append / fixed-width / variable-insert. Report distribution.
- Min success: report the distribution honestly; >=20% in repairable classes strengthens the
  primitive's addressable surface. (No on-disk traces; requires generating agent runs first.)
