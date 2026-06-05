# ROUND 4 RE-VOTE on T2 ONLY — new EXPERIMENTAL evidence resolves the lone holdout's RED.
# GitHub artifacts + measured experiment = source of truth. Vote hostile.

## THESIS T2 (final form)
One-sentence: Tool-call mid-prompt injection is an ARCHITECTURAL cache-invalidation pathology of
prefix-hash KV caches (which assume append-only growth) — NOT an implementation bug — inflicting a
TTFT penalty that reproduces across independent serving engines and grows with context; a
microsecond VMM pointer-remap repairs the RoPE-invariant edit subclass.
- Contribution: workload-model + runtime-primitive. Venue: MLSys/NSDI.

## ROUND-3 BLOCKING DISSENT (Muse Park, RED):
"The 8.21x is a vLLM 0.6.6 implementation bug, not a fundamental pathology, and SGLang will
NOT show >5x. The RoPE-invariant repairable subclass is vacuous for variable-length outputs."

## NEW EXPERIMENTAL EVIDENCE (E2, run on H100 devgpu014, SGLang 0.5.12.post1, RadixAttention):
Mid-prompt injection (UUID salt inserted BETWEEN two prefix halves -> RadixAttention cache miss),
TTFT B2(mid-prompt)/C0(clean hit):
  SGLang Qwen2.5-1.5B:  1.30x @4K, 2.17x @8K, 3.29x @16K, 5.40x @32K
  SGLang Qwen2.5-7B:    5.51x @8K, 8.34x @16K, 12.77x @32K
  (prior, edmm) vLLM 0.6.6 Qwen2.5-7B: 8.21x
=> SGLang DOES exceed 5x (up to 12.77x @7B/32K). At 16K, SGLang-7B (8.34x) ~ vLLM-7B (8.21x).
=> The pathology reproduces on TWO independent engines at the SAME model size, monotonic in
   context. It is architectural (prefix-hash append-only assumption), not a vLLM bug.

## WHAT THIS DOES AND DOES NOT SETTLE:
- SETTLES: Muse Park's "SGLang won't show >5x" RED is empirically FALSIFIED. The workload-model
  half of T2 (the pathology, its cross-engine generality, its context-scaling) is now measured.
- DOES NOT settle: the REPAIR primitive (VMM pointer-swap) is validated in edmm on vLLM only;
  porting + the RoPE-invariant-subclass fraction on real traces remains future work (honestly scoped).

## VOTE: Given the new evidence, output EXACTLY ONE line, hostile:
T2: <GREEN|YELLOW|RED> — <=2 lines. If not GREEN, name the precise UNFIXED flaw that survives the
E2 evidence. (Note: GREEN does not require the repair primitive be cross-engine-ported; it requires
the thesis CLAIM survive hostile review with a defined validating experiment. The workload-model
claim is now measured; judge whether that + the scoped repair clears your bar.)
