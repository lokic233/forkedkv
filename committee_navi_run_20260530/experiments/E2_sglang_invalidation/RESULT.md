# E2 — T2 cross-engine: does SGLang ALSO show the tool-injection penalty? (H100, devgpu014)

## Muse Park's T2 RED being tested:
"The 8.21x is a vLLM 0.6.6 implementation bug, not a fundamental pathology, and SGLang will
NOT show >5x."

## Setup
- Engine: SGLang 0.5.12.post1 (RadixAttention), torch 2.11.0+cu130, FlashInfer 0.6.11.
- Model: Qwen2.5-1.5B-Instruct (/tmp/qwen15b), H100 device 4, mem_fraction_static=0.5.
- Harness: edmm/sglang-integration/test_sglang_baseline.py (pre-existing from prior edmm work).
- C0 = clean prefix-cache HIT (RadixAttention best case).
- B2 = mid-prompt contamination: a unique UUID salt inserted BETWEEN two prefix halves,
  i.e. EXACTLY the tool-call mid-prompt injection pattern. Forces RadixAttention cache miss.
- TTFT (max_new_tokens=1), 3 trials, context sweep 4K/8K/16K/32K.

## RESULT (B2/C0 TTFT penalty)
| Context | C0 hit (ms) | B2 mid-prompt (ms) | Penalty |
|---------|------------|--------------------|---------|
|   4,096 |       17.9 |               23.2 |  1.30x  |
|   8,192 |       20.0 |               43.5 |  2.17x  |
|  16,384 |       30.9 |              101.7 |  3.29x  |
|  32,768 |       53.7 |              289.8 |  **5.40x** |

## VERDICT IMPACT — Muse Park's T2 RED is FALSIFIED
SGLang DOES exhibit the mid-prompt invalidation penalty, and it is NOT a vLLM-specific bug:
it is the SAME architectural cause (prefix-hash chain assumes append-only growth; an interior
insertion invalidates the whole downstream chain -> recompute). The penalty SCALES with context
length and reaches **5.40x at 32K** — clearing the >5x bar Muse Park said SGLang "will not show."

Two engines (vLLM 0.6.6: 8.21x @ 7B; SGLang 0.5.12: 5.40x @ 32K/1.5B) with independent
RadixAttention/prefix-hash implementations BOTH show the pathology => it is ARCHITECTURAL, not
an implementation bug. This is the T2 workload-model core claim, now cross-engine validated.

## HONEST CAVEATS
- edmm's 8.21x was Qwen2.5-7B; this E2 used 1.5B (the harness default / model on disk). Penalty
  is context-length dependent (1.3x@4K -> 5.4x@32K). A like-for-like 7B SGLang run would likely
  show a HIGHER ratio (bigger recompute). Pending: re-run E2 on 7B for apples-to-apples.
- Penalty is the INVALIDATION/RECOMPUTE pathology. The REPAIR primitive (T2's second half,
  VMM pointer-swap) is validated in edmm on vLLM, not yet ported to SGLang. T2's workload-model
  half is now strong cross-engine; the repair-on-SGLang is future work, honestly scoped.

## E2b — APPLES-TO-APPLES: SGLang on Qwen2.5-7B-Instruct (same model size as edmm's vLLM 8.21x)
H100 device 5, SGLang 0.5.12.post1, mem_fraction_static=0.8, same C0/B2 harness.
| Context | C0 hit (ms) | B2 mid-prompt (ms) | Penalty |
|---------|------------|--------------------|---------|
|   8,192 |       23.5 |              129.2 |  5.51x  |
|  16,384 |       34.7 |              289.6 |  8.34x  |
|  32,768 |       56.0 |              714.7 |  **12.77x** |

At 16K, SGLang-7B's 8.34x MATCHES edmm's vLLM-7B 8.21x almost exactly — and at 32K it reaches
12.77x. The mid-prompt invalidation penalty is now confirmed on TWO independent engines at the
SAME model size, monotonically increasing with context. Muse Park's "SGLang won't show >5x"
RED is decisively falsified (5.51x at the SMALLEST scale tested; 12.77x at 32K).

## SUMMARY OF T2 EVIDENCE (cross-engine, cross-model)
| Engine | Model | Penalty | Source |
|--------|-------|---------|--------|
| vLLM 0.6.6 | Qwen2.5-7B | 8.21x | edmm (prior) |
| SGLang 0.5.12 | Qwen2.5-1.5B | 1.30x->5.40x (4K->32K) | E2 |
| SGLang 0.5.12 | Qwen2.5-7B | 5.51x->12.77x (8K->32K) | E2b |
The pathology is ARCHITECTURAL (prefix-hash append-only assumption), reproduces across engines
and model sizes, and grows with context. T2's workload-model core is empirically settled.
