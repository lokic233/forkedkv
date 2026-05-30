# ROUND 1 CONVERGENCE MAP (6 agents, free generation)

Agents: CC48(opus4.8) CC47(opus4.7) CC46(opus4.6) Muse(Avocado/metacode) Codex(5.5) Gemini

## Existing-thesis votes (PART A)
- T1 kernel-transparent KV sharing: mostly YELLOW; Gemini RED. (thin vs vAttention)
- T2 VMM ceiling: YELLOW across the board (most backtest-robust; "section not paper")
- T3 EDMM spec-prefill: YELLOW→RED (recovery is oracle; Gemini RED)
- T4 counterfactual replay: YELLOW→RED (unbuilt; Gemini RED)

## NEW-thesis clusters (independent agents → same ideas = real convergence)

### CLUSTER A — VMM mapping-metadata ceiling characterization (K≈520K, driver-internal)
Proposed by: CC47-T3, Muse-T2, Codex-T3, Gemini-T6(GREEN), CC46-T8, CC48(in T3).
Anti-FlashInfer PASS. No-Code PASS. Measured: Metric4b ±1%, Lab1 392 VMA vs 67M.
Kill-shot: single driver/GPU; "section not paper." Gemini voted GREEN; Muse ranked #1/2.

### CLUSTER B — Tool-call prefix-injection superlinear cross-engine penalty (workload model)
Proposed by: CC48-T2(#1), CC47-T4, Muse-T5, Codex-T4, Gemini-T5, CC46-T5+T9.
Anti-FlashInfer PASS. Measured live 8.21×, vLLM+SGLang superlinear. CC48 & CC46 ranked #1.
Kill-shot: "cache invalidation is known" — blunted by superlinearity+magnitude+cross-engine.

### CLUSTER C — GPU CoW cost decomposition / negative result (HW loses to software)
Proposed by: CC48-T3, CC47-T3, Muse-T3, Codex-T5(GREEN), CC46-T8.
178µs/page, 13µs(7%) data, 93% metadata; B8 null (3% not 47%); vLLM-APC 700× faster.
Codex's single GREEN. Anti-FlashInfer PARTIAL (weakness). Negative-result venue (ATC/EuroSys).

### CLUSTER D — Cross-domain heterogeneous-granularity CoW (KV+RNG+tool-log+retrieval)
Proposed by: CC48-T1, CC47-T2(#1), Muse-T1, Codex-T1. Highest ceiling, but UNBUILT → YELLOW.
4-way ASPLOS debate named it "genuinely new." Conditional-GREEN if ≥2 non-KV domains built.

### CLUSTER E — Write-after-fork exact isolation as the defensible primitive
Proposed by: CC47-T1, Codex-T2, Muse-T4, CC48-T4. Metric5b/5c bit-identical. Anti-FlashInfer PASS.

## GREEN defenses in round 1 (who would defend what as GREEN)
- CC48: B only (workshop/MLSys-short). None at top-tier.
- CC47: NONE today; conditional-GREEN on D if cross-domain built.
- CC46: B only (as combined problem+model paper).
- Muse: C only (honest null result, falsifiable, no unbuilt deps).
- Codex: C only (cleanest evidence, fewest heroic assumptions).
- Gemini: A as GREEN (OSDI/SOSP); B,E as YELLOW.

## SYNTHESIS: the 3 most-converged + most-defensible = A, B, C (all characterizations/
## workload-models — "data speaks for itself", hardest to kill). D/E are higher-ceiling but
## unbuilt/narrower. KEY LEVER: we have an MI350X (devgpu499) → cross-VENDOR data lifts A
## from "single driver" to structural limit, and tests C's generality. That is the honest
## path to 6/6 GREEN, not badgering.
