# E-B FINAL (3-engine complete, 45 rows) — Thesis B
## Penalty ∝ L^k at P=50% (cross-engine):
- vLLM 0.6.6:   k=0.674 R²=0.974  (3.43→12.67× over 4K→32K)
- SGLang 0.5.12: k=0.719 R²=0.998  (2.40→10.89×)
- HF transf 5.6: k=0.790 R²=0.973  (5.67→28.05×)
=> 3-engine penalty-exponent agreement within k=0.67–0.79 (tight). Cross-engine REPRODUCED. ✓
## Position dependence: P25/P75 penalty ratio 2.2–2.8× at L∈{16K,32K}, all 3 engines. Monotonic,
   stable. Penalty ≈ f(recomputed-token-count) — exactly as mechanism predicts. ✓
## THE CRUX (committee's kill test): absolute recompute TTFT exponent ≈1.3 (prior fit) << naive
   quadratic-attention k=2.0. SUB-quadratic. NO emergent system amplifier. The "it's just algebra"
   critique is largely vindicated — and the truth is even tamer than O(L²).
## HONEST VERDICT: B is now a STRONG cross-engine workload CHARACTERIZATION (3 engines, 5 lengths,
   3 positions, fitted laws, R²≥0.97) but NOT a super-quadratic discovery. It is operationally
   real (design agentic serving to avoid mid-prompt cache breaks) but the magnitude is explained
   by ordinary sub-quadratic prefill. Expected committee verdict: YELLOW (MLSys short/workshop),
   NOT a 6/6 GREEN top-tier thesis. Do not force it — this is the honest result.
