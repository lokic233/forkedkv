# Parallel Labs Execution Plan
## Created: 2026-05-30 ~05:45 UTC
## Rollback point: tag `snapshot-pre-parallel-labs` (commit 3028567)

## Track A: Lab 1 R2 (targeting Claude/metacode YELLOW→GREEN on OVERALL)
- GPU allocation: devices 0-3
- Builder model: Claude 4.8
- Goal: Address specific upgrade conditions from Claude+metacode:
  1. Workload where kernel-transparent VA produces measurable end-to-end advantage
  2. Multi-GPU/driver characterization breadth
  3. CUDA Graph integration PoC
  4. Real serving scenario demonstrating the advantage
- Output: bench/lab1r2_*, data/lab1r2_*, LAB1_R2_NOTES.md
- Success: committee gives 4×GREEN overall (zero YELLOW)

## Track B: Labs 2+3 (ncu counters + kernel comparison)
- GPU allocation: devices 4-7
- Builder model: Claude 4.8
- Goal: Raw hardware evidence + kernel throughput delta
  - Lab 2: ncu L2/bandwidth metrics (contiguous only; document VMM incompatibility)
  - Lab 3: SDPA vs paged-attention Triton kernel throughput
- Output: bench/bench_lab2_*, bench/bench_lab3_*, data/lab2_*, data/lab3_*, LAB2_LAB3_NOTES.md
- Success: SDPA measurably outperforms paged-attention (the one delta that matters)

## Audit discipline:
- Every measurement logged to audit_traces/run_log.jsonl
- Every commit includes full reproduction instructions
- Committee receives ALL code + data (no scoped reviews)
- Zero YELLOW tolerance

## Rollback:
- If either track corrupts shared state: `git checkout snapshot-pre-parallel-labs`
- Tracks work on separate files (lab1r2_* vs lab2_*/lab3_*) to avoid merge conflicts
