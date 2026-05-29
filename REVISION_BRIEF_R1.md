# Revision Brief — Round 1

You built v0.1 of "Forkable GPU Memory for Replayable Agent Execution" on cli:devgpu014. The committee (codex/gemini/claude/metacode) ALL gave YELLOW with realistic ASPLOS acceptance probability ~15-30%. They cited revisions that move it to ~40-50%. Your job: implement these revisions to get the prototype to 4-of-4 GREEN (or the closest honest version of that).

You retain the same NO-OVERCLAIM DISCIPLINE from v0.1. Surface null results. Cite every number. Better honest than inflated.

## Workspace

- `/home/dengcchi/branchable_replay/` — your existing prototype, head `4a013a2`
- venv at `.venv/`, python 3.12, torch 2.11+cu128, cuda-python
- 8× H100 available; you used device 0 in v0.1, free to use any
- Git is initialized — commit each round atomically

## What 4 reviewers said (synthesized)

### What they liked (don't break these)
- Real CUDA VMM mechanism (cuMemCreate/cuMemMap/cuMemUnmap), not os.fork
- Honest self-reporting — null results clearly stated
- Capacity story (6→64 branches) is the genuine contribution
- All numbers in WRITEUP.md match their CSVs
- Citations correctly differentiate from vAttention ASPLOS'25, ChunkAttention ASPLOS'24, etc.

### What MUST be fixed (4-of-4 reviewers asked)

#### P0-A: Real auto-regressive token decode loop in Metric 5
Current Metric 5 fills KV pages synthetically. ASPLOS will reject "End-to-End" framing without actual token generation.

What to build: a single-layer attention decode loop using a real model (Qwen2.5-7B or Llama-3.1-8B is fine; even pulling just one transformer layer's weights is acceptable). Run autoregressive decode for 64-256 tokens per branch, with the KV pages backed by your VMM CoW manager. Compare against full-clone baseline.

Key files: `bench/bench_metric5_e2e_trajectories.py`. New: `src/decode_layer.py` (a thin wrapper around one HF attention layer that reads K/V from the branch manager).

What to measure (minimum): tokens-per-second per branch, peak HBM, total bytes copied across N branches × 100 tokens of decode. Compare CoW vs full-clone.

Effort estimate from committee: 2-3 days.

#### P0-B: Rename "End-to-End" → "Macro-benchmark"
WRITEUP.md uses "End-to-End" for Metric 5, which 3 reviewers called overclaim. Even after P0-A above, frame Metric 5 honestly. If P0-A succeeds, you can label it "End-to-End single-layer decode" and Metric 5 then has both an autoregressive component (new) and a memory-mechanism component (existing). Both are honestly named.

Effort: 5 minutes (and propagate through writeup).

#### P0-C: Dynamic VA growth — the structural blocker
Right now `vmm_pool.py` reserves all VA upfront. Real agents grow KV as decode proceeds. Reviewers flagged this as "structurally incapable of dynamic generation."

What to build: extend `vmm_pool.py` so a branch can `append_page()` to its tail. Keep CoW semantics correct. Add a unit test that grows two branches independently after fork.

Effort: 1-2 days.

#### P0-D: Tail-divergence model in Metric 2b
Currently `bench_metric2b_divergence.py:29` writes the FIRST n pages. Real agents diverge at the TAIL. Switch to writing the last n pages and re-run.

Effort: 1 day (with rerun).

### Hidden bugs the v0.1 builder did NOT flag (committee found these — must fix)

#### B1: Bytes-written formula contradiction (claude)
`WRITEUP.md` line 99: formula says "prefix_bytes × fanout" but the data shows 2× that. Either fix formula to "2 × prefix_bytes × fanout" OR redefine metric to count only `bytes_copied`. Internal consistency must be restored.

Effort: 30 min.

#### B2: Metric 3 +7.7% at seqlen=8192 is GPU clock drift, not real overhead (claude, gemini)
Increase warmup from current value to ≥50 reps. Better: interleave contiguous and VMM measurements within each rep so clock drift cancels. Re-run and update `data/metric3_attn_overhead.csv`. If real overhead at 8192 is still high, REPORT THAT honestly — but if it's clock artifact, fix the methodology.

Effort: 2-4 hours.

#### B3: Metric 2b effective divergence rounds to 6.25%, not 5%
`int(round(0.05 * 32)) = 2`, so "5% divergence" is actually 6.25%. Fix in two ways: (a) use a prefix size where 5% is an integer, OR (b) report effective divergence in CSV header and in WRITEUP.md.

Effort: 30 min.

#### B4: `hash()` in `replay.py:70-78` is not reproducible across processes (codex)
Replace with stable hash (`hashlib.blake2b` or similar). Determinism matters for replay correctness.

Effort: 1 hour.

#### B5: `_cow` has 2 extra map ops via temp scratch VA (LIMITATIONS.md #2 — already known but unmeasured)
Add a microbenchmark `bench/bench_cow_overhead.py` that times the temp-VA-map vs the actual D2D copy. If the overhead is real, document it. Don't claim the optimization unless you implement it.

Effort: 1 day.

### Strongly recommended (3-of-4 asked)

#### P1-A: Run Metric 4 to true OOM
Currently the cap is 64 (hard-coded MAX_BRANCHES at `bench/bench_metric4_capacity.py:20`). Sweep to 128, 256 until CoW actually OOMs. Report the true ceiling.

Effort: 2-3 hours of GPU time.

#### P1-B: vLLM integration sketch (Option B framing)
You did not patch vLLM (correctly — too much surgery). But ASPLOS reviewers will ask "deployability?" Two paths:
- Option A (heavyweight): actually patch vLLM block manager. 1-2 weeks.
- Option B (lightweight, RECOMMENDED): write a 2-3 page design sketch in `prototype_status.md` describing the changes needed to vLLM block manager, scheduler, and the engineering effort. Cite vAttention's standalone-prototype path.

Effort for Option B: 1 day.

#### P1-C: Expand SWE-Bench instances 7 → 50+
Current N=7 is too thin. Pull 50-100 representative instances spanning the size distribution, re-run metric 5.

Effort: 4-8 hours.

## Engineering decisions you must commit to early

1. Which model for the real decode loop in P0-A? Recommendation: Qwen2.5-7B (smaller download, fits on 1 H100 even with full layer weights). Document the choice in `prototype_status.md`.
2. How many decode layers? For v0.2, ONE layer is sufficient. Real point is to prove the CoW pages support real attention computation.
3. Do you implement P1-B Option A (real vLLM patch) or Option B (design sketch)? Recommendation: B for v0.2; A is a v0.3 / camera-ready-revision target.

## Deliverables for v0.2 in `~/branchable_replay/`

Update or create:
- `src/decode_layer.py` — single-layer attention decode wrapper (NEW)
- `src/vmm_pool.py` — add `append_page()` API
- `src/replay.py` — replace Python `hash()`
- `bench/bench_metric2b_divergence.py` — switch to tail writes
- `bench/bench_metric3_attn_overhead.py` — fix warmup methodology
- `bench/bench_metric4_capacity.py` — remove 64 cap, sweep to OOM
- `bench/bench_metric5_e2e_trajectories.py` — add real decode loop
- `bench/bench_cow_overhead.py` — NEW microbenchmark for B5
- `bench/bench_metric4b_capacity_divergence.py` — NEW (claude asked)
- `WRITEUP.md` — fix formula B1, rename Metric 5, propagate revisions
- `LIMITATIONS.md` — update with B5 measured overhead, dynamic VA fix
- `prototype_status.md` — document vLLM design sketch (P1-B Option B)
- `data/*.csv` — regenerate every metric whose script changed
- `figures/*.png` — regenerate
- `REVISION_R1_NOTES.md` — what you actually did this round, what you skipped, why

## Exit criteria

You stop and report when EITHER:
- ALL P0 + B1-B4 done + P1-A or P1-B done, all measurements rerun, WRITEUP.md updated
- A hard blocker prevents progress (document it)

Time budget for this round: 4-12 hours of wall clock. The user has authorized full Path A push for ASPLOS.

DO NOT:
- Do partial work and call it done. The committee will catch any gap.
- Add new features the committee didn't ask for.
- Tune to make numbers look better — keep null results visible.
- Touch cli:devgpu499.

REPORT BACK with:
1. What you completed per item (P0-A through P1-C, B1-B5)
2. New numbers from rerun metrics
3. New top concerns you have for the next committee review
4. Estimated additional effort if committee asks for another round

GO.
