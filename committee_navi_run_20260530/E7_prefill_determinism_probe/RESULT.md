# E7 — Pre-thesis probe: does fp16 prefix-cache/chunked prefill change MODEL OUTPUTS? (H100)
# Motivation: E6's control found cached-prefix KV != recompute KV under fp16 (max|dK| 6.25e-2).
# Question: does that ever flip a generated token (a reproducibility finding vs a deployed opt)?

## Probes (Qwen2.5-7B, 28 layers, real attention, greedy):
1. Identical input x40: argmax flips = 0/40 (deterministic at fixed seq len).
2. Chunked (2-pass) vs single-pass prefill, random inputs x60: argmax flips = 0/60.
   median top1-top2 logit margin = 0.582.
3. Real prompt, 60-token greedy generation, single vs chunked prefill: 0/60 divergent steps,
   byte-identical output.

## VERDICT: NEGATIVE. The fp16 KV numerical difference from chunked/cached prefill does NOT flip
## greedy tokens on Qwen2.5-7B/H100. A "prefix caching silently changes outputs" thesis is NOT
## supported by measurement. Do NOT pursue it. (Self-caught before proposing to committee.)

## Side finding for honesty: E5's compaction-pathology angle also weak — only 8 compaction events
## and 15 system-reminder injections across 325 sessions (median 16 turns); compaction is rare,
## not a frequent pathology. No thesis there either.
