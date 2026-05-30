# Revision Brief — Round 3 (final push for 4×GREEN)

R2 verdict: 3×GREEN + 1 borderline-YELLOW (metacode). The 3 GREEN reviewers are satisfied.
Metacode's 3 remaining asks are ALL measurement-only (no new mechanism code needed):

## P0-1: Full 28-layer depth validation
- Run bench_metric5b_decode.py with --num_layers 28 --prefix_tokens 2048 --decode_tokens 32 --branches 8
  (reduce prefix+decode+branches to fit memory for full model)
- If OOM, reduce further until it runs. Document whatever config works.
- Report: peak HBM CoW vs clone, bytes copied, tok/s, bit-identical assert (must pass)
- The point: prove the VMM CoW mechanism holds at full model depth (not just 4 of 28 layers)
- Effort: load all 28 layer weights (they're all in the same safetensors shard), update QwenLayerN to accept N=28
- KEY: if it OOMs at N=28 even with small prefix, report that honestly and show which N is max

## P0-2: Concurrency ceiling model
- In bench_metric4_capacity.py, sweep prefix sizes: 1 GiB, 3 GiB, 6 GiB, 12 GiB
- For each, record max_branches before CoW OOM
- Fit: max_branches ≈ K / prefix_pages (where K is the driver's mapping-table limit)
- Report K (expect ~500K total mappings based on 84 branches × 6144 pages = 515K)
- Add to WRITEUP.md: "The ceiling is predictable: max_branches ≈ 515K / prefix_pages"
- This turns a measured limit into a quantified, predictable trade-off

## P0-3: Partial-page CoW waste quantification
- In the existing metric5b data (3000-token unaligned prefix), compute:
  wasted_bytes = bytes_copied - (valid_tokens_in_copied_pages × kv_bytes_per_token_per_layer)
- For 3000-token prefix: last page has 952 valid tokens out of 2048 capacity
  → CoW copies 2 MiB but ~952/2048 = 46% is real data, 54% is waste for that page
  → But TOTAL waste is small because only 1 of 2 prefix pages gets CoW'd
- Report this in WRITEUP.md + add to CSV as extra column
- The point: be transparent about granularity overhead before a reviewer catches it

## P1 (nice-to-have):
- P1-1: CUDA-graph safety note in LIMITATIONS.md (30 min — just text)
- P1-2: Publish raw first-10 tokens of branch-0 in metric5b CSV for non-degeneracy proof (30 min)

## Engineering notes
- All 28 layers of Qwen2.5-7B are in the same safetensors shard (model-00001-of-00004.safetensors has layers 0-6; need to check remaining shards for layers 7-27)
- Loading 28 layers' attention weights alone: 28 × (q+k+v+o) × (~100MB each) ≈ 11 GB. Should fit one H100 with room for KV.
- The existing QwenLayerN class accepts arbitrary N — just need to ensure it loads from multiple shards if needed
- MLP weights add ~14GB more. Total model ≈ 25GB loaded + KV headroom. Doable on 97GB H100.

## Exit criteria
- P0-1 done (even if max feasible N < 28, report honestly what N works)
- P0-2 done (concurrency model with fit equation)
- P0-3 done (waste % in CSV + WRITEUP)
- WRITEUP.md + LIMITATIONS.md updated
- REVISION_R3_NOTES.md written
- Git commit

## NO-OVERCLAIM: same discipline as always. If 28 layers OOMs, say so. If waste is 54%, say 54%.

Time budget: 4-8 hours. This is the final push.
GO.
