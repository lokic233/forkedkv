# Revision Notes — Round 3 (final push for 4×GREEN)

R2 verdict: 3×GREEN + 1 borderline-YELLOW (metacode). The three GREEN reviewers were
satisfied; metacode had **3 measurement-only asks** (no new mechanism). This round
addresses all three plus the two cheap P1s. No mechanism code changed except the one-file
fix that lets `QwenLayerN` load all 28 layers from all 4 safetensors shards.

---

## P0-1 — Full 28-layer depth validation  ✅ DONE, PASSED

**Ask:** prove the VMM CoW mechanism holds at the FULL model depth, not just a 4-layer subset.

**What we did.** `src/decode_layer.py:QwenLayerN` previously loaded only shard
`model-00001-of-00004.safetensors` (which holds layers 0–6), so it could not exceed 7 layers.
Fixed it to use the official `model.safetensors.index.json` weight map: each wanted tensor is
loaded from whichever shard holds it (the 28 layers span all 4 shards; embed in shard1,
model.norm in shard4). The 28-layer model loads in ~3 s using ~13.5 GiB. Added a `--out`
arg to `bench_metric5b_decode.py` so the N=28 run writes a separate CSV (keeps the N=4 result).

**Result — NO OOM at full depth; the mechanism holds.** N=28, 8 branches, 3,000-token
UNALIGNED prefix, 32 decode tokens each (`data/metric5b_decode_N28.csv`):

| N=28 | CoW fork | full clone | delta |
|------|----------|-----------|-------|
| peak live HBM | **1,120 MiB** | 2,016 MiB | **−44%** |
| KV bytes copied | **896 MiB (448 CoW events)** | 1,792 MiB | −50% |
| tok/s | 39 | 40 | ≈ parity |
| ALL 8 branches bit-identical CoW vs clone | — | — | **HARD ASSERT PASSED** |
| branch-0 decoded tokens | 32 unique (non-degenerate) | | |

Because the unaligned prefix forces real CoW writes, the N=28 run exercises BOTH full depth
AND the partial-page CoW write path (448 events across 28 layers × 8 branches × 2 K/V ranges).
This is the strongest possible version of the depth test. We also ran an aligned 2,048-token
N=28 (0 CoW events, peak CoW 1008 vs clone 1904 MiB, 8/8 bit-identical) earlier to confirm
load + decode before adding the CoW-write stress.

**Honest note:** tok/s (~39 at N=28) is an unoptimized Python reference loop, NOT a serving
number. This is a memory/correctness result. Documented in LIMITATIONS #3, #7, #12.

---

## P0-2 — Concurrency ceiling model  ✅ DONE (clean 1% fit)

**Ask:** sweep prefix sizes, record max_branches before OOM, fit max_branches ≈ K / prefix_pages,
report K.

**What we did.** New `bench/bench_metric4b_ceiling.py` sweeps prefix {1, 3, 6, 12 GiB} (512 /
1536 / 3072 / 6144 pages), each in a FRESH subprocess (a driver OOM can poison the context),
forking CoW branches until the driver OOMs. Output `data/metric4b_ceiling.csv`.

| prefix | pages (P) | max branches (B) | OOM call | live HBM | K = B×P |
|--------|-----------|------------------|----------|----------|---------|
| 1 GiB  | 512   | **1021** | cuMemSetAccess | 1.0 GiB | 522,752 |
| 3 GiB  | 1,536 | **339**  | cuMemSetAccess | 3.0 GiB | 520,704 |
| 6 GiB  | 3,072 | **169**  | cuMemSetAccess | 6.0 GiB | 519,168 |
| 12 GiB | 6,144 | **84**   | cuMemSetAccess | 12.0 GiB| 516,096 |

**K = branches × prefix_pages is constant to within 1%** (median **K ≈ 519,936 ≈ 520K**),
every point OOMs at the identical forensic call `cuMemSetAccess`, live HBM flat at one prefix.
The 12 GiB row reproduces R2's 84-branch Metric 4 result exactly.

> **Ceiling model: max_branches ≈ 520,000 / prefix_pages** — a quantified, predictable
> trade-off (the driver's per-device mapping-table capacity). Added to WRITEUP.md as a new
> Metric 4b section + a figure (`figures/metric4b_ceiling.png`).

---

## P0-3 — Partial-page CoW waste quantification  ✅ DONE (54%)

**Ask:** quantify wasted bytes from the 2 MiB CoW granularity on an unaligned prefix; add to
WRITEUP + CSV.

**What we did.** Added a P0-3 block to `bench_metric5b_decode.py` computing:
`wasted = bytes_copied − valid_tokens_in_copied_pages × kv_bytes_per_token_per_layer`,
with new CSV columns `wasted_bytes`, `waste_pct_of_copied`, `cow_bytes_pct_of_clone`.

For the 3,000-token prefix the tail page holds **952 of 2,048 tokens (46%)**, so each CoW
copies 2 MiB of which 54% is waste:

| config | CoW events | copied | valid | **wasted** | waste % | vs clone |
|--------|-----------|--------|-------|------------|---------|----------|
| N=4, 16 branches | 128 | 256 MiB | 119 MiB | **137 MiB** | **54%** | 50% of clone traffic |
| N=28, 8 branches | 448 | 896 MiB | 416.5 MiB | **480 MiB** | **54%** | 50% of clone traffic |

**Honest framing (in WRITEUP §Metric 5b P0-3):** 54% of the bytes copied on the boundary page
are partial-page waste — exactly what vLLM APC's 16-token blocks avoid. But the win survives:
even counting all waste, CoW copies only 50% of full-clone's total byte traffic, because clone
copies the entire multi-page prefix per branch while CoW only touches the one overwritten tail
page. Fully-filled interior prefix pages stay aliased. (Initial version had a buggy
"% of total prefix" denominator that read 584%; corrected to the meaningful "% of clone
traffic" comparison.)

---

## P1-1 — CUDA-graph safety note  ✅ DONE (LIMITATIONS #15)

Added a design note: CoW remaps (`cuMemUnmap`/`cuMemMap`/`cuMemSetAccess`) mutate the
VA→physical mapping and are NOT graph-legal inside a captured/replaying CUDA Graph. Correct
integration is to fork/remap at a branch boundary OUTSIDE graph capture; since a fork is an
explicit causal boundary (not per-token), the steady-state decode graph never remaps. Not
integrated with a graph-capturing engine — documented as a deployment constraint.

## P1-2 — Publish raw first-10 tokens  ✅ DONE

Added a `branch0_first10_tokens` row to both metric5b CSVs (non-degeneracy proof):
- N=4:  24276, 17079, 4031, 8596, 27476, 624, 13, 659, 101529, 18830
- N=28: 5696, 83053, 59431, 15512, 9248, 20465, 29880, 34106, 110479, 110911

---

## Files changed
- `src/decode_layer.py` — `QwenLayerN` loads all 28 layers via the safetensors weight map (was shard-1-only).
- `bench/bench_metric5b_decode.py` — `--out` arg; P0-3 waste computation + CSV columns; P1-2 raw tokens.
- `bench/bench_metric4b_ceiling.py` — NEW: P0-2 prefix-size sweep + K/P fit.
- `bench/make_figures.py` — NEW Metric 4b ceiling figure.
- `data/metric4b_ceiling.csv` — NEW (P0-2).
- `data/metric5b_decode_N28.csv` — NEW (P0-1 full depth).
- `data/metric5b_decode.csv` — regenerated with P0-3 waste columns + raw tokens.
- `figures/metric4b_ceiling.png` — NEW.
- `WRITEUP.md` — v0.4: TL;DR (ceiling model + full-depth + waste bullets), new Metric 4b
  section, Metric 5b rewritten for full depth + P0-3 waste subsection, closing updated.
- `LIMITATIONS.md` — v0.4: #3/#7/#8/#12 updated for full depth + ceiling model; NEW #14
  (partial-page waste), NEW #15 (CUDA-graph safety).

## No-overclaim discipline
- 28 layers did NOT OOM — reported the real numbers (no need to fall back to a lower N).
- Waste is 54% — reported as 54%, not minimized.
- tok/s ~39 at N=28 — flagged as unoptimized reference loop, not a serving claim.
- K fit is 1% spread — reported the per-point products, not just the median.

## Exit criteria — ALL MET
- [x] P0-1 full 28-layer depth (no OOM, 8/8 bit-identical, 448 CoW events at full depth)
- [x] P0-2 concurrency ceiling model (max_branches ≈ 520,000 / prefix_pages, 1% fit)
- [x] P0-3 partial-page waste (54%, in CSV + WRITEUP)
- [x] P1-1 CUDA-graph safety note
- [x] P1-2 raw first-10 tokens in CSV
- [x] WRITEUP.md + LIMITATIONS.md updated
- [x] REVISION_R3_NOTES.md written
- [x] Git commit
