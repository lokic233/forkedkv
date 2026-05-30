# Revision Brief — Round 2

R1 was a major improvement: 4-of-4 YELLOW → **3 strong YELLOWs (borderline GREEN) + 1 GREEN**.
ASPLOS acceptance probability: 15-30% (v0.1) → **35-45% median (one outlier at 65-75%)**.
Every reviewer says "one focused R2 lands ASPLOS" — they all converge on the same revisions.

Goal of R2: get to **4-of-4 GREEN**. The asks are concrete and bounded.

You retain the NO-OVERCLAIM DISCIPLINE. Surface null results. Cite every number.

## Workspace
- /home/dengcchi/branchable_replay/  (head a1756f7, R1 complete, working tree clean)
- venv at .venv/, all infra working
- Qwen2.5-7B already downloaded (~/branchable_replay/.venv cache)

## What 4 reviewers said (synthesized)

### What they liked (don't break these)
- The R1 mechanism is real (CUDA VMM, correct CoW, refcount, dynamic VA via append_page)
- Honest self-reporting maintained (null results, parity, "VA limit not data" surprise)
- Bit-identical CoW vs clone tokens (when actually verified)
- 84-branch ceiling = scientifically interesting (capacity 14× over clone)
- Metric 3 +0.05% overhead (clean honest result)
- B5 microbench is the kind of evidence reviewers love

### What MUST be fixed for 4-of-4 GREEN

#### P0-1: Multi-layer decode (2-4 layers) — addresses C1
**Asked by all 4 reviewers.** Single-layer is correctly flagged as a toy. Extending to 2-4 layers proves the mechanism composes.

What to build:
- Extend src/decode_layer.py: a QwenLayerN class loading layers 0..N-1 (suggested N=4)
- Each layer maintains its own per-branch K/V pages (multiple BranchKV instances per branch, one per layer)
- Inter-layer residual connections + intermediate hidden state flow through the stack
- bench/bench_metric5b_decode.py: take --num-layers 4 flag; report per-layer KV page count + total HBM
- Show tokens are no longer degenerate (even if repetitive, not all identical to one fixed point)

Effort: 2-3 days.

#### P0-2: VA pooling — turn the 84-branch metadata ceiling into the true data ceiling — addresses C2
**Asked by 3-of-4.** This is the gemini-65% reviewer's load-bearing item. The 84 branches use only 12 of 97 GiB; the limit is mapping metadata. A VA pool/freelist would push capacity dramatically.

What to build:
- src/vmm_pool.py: add a free-list of reserved VA ranges. On branch destroy, return the VA range to the pool. On new branch creation, prefer reuse from the pool.
- Optionally: free physical handles when refcount drops to 0 (already done) AND pool the unmap'd VA.
- Re-run bench_metric4_capacity.py with pooling — measure new ceiling.
- Instrument WHICH cuda call is failing at the OOM (per gemini R2-3 ask) — add per-call-site error annotation in vmm_pool.py.

Goal: turn "84 branches, VA-limited" into "X00+ branches, data-limited" OR "84 branches, here's exactly which CUDA call returns CUDA_ERROR_OUT_OF_MEMORY". Either is publishable.

Effort: 3-4 days (3 for pool, 0.5 for instrumentation, 0.5 for sweep).

#### P0-3: Harden bit-identical correctness assertion — addresses gap codex+claude+gemini caught
**Asked by 3-of-4.** The "bit-identical CoW vs clone" claim in v0.2 is currently a print, not an assert, and only checks branch-0.

What to fix:
- bench/bench_metric5b_decode.py: replace the print at line 193-196 with `assert match, f"branch {b} mismatched at token {first_mismatch}"` 
- Loop over ALL branches, not just branch 0
- Write to data/metric5b_decode.csv: tokens_match (per-branch), token_checksum_cow, token_checksum_clone, first_mismatch_index_or_-1

Effort: 0.5 days.

#### P0-4: CoW-on-write decode stress case — addresses C3 / codex top concern
**Asked by 3-of-4.** Current Metric 5b is append-only — branches never overwrite shared prefix pages, so CoW never fires (0 bytes copied). The mechanism's hot path isn't tested in the headline.

What to build:
- New bench/bench_metric5c_decode_cow_write.py — deliberately overwrite a shared prefix page mid-decode (simulating a speculative edit / tree-of-thought rollback)
- Verify _cow() fires (refcount went 2→1 on parent's copy)
- Verify only one page is copied (not the whole prefix)
- Verify the originating branch's tokens are NOT corrupted, the writing branch's tokens are different
- Report bytes copied per branch

Effort: 1 day.

### Hidden bugs / weakening claims (must fix)

#### B6: Metric 5b uses page-aligned prefix → "0 bytes copied" is artifact
gemini and claude caught this. With prefix=4096 = exactly 2 pages of 2048 tokens, and decode always appending past end, no CoW fires. Real prefixes aren't page-aligned.

Fix: Re-run Metric 5b with prefix_tokens=4000 (unaligned). Report the partial-page CoW cost. Update WRITEUP "0 bytes copied" to be honest about the page-alignment caveat.

Effort: 1 hour.

#### B7: B5 docstring lies about CUDA events
claude caught this. bench/bench_cow_overhead.py:23 says "CUDA-event + perf_counter" but only uses perf_counter.

Fix: either add CUDA events properly, or correct docstring. Recommendation: just fix docstring (perf_counter is fine for this measurement).

Effort: 5 minutes.

#### B8: "47% removable" claim is unsubstantiated until you remove it
3-of-4 reviewers flagged this. Claim says scratch-VA bookkeeping is 47% of CoW cost AND removable. The "removable" half is hypothesis until built.

Fix: implement reusable scratch VA pool in KVBranchManager._cow(). Pre-reserve N scratch VA pages at manager init; recycle them. Re-run bench/bench_cow_overhead.py and bench_metric1_fork_latency.py and bench_metric5b_decode.py with the optimization.

Goal: drop full CoW from 175.7µs to ~93µs (= 175.7 - 82.2). Turn the latency parity story into a latency win story. Eliminates the biggest reviewer attack.

Effort: 1-2 days.

### Strongly recommended (1-2 reviewers asked)

#### P1-A: Expand SWE-bench from N=7 to N=20-30 — addresses C4 / R0 P1-C still open
**Asked by codex + claude + metacode.** Skipped in R1, must land in R2.

Fix: pull 13-23 more representative SWE-Bench-Verified instances spanning the size distribution. Re-run bench/bench_metric5_e2e_trajectories.py.

Effort: 1 day.

#### P1-B: Compare against vLLM APC / block-table prefix sharing
**Asked by codex.** The strongest baseline isn't "full clone" — it's vLLM's existing prefix-sharing. Without comparison, the result reads "we beat the worst baseline."

Fix: add a section in WRITEUP discussing vLLM APC / block-table sharing. If time permits: a small toy benchmark using a vLLM block manager simulation. Otherwise: an analytic comparison citing vLLM's design + why VMM CoW gives strictly more (write-after-share semantics, dynamic forking, not just read-only prefix dedup).

Effort: 2-5 days for analytic+toy; longer for real vLLM patch. **Recommendation: analytic-only for R2** (1 day).

## R2 engineering decisions to commit early
1. Multi-layer decode: how many layers (4 recommended)? How are layers chained — full residual+MLP or attention-only? **Recommendation:** N=4, attention-only per layer (so the comparison stays focused on KV management, not MLP perf).
2. VA pooling design: per-branch reuse, or process-wide pool? **Recommendation:** process-wide free-list keyed by (size_in_pages).
3. Scratch-VA pool size: how many concurrent CoW operations to support? **Recommendation:** start with 8, document.

Document each in prototype_status.md "R2 decisions" section before implementing.

## Deliverables for v0.3 in ~/branchable_replay/

Update or create:
- src/decode_layer.py — extend to multi-layer (P0-1)
- src/vmm_pool.py — add VA pool freelist + per-call-site error instrumentation (P0-2 + gemini R2-3)
- src/kv_branch_manager.py — wire VA pool, add scratch-VA pool for _cow (P0-2 + B8)
- bench/bench_metric5b_decode.py — multi-layer support, hard correctness assert ALL branches, unaligned prefix (P0-1 + P0-3 + B6)
- bench/bench_metric5c_decode_cow_write.py — NEW: CoW-on-write stress (P0-4)
- bench/bench_metric4_capacity.py — re-run with VA pooling (P0-2)
- bench/bench_cow_overhead.py — re-measure post-optimization (B8)
- bench/bench_metric1_fork_latency.py — re-run (B8 may improve fork latency)
- bench/bench_metric5_e2e_trajectories.py — N=20-30 (P1-A)
- WRITEUP.md — update everything; add P1-B vLLM-APC comparison section
- LIMITATIONS.md — update; remove items that R2 fixes
- prototype_status.md — R2 decisions, vLLM updated
- REVISION_R2_NOTES.md — what you did, what you skipped, why

## Exit criteria

You stop and report when EITHER:
- All P0 (4 items) + B6+B7+B8 done, ≥1 P1 done, all measurements rerun, WRITEUP updated
- A hard blocker prevents progress (document it)

Time budget: 6-16 hours wall clock for R2. The user is committed to ASPLOS submission. R2 is the round where we land 4×GREEN.

DO NOT:
- Do partial work; the committee will catch any gap
- Add new features the committee didn't ask for (focus on the 7 items above)
- Touch cli:devgpu499
- Hide null results

REPORT BACK FORMAT (same as R1):
1. What completed (per item, DONE / PARTIAL / SKIPPED with reason)
2. New numbers from rerun metrics — include v0.1, v0.2, v0.3 columns where comparable
3. Top concerns for R3 committee (if any)
4. Effort estimate if R3 is needed

GO.
