# Lab 1 R1 — Revision Notes (addressing committee RED + YELLOW)

**Date:** 2026-05-30
**Trigger:** Committee verdict on `REVISION_BRIEF_LAB1_R1.md`:
- codex: YELLOW (phrasing too strong)
- claude: GREEN with YELLOW qualification (positive attribution circumstantial)
- gemini: **RED** (3 fundamental positioning attacks)
- metacode: GREEN → YELLOW

This revision targets 4×GREEN by addressing **all four** P0 items in the brief.

---

## What changed

### P0-1 — Software prefix-sharing baseline (addresses Gemini Attack 1: "strawman baseline")

**New code:**
- `src/baseline_prefix_sharing.py` — vLLM-APC-equivalent block-table allocator with
  refcounted physical "blocks." Fork = list-copy + refcount++ per block. Write to a
  shared block = block-granularity CoW (alloc fresh block, simulate copy, repoint,
  decref). Mirrors vLLM v0.6 PagedAttention / APC mechanics.
- `bench/bench_software_baseline.py` — head-to-head: software vs hardware on
  (M1) fork latency, (M2) CoW granularity, (M3) capacity.
- `bench/make_baseline_compare_figure.py` — figures.

**New data:**
- `data/baseline_compare_m1_fork_latency.csv` — fork latency vs prefix length
- `data/baseline_compare_m2_cow_granularity.csv` — bytes per single-token CoW write
- `data/baseline_compare_m3_capacity.csv` — max branches at fixed prefix

**New figures:**
- `figures/baseline_compare_m1_fork_latency.png`
- `figures/baseline_compare_m3_capacity.png`

**Empirical result (the RED-attack-resolving truth):**
| Axis              | Software (vLLM-APC-style) | Hardware (ForkedKV) | Winner   |
|-------------------|---------------------------|---------------------|---------|
| Fork latency      | 0.5–9.3 µs                | 65–6588 µs          | Software (~700×) |
| CoW granularity   | 2 MiB (16 tokens)         | 2 MiB (16 tokens)   | Tie at our sizing; software finer in production |
| Capacity at 32-block prefix | 100,000+ branches | ~16,250 (driver ceiling) | Software (~6×) |
| Kernel compatibility | Requires PagedAttention | Standard FlashAttn/SDPA OOB | **Hardware** |

This is intellectually honest. Software wins on the headline practical metrics.
The contribution must therefore PIVOT to:
1. **Architectural characterization of GPU VMM** (the Lab 1 result, the K≈520K
   ceiling, the cuMemSetAccess attribution, the VA-pool reuse model — useful
   regardless of whether anyone deploys this mechanism).
2. **Kernel-transparent contiguous VA** (the only mechanism-level differentiator
   that survives Gemini's empirical attack — Metric 3 ≈ 0% overhead vs needing
   a paged-attention kernel for software prefix sharing).

### P0-2 — Honest positioning paragraph in WRITEUP

Updated TL;DR (§ "TL;DR — honest, R4 repositioned") and added new section
**"Comparison vs Software Prefix Sharing — EMPIRICAL (R4 P0-1)"** that:
- Acknowledges vLLM APC / RadixAttention achieve similar capacity in software
- Identifies ForkedKV's REAL advantage (kernel-transparent contiguous VA)
- Identifies ForkedKV's REAL disadvantages (fork latency, capacity, granularity)
- Frames the paper as "architectural characterization + kernel-transparent VA"

The old "Comparison vs vLLM APC — analytic (R2 P1-B)" section is retained because
its design-level reasoning is still correct; the new EMPIRICAL section follows
it and supersedes it for headline claims.

The "What this buys you" section was rewritten with separate "Buys (vs naive
full-clone)" and "Buys (vs strong software baseline)" paragraphs to make the
two framings explicit and prevent future strawman accusations.

### P0-3 — Fix "OS transparency" / "page-fault-on-write" language (Gemini Attack 3)

**Globally retracted** the "OS-style CoW" and "page-fault-on-write" phrasings.

Edits:
- `src/vmm_pool.py` module docstring — replaced the "OS-style CoW over the GPU MMU"
  paragraph with the honest "software-mediated CoW with driver-level physical-page
  remap" explanation.
- `src/kv_branch_manager.py` module docstring + interior comment — same.
- `WRITEUP.md` §100 (Priority-1 primitives) — the unqualified
  "Page-fault-on-write" bullet is now explicit about software detection vs
  driver-level remap.
- `WRITEUP.md` new section **"Honest framing — what is and isn't OS-like"** — explains
  why hardware faults are unavailable, what we get instead, and the latent path if
  NVIDIA exposes write-protect faults in a future driver.
- `prototype_status.md` D4 — rephrased to "Software-mediated CoW with driver-level
  remap"; explicitly retracts the page-fault language.
- `LIMITATIONS.md` — added new item **#16** marking the retraction explicitly.

Remaining occurrences of "page-fault-on-write" / "OS-style CoW" in WRITEUP/source
are now ALL inside qualified RETRACTION or EXPLANATION blocks (verified via grep).

### P0-4 — Workload justification for CoW-on-write (Gemini Attack 2: "non-problem")

Added new WRITEUP section **"Workload justification — when does CoW-on-write actually
fire?"** that:
- Honestly acknowledges vanilla autoregressive decode is append-only (CoW only fires
  on the boundary page).
- Lists 5 real workloads where in-place KV mutation IS a real operation: speculative-
  decoding rollback, tree-of-thought branch-and-edit, multi-turn agents that rewrite
  history (tool-call retries), sliding-window / context-compression eviction,
  reasoning models with backtracking.
- Notes that Metric 5c (R2 P0-4) already exercises the tree-of-thought pattern and
  proves the mechanism with assertions.
- Concedes the CoW-on-write capability is forward-looking for chat-completion serving
  but valuable for branch-and-edit workloads.

Added `LIMITATIONS.md #17` codifying this acknowledgement.

### Lab 1 phrasing fix (all four reviewers asked)

Replaced the phrasing "determined by an NVIDIA driver-internal mapping-table limit"
in three places with the committee-approved version:

> "The ceiling is independent of `vm.max_map_count` (which retains >99.9% headroom)
> and manifests exclusively within the CUDA VMM driver's `cuMemSetAccess` path.
> This is consistent with a per-context mapping-metadata capacity in the NVIDIA
> driver (~520K entries on H100, driver 580.82.07) — a structural limit not
> tunable from userspace."

Locations updated:
- `WRITEUP.md` §"Lab 1 corollary" (the long version)
- `WRITEUP.md` TL;DR Lab 1 bullet (the short version)
- `LAB1_NOTES.md` "Interpretation §" trailing summary

The change is intentionally weaker on the closed-source-internal-attribution claim
("consistent with") and stronger on the empirical claim ("manifests exclusively
within the cuMemSetAccess path"). This addresses claude's YELLOW qualification
directly.

---

## Files touched

| File | Change |
|------|--------|
| `src/baseline_prefix_sharing.py` | NEW — vLLM-APC-style allocator (~170 lines) |
| `src/vmm_pool.py` | Docstring: retracted "OS-style CoW" framing |
| `src/kv_branch_manager.py` | Docstring + interior comment: retracted "page-fault-on-write" |
| `bench/bench_software_baseline.py` | NEW — head-to-head benchmark |
| `bench/make_baseline_compare_figure.py` | NEW — figures for new bench |
| `data/baseline_compare_m1_fork_latency.csv` | NEW |
| `data/baseline_compare_m2_cow_granularity.csv` | NEW |
| `data/baseline_compare_m3_capacity.csv` | NEW |
| `figures/baseline_compare_m1_fork_latency.png` | NEW |
| `figures/baseline_compare_m3_capacity.png` | NEW |
| `WRITEUP.md` | Repositioned TL;DR; new EMPIRICAL software-baseline section; new "Honest framing" section; new "Workload justification" section; "What this buys you" split into vs-naive and vs-software paragraphs; Lab 1 phrasing fix (×2); page-fault-on-write retraction. |
| `LIMITATIONS.md` | Updated #13 (vLLM analytic→empirical); added #16 (OS-style retraction); added #17 (append-only acknowledgement) |
| `LAB1_NOTES.md` | Replaced "fixed driver constant" claim with committee-approved "consistent with per-context mapping-metadata capacity" phrasing |
| `prototype_status.md` | D4 rephrased to "Software-mediated CoW with driver-level remap" |

---

## What survived the strong-baseline attack (the new headline)

1. **Forensic GPU VMM characterization** (Metrics 4, 4b, Lab 1, Metric 5b
   partial-page waste, B5/B8 cost decomposition, VA-pool reuse). This is the
   architectural-paper material — it stands on its own as a study of NVIDIA's
   VMM substrate for branching workloads, useful even if nobody adopts our
   exact mechanism.

2. **Kernel-transparent contiguous virtual address.** The one mechanism-level
   advantage that survives the empirical comparison. Software prefix sharing
   forces a paged-attention kernel; we do not. For deployments unwilling to
   maintain custom attention kernels, this is a real engineering tax saved.

## What did NOT survive

- "Practical capacity advantage." Against vLLM APC's block-table sharing,
  software wins capacity by ~6× at our test prefix; the 14× claim only holds
  against naive full-clone (which nobody deploys).
- "Lower fork latency." Software is ~700× faster on fork.
- "Finer CoW granularity." Tie at our default sizing; software wins at smaller
  models.
- "OS-style CoW" / "page-fault-on-write." Retracted globally (P0-3).

The honesty is the point. The paper now claims less and what it claims is harder
to attack.

---

## Exit criteria (from REVISION_BRIEF_LAB1_R1.md)

- [x] Software baseline implemented and compared (P0-1)
- [x] WRITEUP repositioned with honest comparison (P0-2)
- [x] Lab 1 phrasing fixed per committee consensus
- [x] "OS transparency" language corrected globally (P0-3)
- [x] Workload justification added (P0-4)
- [x] All changes committed + this REVISION_LAB1_R1_NOTES.md
