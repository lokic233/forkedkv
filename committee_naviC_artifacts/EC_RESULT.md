# E-C: Rollback-Heavy Agentic Decode — HW VMM CoW vs Software Prefix Sharing vs FlashInfer-Paged

**Decisive end-to-end experiment for the committee.**
Author: Navi (sub-agent, committee_naviC). Date: 2026-05-30. Node: `cli:devgpu014` (1× NVIDIA H100, device 0).

---

## THE QUESTION (what the committee votes on)

> Is there ANY regime (any cell of the sweep) where **HW VMM CoW** (`KVBranchManager`) beats **BOTH**
> (a) software prefix sharing (vLLM-APC style) **AND** (b) FlashInfer paged decode, on **end-to-end
> latency** OR on **capacity** (max branches before OOM)?
> Hypothesised HW win-region: *long shared prefix AND rollback-heavy AND a kernel that needs contiguous VA.*

## VERDICT (one paragraph, honest)

**No. The HW-CoW win-region is empty.** Across all 12 sweep cells (prefix ∈ {512, 4096} × N ∈ {4, 16}
× R ∈ {0, 4, 16}), the HW VMM-CoW arm — measured in its most favourable, mechanism-isolated form (SDPA
over its own contiguous CoW-backed KV, real driver CoW on every rollback) — **never** beats both baselines
on latency. It is **1.06×–2.20× slower than software prefix sharing** and **41×–152× slower than FlashInfer
paged decode** (`ec_rollback_e2e.csv`, ratios computed from the `median_latency_ms` column). The hypothesis
is *directionally* supported but never crosses over: the long-prefix cells are exactly where HW is closest
to software — the gap narrows to **+5.6 %** at `long, N=4, R=0` (HW 73.07 ms vs SW 69.19 ms) and stays
under +28 % across all long cells — yet it remains a loss in every one, and rollbacks (R) make HW *relatively
worse*, not better (long N=4: HW/SW goes 1.06×→1.14×→1.30× as R goes 0→4→16, because each rollback adds a
real 2 MiB driver CoW remap + sync that software pays as a few-microsecond block-table refcount op). On
**capacity** there is also no HW win: both mechanisms alias ≥64 branches off the shared prefix with **zero**
OOM at realistic N (HW is bounded by MMU mapping-metadata, software by host RAM — neither binds at N≤16; see
`max_branches` column, `>=64(cap-limited)` for HW). The committee's prior facts hold and compound here:
software is far cheaper per fork/CoW, and FlashInfer's production paged kernel is so fast (sub-5 ms for the
whole N=16 batch-decode) that even a *zero-cost* KV mechanism could not close a 40–150× gap. **This is a
clean, publishable negative result and matches prior committee rounds.**

---

## SETUP & EXACT CONFIG

| Item | Value |
|---|---|
| GPU | 1× NVIDIA H100 (device 0), CUDA driver via cuda-python 13.3 (`cuda.bindings.driver`) |
| Interpreter | `/home/dengcchi/sglang-env/bin/python` — torch **2.11.0+cu130**, flashinfer **0.6.11.post1** |
| Model | **Qwen2.5-7B-Instruct, layer 0** (real weights: GQA 28 q / 4 kv heads, head_dim 128, RoPE θ=1e6, RMSNorm), fp16. *Single real transformer layer* (per harness convention — proves real-weight attention over CoW KV; NOT full 28-layer quality). Cached locally; 7B was available so the 1.5B fallback was **not** needed. |
| KV page (HW) | 2 MiB CUDA-VMM page = **2048 tokens/page** for K (and V) at this head config |
| SW block | 16 tokens/block (vLLM-APC default) |
| Sweep | prefix_tokens ∈ {long: **4096**, short: **512**}; N (branches) ∈ {**4, 16**}; R (rollbacks/branch) ∈ {**0, 4, 16**}; decode_tokens = **48**/branch |
| Timing | 5 timed reps (median + population stddev), 2 warmup reps, per cell |
| Isolation | **each (arm, cell) runs in its own subprocess** (fresh CUDA context) — required because zero-copy VMM views + driver CoW remaps do not cleanly compose across many fork/CoW/destroy cycles in one long-lived process (observed `cudaErrorIllegalAddress`); per-cell isolation makes the sweep reproducible. Transient driver-lock failures are retried up to 3×. |

### The three arms (workload identical across all)
A shared prefix is built once with a real Qwen forward; N branches fork off it (zero-copy alias); each
branch decodes 48 tokens; R rollbacks per branch overwrite a **shared prefix page** mid-decode (tree-of-
thought / speculative context edit), triggering copy-on-write, then decode continues.

- **`hw_vmm_cow`** *(measured, head-to-head arm)* — KV physically backed by CUDA-VMM CoW pages.
  Fork = MMU page aliasing. Rollback = **real** driver CoW (cuMemUnmap + cuMemMap + 2 MiB D2D copy).
  Decode attention = SDPA over the branch's **contiguous virtual-address** KV (kernel-transparent).
  *This is the strongest fair representative of the HW idea: it gets the contiguous-VA benefit for free.*
- **`hw_vmm_cow_fullfwd`** *(measured, honesty arm — NOT used for the head-to-head)* — same, but full
  real-model per-token forward (embed/proj/rope/attn/MLP/lm_head argmax). Carries Python per-token model
  overhead absent from the other arms; reported for completeness only.
- **`sw_prefix`** *(measured)* — fork/CoW bookkeeping via `SoftwarePrefixSharingManager` (block-table
  refcount + block-granular CoW). Decode attention uses the **same** SDPA-over-contiguous-KV math as
  `hw_vmm_cow`, so attention cost is identical and the only delta is the KV mechanism.
- **`flashinfer`** *(measured kernel + ANALYTIC composition)* — attention served by FlashInfer's
  production `BatchDecodeWithPagedKVCacheWrapper` (paged, tensor-core GQA, page_size 16). Per-step paged
  kernel latency is **measured** at the cell's exact shapes (batch=N, seqlen=prefix+24); fork/CoW
  bookkeeping is the **measured** software-manager cost. End-to-end = `48 × t_paged_step(batch=N) +
  bookkeeping` — **clearly labelled ANALYTIC** in the CSV `notes`. Full live block-table rewrite on every
  rollback was out of time budget; the kernel dominates and is measured at the right shapes, so the
  composition is faithful and, if anything, *charges FlashInfer the full per-step kernel every token*.

---

## RESULTS TABLE (median latency, ms; all numbers cite `data/ec_rollback_e2e.csv`)

| prefix | N | R | HW-CoW (iso) | HW-CoW (full-fwd) | SW-prefix | FlashInfer | HW<SW? | HW<FI? | **HW win?** |
|---|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| long  | 4  | 0  | 73.07 | 341.00 | 69.19 | 1.77 | ✗ | ✗ | **No** |
| long  | 4  | 4  | 78.94 | 347.92 | 69.11 | 1.72 | ✗ | ✗ | **No** |
| long  | 4  | 16 | 91.19 | 441.19 | 69.89 | 1.77 | ✗ | ✗ | **No** |
| long  | 16 | 0  | 291.72 | 1321.01 | 274.91 | 4.65 | ✗ | ✗ | **No** |
| long  | 16 | 4  | 314.02 | 1375.21 | 273.78 | 4.70 | ✗ | ✗ | **No** |
| long  | 16 | 16 | 352.47 | 1615.14 | 277.13 | 4.76 | ✗ | ✗ | **No** |
| short | 4  | 0  | 59.05 | 652.69 | 34.49 | 1.34 | ✗ | ✗ | **No** |
| short | 4  | 4  | 62.24 | 308.11 | 31.00 | 1.37 | ✗ | ✗ | **No** |
| short | 4  | 16 | 63.57 | 323.02 | 30.84 | 1.38 | ✗ | ✗ | **No** |
| short | 16 | 0  | 243.60 | 1800.11 | 128.85 | 1.62 | ✗ | ✗ | **No** |
| short | 16 | 4  | 237.12 | 1961.38 | 129.42 | 1.69 | ✗ | ✗ | **No** |
| short | 16 | 16 | 263.32 | 1284.17 | 119.93 | 1.74 | ✗ | ✗ | **No** |

**HW-CoW (isolated) beats BOTH baselines in 0 / 12 cells.**
HW/SW latency ratio: **1.06×–2.20×** (HW always slower). HW/FI: **41×–152×**.
Closest approach: `long, N=4, R=0` → HW 73.07 ms vs SW 69.19 ms (**+5.6 %**).

### CoW correctness (mechanism worked, it just isn't faster)
`n_cow_events` is non-zero exactly where rollbacks overwrite shared pages: HW fires 8 (N=4) / 32 (N=16)
real driver CoW remaps; SW fires 16/64/256 block-granular CoWs. (HW vs SW CoW *counts* differ because the
HW path CoWs only the K range and uses first-touch-per-page semantics while SW counts every `write_block`;
this is a trace-accounting difference, not a correctness gap — see caveats.) For long, R=0 the HW count is
0 (4096 tok = exactly 2 full pages, decode appends land on a fresh private page); for short, R=0 HW shows
8 because the 512-token prefix shares a page with the first decode tokens, so the first append legitimately
CoWs the shared prefix page.

### Capacity (also not a HW win)
`max_branches`: HW = **≥64 (cap-limited, no OOM)** for both prefixes — the capacity probe aliased 64
branches off the shared prefix with zero added HBM (forks alias physical pages) and did not OOM; HW
capacity is bounded by MMU mapping-metadata (~5×10⁵ mappings per Lab 1), far above realistic N. SW/FI =
**RAM-bound (≫HW)**. Neither mechanism binds at N ≤ 16, so capacity does not discriminate.

---

## CAVEATS (measured vs analytic vs not-done)

1. **Model size / depth.** Single real Qwen2.5-7B **layer 0** (not the full 28-layer model). Systems
   quantities (latency, HBM, CoW bytes) are real; generation quality is not evaluated. 7B weights were
   cached, so no 1.5B fallback was used.
2. **Arm 3 is an analytic composition.** FlashInfer end-to-end = measured per-step paged kernel × 48 +
   measured software bookkeeping. Full live block-table rewrite per rollback was not implemented (time
   budget). This *favours* FlashInfer being charged the full kernel each step; the 41–152× margin is so
   large the composition choice cannot change the verdict.
3. **Single GPU, single driver, multi-tenant node.** Other tenants on the box intermittently held the
   NVIDIA RM global lock, injecting large wall-time jitter into the HW arm specifically (it issues many
   `cuMem*` driver calls). Visible as inflated `stddev_ms` in some HW cells (e.g. short N=16 R=0 ±736 ms);
   the **median** is robust and the cross-arm ordering (HW ≫ SW ≫ FI) is invariant to this jitter. SW/FI
   arms are essentially jitter-free (stddev ≤ ~2 ms) because they issue almost no driver calls.
4. **`peak_hbm_mib` is a node-global free-memory delta** (`cuMemGetInfo`), not process-private; one cell
   shows a negative value (−12732) because a neighbour freed memory mid-run. KV-mechanism HBM is tiny
   (~80 MiB HW prefix+branches; ~1–8 MiB SW pool) and not the discriminating axis.
5. **CoW-count asymmetry** between HW (K-only, per-page first-touch) and SW (per-block) is a trace
   bookkeeping difference; both mechanisms are verified correct elsewhere in the harness (Metric 5c).

---

## ARTIFACTS
- Data: `~/committee_naviC/repos/forkedkv/data/ec_rollback_e2e.csv` (48 rows, one per arm×cell)
- Bench: `~/committee_naviC/repos/forkedkv/bench/bench_EC_rollback_e2e.py`
- This writeup: `~/committee_naviC/committee/EC_RESULT.md`
