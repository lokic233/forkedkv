# E-E: Write-after-fork bit-identical KV isolation as a runtime PRIMITIVE — gating result

**Cluster E thesis under test:** *Write-after-fork EXACT bit-identical isolation of GPU
KV state is a runtime PRIMITIVE for safe speculative/untrusted context edits — providing
O(1-page) verified rollback with driver-handle-level proof of non-corruption: a
correctness/safety guarantee, NOT a performance win.*

**Hostile question this experiment answers:** Does HW VMM CoW provide an
isolation/correctness primitive that software prefix-sharing (vLLM-APC-style refcounted
block tables) genuinely CANNOT match? Or does software give equivalent bit-identical
isolation + cheap rollback, collapsing the thesis into software-equivalence (a kill)?

---

## Setup

- **Node:** cli:devgpu014 (1×H100, 97.9 GiB), driver 580.82.07, CUDA 12.8, cuda-python via
  `~/branchable_replay/.venv` (torch 2.11+cu128). GPU 2 (idle) used; allocations bounded
  to ≤ a handful of 2 MiB pages per arm (<10 MiB live KV) — well within headroom.
- **Workspace:** isolated `~/committee_naviC/` only.
- **Model:** real Qwen2.5-7B-Instruct **layer-0** K/V (n_kv=4, head_dim=128, fp16), genuine
  RMSNorm+QKV-proj+RoPE+SDPA forward to fill the prefix with real model bytes (2048 tokens
  = exactly one 2 MiB VMM page / one SW block).
- **Threat model (identical for both arms):** an agent forks N speculative branches off a
  shared prefix; branch `c0` is UNTRUSTED and overwrites a shared prefix page mid-decode
  (`fill 0xAB`). The runtime must (a) keep sibling `c1`/parent BIT-IDENTICALLY uncorrupted
  and (b) roll `c0` back to a verified-clean state cheaply.
- **Sweep:** N∈{4,8}, prefix=2048 tok (1 page), 3 rollbacks each, **3 reps** → 12 rows
  (`repos/forkedkv/data/ee_isolation.csv`). Bench: `repos/forkedkv/bench/bench_EE_isolation.py`.

**Two arms, run through the IDENTICAL scenario:**
- **ARM-HW** — `KVBranchManager` VMM CoW. Isolation verified TWO ways: (1) **driver-handle
  proof** via `cuMemRetainAllocationHandle` (the contaminated branch maps a *physically
  distinct* allocation handle; the sibling maps the *same* handle as the pre-fork
  snapshot), and (2) byte-compare to the pre-fork snapshot. Rollback = `cuMemUnmap` the
  dirty private page + `cuMemMap` the snapshot's physical handle + incref (re-alias).
- **ARM-SW** — `RealGpuPrefixSharingManager`, a vLLM-APC-equivalent refcounted block-table
  allocator I built **on REAL GPU memory** (real `cuMemcpyDtoD` CoW copies, real read-backs)
  for a fair fight — the repo's existing `baseline_prefix_sharing.py` stored blocks on host
  numpy and only *simulated* copies, which would be apples-to-oranges. Isolation verified
  by byte-compare to snapshot + refcount bookkeeping; **no physical-handle primitive
  exists** for software. Rollback = repoint the block-table entry to the snapshot block.

---

## Results (means over 6 rows per arm; all reps identical on the boolean facts)

| metric | ARM-HW (VMM CoW) | ARM-SW (block-table, real GPU) |
|---|---|---|
| isolation_verified (all reps) | **True** | **True** |
| sibling bit-identical to snapshot (all) | **True** | **True** |
| rollback bit-identical to snapshot (all) | **True** | **True** |
| **driver_handle_proof** (all) | **True** | **False (impossible)** |
| rollback bytes copied | **0** | **0** |
| rollback cost (µs) | **247.81** | **137.98** |
| fork cost (µs) | 134.70 | **0.56** |
| contaminated-write CoW (µs) | 460.74 | **201.29** |
| write bytes copied (1 page) | 2,097,152 | 2,097,152 |
| **contiguous_va_preserved** | **True** | **False** |

Driver-handle evidence (HW, representative row, `notes` column):
`handle_distinct=True; sib_handle==snap=True; cont_handle=100456544;
sib_handle=441691296; snap_handle=441691296` — the contaminated branch is backed by a
physically different allocation handle (100456544) than the sibling/snapshot (441691296),
verified at the driver/MMU level, not by application bookkeeping.

---

## HONEST VERDICT — Cluster E mostly COLLAPSES into software-equivalence (a near-kill),
## with ONE narrow, real, but weak surviving distinction.

On the property the thesis actually claims — **bit-identical isolation + O(1) cheap
rollback** — **software matches hardware exactly.** Both arms verified bit-identical
sibling non-corruption and bit-identical rollback across all 3 reps at N=4 and N=8
(isolation_verified=True, sibling_bitidentical=True, rollback_bitidentical=True for both),
and both roll back by copying **0 bytes** (pure refcounted re-alias / block-table
repoint). On the *measured operations* software is strictly **faster**: fork 0.56 µs vs
134.70 µs (≈240×), contaminated-write CoW 201 µs vs 461 µs (2.3×), rollback 138 µs vs
248 µs (1.8×) — consistent with sibling C*'s finding that HW VMM CoW loses on speed. So the
core thesis claim ("a correctness guarantee SW cannot give") is **false as stated**:
refcounted software prefix-sharing already provides bit-identical isolation + cheap
verified rollback, and does so cheaper. **The ONLY thing HW provides that software
genuinely cannot** is a *physical*, driver-handle-verifiable proof of separation
(`cuMemRetainAllocationHandle`: contaminated handle 100456544 ≠ sibling/snapshot
441691296, all reps, driver_handle_proof=True for HW and structurally impossible for SW)
**plus** contiguous-VA preservation on the recovered branch (contiguous_va_preserved=True
vs False), which lets an *unmodified* FlashAttention kernel run on the rolled-back branch
where the SW block-table forces a custom paged kernel. **But neither survives hostile
review as a load-bearing isolation primitive:** the handle proof only has value under a
threat model where you distrust your own allocator's bookkeeping yet still trust the CUDA
driver and the kernel's address generation — a self-undermining and narrow assumption,
since in both arms the runtime controls the only code path that touches KV and a correct
refcounted allocator already guarantees the bytes; and the contiguous-VA advantage is a
*kernel-compatibility/performance* property (already shown a net loss by C*), not a
correctness/safety one. **Recommendation: Cluster E is NOT a viable distinct
correctness primitive — it collapses into software-equivalence and should be killed as a
standalone safety thesis.** The single honest residue worth one sentence in any survivor
writeup: HW VMM CoW is the only mechanism that can emit a driver-level *physical* proof of
KV-page non-aliasing (`cuMemRetainAllocationHandle`) — a verifiability nicety, not a
capability software lacks for actually achieving isolation.

### Caveats (do not over-read the numbers)
- **Single-page prefix, one transformer layer, layer-0 only.** Real KV is 28 layers; the
  isolation logic is per-page and layer-independent, but absolute latencies are not
  end-to-end decode costs.
- **Software CoW here is bandwidth-fair but kernel-idealized:** the SW arm pays the real
  2 MiB D2D copy on a contaminated write but does NOT model the custom paged-attention
  kernel it must run (the contiguous_va_preserved=0 cost is asserted, not benchmarked here
  — C* covers the kernel asymmetry). This makes the SW arm's *measured* speed an upper
  bound on its real-world advantage, but does not change the isolation/rollback equivalence
  verdict, which is exact (bit-compare) and reproduced across all reps.
- **HW fork/CoW/rollback latencies** include driver `cuMemMap`/`cuMemUnmap`/`cuMemSetAccess`
  calls; these are the documented dominant cost (consistent with the repo's B5/B8 notes).
- **Threat model is the crux, not the bytes:** the entire thesis lives or dies on whether a
  *physical* handle proof is required. We argue it is not, under any realistic single-engine
  serving runtime that owns its allocator.

### Artifacts
- Data: `~/committee_naviC/repos/forkedkv/data/ee_isolation.csv` (12 rows, 3 reps × N∈{4,8})
- Bench: `~/committee_naviC/repos/forkedkv/bench/bench_EE_isolation.py`
- This writeup: `~/committee_naviC/committee/EE_RESULT.md`
