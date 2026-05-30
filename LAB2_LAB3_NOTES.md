# Lab 2 & Lab 3 — Corrected Results Summary

This note supersedes the earlier Lab 2/3 summary. The original version mixed an
intermediate ncu run with the final committed CSV and over-weighted a minimal
Triton paged-attention baseline. The authoritative data is:

- Lab 2: `data/lab2_ncu_summary.csv`, `data/lab2_ncu_summary.txt`, `data/lab2_ncu_counters.csv`
- Lab 3: `data/lab3_kernel_comparison.csv` (minimal Triton paged baseline; retained as a weak-baseline microbench)
- Lab 3b: `data/lab3b_flashinfer.csv`, `LAB3B_NOTES.md` (production FlashInfer baseline; authoritative for paged-kernel claims)

## Lab 2: Hardware Counter Evidence (ncu)

**Goal:** test whether a VMM-backed contiguous-VA tensor perturbs the GPU compute
pipeline compared with a standard contiguous allocation.

**Method:** NVIDIA Nsight Compute (`ncu`) profiling one PyTorch SDPA-dispatched cuDNN
sm90 flash-attention kernel on H100. A single SDPA call is wrapped in
`cudaProfilerStart/Stop`.

## Final clean result: seqlen=4096

| Metric | Contiguous | VMM-backed contiguous VA | Δ% |
|---|---:|---:|---:|
| `gpu__time_duration.sum` | 574,768 ns | 574,976 ns | **+0.04%** |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 71.765% | 71.735% | **-0.04%** |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 8.995% | 9.005% | **+0.11%** |
| `smsp__inst_executed.sum` | 1.56552976e8 | 1.565534535e8 | **+0.0003%** |
| `lts__t_sector_hit_rate.pct` | 86.95% | 82.505% | **-5.11%** |
| `lts__t_sectors_aperture_device.sum` | 3.862e7 | 4.081e7 | **+5.65%** |
| `lts__t_sectors_aperture_peer.sum` | 0 | 0 | n/a |
| `lts__t_sectors_aperture_sysmem.sum` | 3.80e4 | 3.05e4 | -19.6% (tiny absolute count) |

**Interpretation:** runtime, SM throughput, DRAM throughput, and instruction count
are indistinguishable. The only material counter movement is a mild adverse L2
locality shift (hit rate -5.1%, device sectors +5.7%), likely from physical-page
placement / set hashing. It is not visible in kernel runtime for this isolated kernel.

## Lab 2 limitations

- **No direct TLB counter:** CUDA 12.8 / driver 580 exposes no ncu metric containing
  `tlb` (`ncu --query-metrics | grep -i tlb` is empty). Lab 2 supports read-path
  transparency through downstream counters; it does not directly measure TLB walks.
- **No clean VMM data at seqlen=2048:** ncu replay hangs/timeouts for the VMM case at
  the sub-200 µs kernel point. The final CSV intentionally contains clean VMM data for
  seqlen=4096 only.
- **One kernel / one GPU:** only the cuDNN sm90 flash-attention kernel on H100 was
  profiled. Other kernels and architectures remain future work.

## Lab 3: Minimal Triton paged baseline (weak baseline)

Lab 3 compared contiguous SDPA against a minimum-viable Triton paged-attention kernel.
It showed large contiguous wins (2–11×), but this is **not** the production conclusion:
the paged kernel lacks split-K, tensor-core GQA, and production tuning. Treat Lab 3 as a
sanity check that naive block-table indirection is expensive, not as a paper headline.

## Lab 3b: Production FlashInfer baseline (authoritative)

Lab 3b corrected the overclaim by comparing against FlashInfer 0.6.12 with a fair
GQA-native SDPA baseline (`enable_gqa=True`). The result is much smaller:

| B | S | SDPA-GQA contiguous | FlashInfer paged | Ratio |
|---|---:|---:|---:|---:|
| 32 | 512 | 0.024 ms | 0.036 ms | 1.52× |
| 32 | 2048 | 0.075 ms | 0.091 ms | 1.22× |
| 32 | 8192 | 0.246 ms | 0.265 ms | 1.08× |
| 64 | 512 | 0.048 ms | 0.058 ms | 1.21× |
| 64 | 2048 | 0.132 ms | 0.145 ms | 1.10× |
| 64 | 8192 | 0.474 ms | 0.499 ms | 1.05× |

**Correct conclusion:** the large 2–11× kernel-speed claim is retracted. Contiguous VA
is modestly faster (5–52%, largest at short context; ~5–10% at long context) and avoids
block-table machinery, but raw kernel speed is not a load-bearing thesis contribution.
ForkedKV's surviving research case rests on the GPU VMM mapping-metadata ceiling,
mutation control-plane serialization, and kernel-transparent branch/CoW semantics — not
on a large attention-kernel speedup.
