# Overall Revision Brief — Flipping Claude/Metacode from YELLOW to GREEN

## Current state:
- Gemini: GREEN ("ship it")
- Claude: YELLOW ("honest but too narrow for ASPLOS")
- Metacode: YELLOW ("model of intellectual honesty but contribution below threshold")
- Codex: TBD (large output, likely similar)

## The problem they identified:
After the honest repositioning, the paper claims:
1. Forensic GPU VMM characterization (driver limits, TLB, contention)
2. Kernel-transparent contiguous VA as one mechanism differentiator

Claude says: "honest repositioning can shrink a contribution below acceptance threshold."
Metacode says: "the contribution is real but narrow."

## What would make it ASPLOS-sized again (without overclaiming):

The characterization becomes a FULL paper when it includes:
1. **Raw hardware counter evidence** (ncu TLB metrics) — not just inferred from timing
2. **Production-representative comparison** — ForkedKV integrated into a real serving path,
   showing the kernel-transparent VA advantage actually matters for throughput
3. **Predictive model validated** — the K≈520K ceiling model predicting behavior on
   different hardware/driver versions

## Lab 2: ncu Hardware TLB Counters (addresses "need raw silicon evidence")

ncu IS available on devgpu014 at /usr/local/cuda-12.8/bin/ncu (version 2025.3.1.0).
Note from Exp 1: `--query-metrics | grep -i tlb` found NO direct TLB miss metric on
this build. BUT there are L2 metrics (lts__t_sector_hit_rate, aperture analysis) and
the `--set full` collection may expose more.

**What to do:**
a) Run `ncu --set full` on ONE attention kernel (FlashAttention/SDPA) reading from
   VMM-backed KV pages. Capture ALL available metrics. Look for:
   - lts__t_sector_hit_rate.pct (L2 cache efficiency)
   - lts__t_sector_aperture_device.pct (device vs system memory fraction)
   - sm__throughput.pct (SM utilization)
   - dram__throughput.avg.pct_of_peak (HBM bandwidth saturation)
b) Compare: contiguous baseline vs VMM-paged at seqlen 8192, 64 branches
c) If L2 hit rate is identical → proves the contiguous VA layout gives identical
   cache behavior to standard allocation (the hardware doesn't distinguish)
d) This is the "raw counter" evidence Claude/metacode want

**If ncu still can't give TLB-specific metrics:** Document which metrics ARE available,
show the L2/bandwidth metrics are identical, and honestly note TLB counters aren't
exposed on this driver version. That's still more than timing alone.

## Lab 3: Minimal Serving Integration (addresses "too narrow / no production relevance")

The strongest way to prove kernel-transparent VA matters in practice:
- Take a real multi-branch agent workload (tree-of-thought on GSM8K or MATH)
- Run it with: (a) vLLM APC (software prefix sharing + custom paged kernel),
  (b) ForkedKV + standard FlashAttention
- Measure: time-to-first-token on branch, attention kernel throughput, total QPS
- IF ForkedKV's attention throughput is higher (because FlashAttention is faster than
  paged-attention on contiguous KV): THAT is the production-relevant delta

**Minimum viable version (doesn't require full vLLM integration):**
- Build a mini serving loop: prefill prompt → fork N branches → decode each
- Use torch SDPA (represents "standard kernel") for ForkedKV
- Use a simplified paged-attention kernel (block-table lookup) for the software baseline
- Compare attention kernel latency at equal batch size
- This isolates the one claim we're making: contiguous VA → faster standard kernels

## Deliverables needed:
- `bench/bench_lab2_ncu_counters.py` + `data/lab2_ncu_*.csv`
- `bench/bench_lab3_kernel_comparison.py` + `data/lab3_kernel_comparison.csv`
- Updated WRITEUP with "Hardware Counter Evidence" section + "Kernel Throughput Comparison"
- Updated LIMITATIONS if ncu doesn't give TLB-specific metrics

## Exit criteria:
Both Labs 2+3 complete, WRITEUP updated, committed. Then re-dispatch committee.

## NO-OVERCLAIM:
- If ncu shows ZERO difference between VMM and contiguous: report that (supports "transparent")
- If paged kernel is actually just as fast as SDPA: report that (weakens our delta)
- Never claim production numbers from a prototype loop

Time budget: 8-16 hours for both labs combined.
GO.
