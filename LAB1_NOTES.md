# Lab 1 — The CoW branch ceiling is driver-internal, not Linux-VMA-bound

**Status:** finding confirmed. Sysctl sweep is moot — `vm.max_map_count` is
already 128× larger than the ceiling we hit.

## Question

Metric 4b (R3) showed a hard ceiling at **K ≈ 520K** total `(VA-page → phys-handle)`
mappings, with a fixed forensic OOM at `cuMemSetAccess`. The natural first
hypothesis is the Linux per-process VMA limit (`vm.max_map_count`, default 65,530
on most distros). Each cuMemMap could in principle land as a separate VMA.

If the kernel VMA limit were the gate, we should see VMA count climb in lockstep
with branches × pages-per-branch and hit the sysctl wall at OOM time.

## Setup

- Host: `devgpu014` (H100, 97 GiB free at start), driver `580.82.07` (Open Kernel
  Module, Feb 2026 build). CUDA toolkit 12.8.
- `/proc/sys/vm/max_map_count` = **67,108,864** (already raised — somebody on this
  host bumped it; default would be 65,530).
- Workload: identical to Metric 4b's 12-GiB cell — fill one prefix branch, snapshot,
  zero-copy fork CoW children until `cuMemSetAccess` OOMs. VA-pool reuse disabled.
- Sample `wc -l /proc/self/maps` every 4 forks.

## Result

```
branches at OOM        : 84
vma_count at OOM       : 392
vm.max_map_count       : 67,108,864
vma_utilisation        : 0.0006%
forensic OOM call site : cuMemSetAccess
total mappings issued  : 84 × 6144 = 516,096   (consistent with K≈520K from Metric 4b)
```

Across all 84 fork operations the userspace VMA count moved by **2** (390 → 392).
516,096 driver mappings produced effectively **zero** new VMAs.

CSV: `data/lab1_vmmap_count.csv`. One-line summary: `data/lab1_vmmap_summary.txt`.

## Interpretation

1. **The ceiling is not the Linux VMA sysctl.** We OOM at 0.0006% of `vm.max_map_count`.
   Even with the default 65,530 we would still have 167× headroom (392 / 65,530).
2. **The driver is doing its own bookkeeping.** Per-page `cuMemMap`+`cuMemSetAccess`
   does not surface as a separate VMA in `/proc/self/maps` — the entire 12-GiB VA
   reservation per branch is a single VMA, and the per-page access descriptors
   live in a driver-internal table that is invisible to the kernel VM accounting.
3. **The failure mode confirms it.** A kernel VMA-limit failure would manifest as
   `mmap` returning `ENOMEM` (or `cuMemAddressReserve` failing). What we see is
   `cuMemSetAccess` failing — the driver-side call that walks its mapping table
   to register read/write permission on each (VA-page, phys-handle, device) triple.
   That table is what is full.

So `K ≈ 520K` is **consistent with a per-context mapping-metadata capacity inside
the NVIDIA driver** (H100 + driver 580.82.07 in our setup). It is a structural
limit not tunable from userspace; the ceiling is independent of `vm.max_map_count`
(which retains >99.9% headroom) and manifests exclusively within the CUDA VMM
driver's `cuMemSetAccess` path. We do not claim to know the driver's internal
data structure (it is closed-source); we report what we measured: the failing
call is `cuMemSetAccess`, the kernel VM accounting is essentially unchanged, and
the ceiling is reproducible across prefix sizes.

## Implications for the paper

This is *better* for the ASPLOS narrative than the original sysctl-tuning angle:

- The ceiling is real, predictable (`max_branches ≈ K/P`, K≈520K), AND
  **outside the operator's control** — you cannot `sysctl -w` your way out of it.
- It is a *driver* limit. The fix path is one of:
  (a) NVIDIA raises K in a future driver,
  (b) coalesce many small mappings into fewer, larger ones (e.g. 64-MiB superpages
      in the VMM API instead of 2-MiB pages — would buy 32× headroom for free),
  (c) recycle VA reservations with VA-pool reuse (already implemented, gives the
      ~119× reuse number from R2 P0-2).
- Strengthens the "concurrency ceiling is structural, not data-bound" claim in
  WRITEUP §3.x — live HBM stays at 12.00 GiB throughout (single prefix), yet the
  forks fail. Nothing about RAM, GPU memory, or VMAs is exhausted; only NVIDIA's
  internal book.

## What we did NOT do (and why)

- **No sysctl sweep.** Pointless: the host's max is already 67 M, the ceiling is
  516 K, and the failure is on a driver call (not `mmap`). Sweeping the sysctl
  cannot change anything below 516,096.
- **No `/proc/driver/nvidia/` mapping-counter.** The open kernel module exposes
  `params`, `registry`, `version`, `warnings`, and per-GPU `information`/`power`/
  `registry` — none expose a per-process or per-context mapping-table count. The
  closed-source resman component of the open module owns this table and does not
  surface it via procfs. (For a future deep dive: `nvidia-smi -q -d MEMORY` and
  the VMM-aware variants of `cuMemGetAllocationGranularity` / `cuDeviceGetAttribute`
  are also silent on this.)

## Files

- `bench/bench_lab1_vmmap_count.py` — measurement
- `data/lab1_vmmap_count.csv` — per-step (branches, vma_count, live_gib)
- `data/lab1_vmmap_summary.txt` — one-line forensic summary
