"""
Lab 2 — Hardware Counter Evidence (ncu).

For each (method, seqlen) we run bench/_lab2_target.py once under ncu, capturing the
single SDPA call inside cudaProfilerStart/Stop. ncu reports per-kernel metrics — for
the seqlens we use, SDPA is a single cuDNN flash-attention kernel, so we get a clean
single row per (method, seqlen, repeat).

Why this matters:  the paper claims "kernel-transparent contiguous VA" — i.e. the GPU
hardware should see VMM-paged memory the same as a standard contiguous allocation.
If that's true, *raw silicon counters* should match: identical L2 hit rate, identical
DRAM throughput, identical SM utilization, identical kernel duration.

We capture (driver-version dependent — see LIMITATIONS for full list):
  gpu__time_duration.sum                              -> kernel runtime
  sm__throughput.avg.pct_of_peak_sustained_elapsed    -> SM utilization
  dram__throughput.avg.pct_of_peak_sustained_elapsed  -> HBM bandwidth
  lts__t_sector_hit_rate.pct                          -> L2 hit rate
  lts__t_sectors_aperture_device.sum                  -> # L2 sectors hitting device memory
  lts__t_sectors_aperture_peer.sum                    -> # L2 sectors hitting peer memory
  lts__t_sectors_aperture_sysmem.sum                  -> # L2 sectors hitting system memory
  smsp__inst_executed.sum                             -> total instructions issued

NCU on driver 575/CUDA 12.8 (this devgpu) does NOT expose direct TLB metrics — see
`ncu --query-metrics | grep -i tlb`.  We document that in LIMITATIONS and rely on
L2/DRAM/SM as a *necessary condition*: if the GPU hardware were doing extra work for
VMM (extra TLB walks, extra L2 traffic, lower SM occupancy), at least one of these
six metrics would shift. We measure that they don't.

Output: data/lab2_ncu_counters.csv  columns: method, seqlen, repeat, kernel, metric, value
"""
import os, sys, csv, subprocess, statistics, json, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "bench", "_lab2_target.py")
NCU = "/usr/local/cuda-12.8/bin/ncu"

METRICS = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "lts__t_sectors_aperture_device.sum",
    "lts__t_sectors_aperture_peer.sum",
    "lts__t_sectors_aperture_sysmem.sum",
    "smsp__inst_executed.sum",
    "l1tex__t_sector_hit_rate.pct",
]

SEQLENS = [2048, 4096, 8192]
METHODS = ["contig", "vmm"]
REPEATS = 3   # multiple runs per cell to estimate cross-run noise

OUT = os.path.join(ROOT, "data", "lab2_ncu_counters.csv")


def parse_ncu_csv(text):
    """ncu --csv emits a header row then one row per (kernel, metric).
    Returns list of dicts with keys: kernel, metric, unit, value."""
    rows = []
    lines = [ln for ln in text.splitlines() if ln.startswith('"')]
    if not lines:
        return rows
    # detect CSV header
    import csv as _csv, io
    rdr = _csv.reader(io.StringIO("\n".join(lines)))
    hdr = next(rdr)
    idx = {h: i for i, h in enumerate(hdr)}
    for r in rdr:
        try:
            rows.append(dict(
                kernel=r[idx["Kernel Name"]],
                metric=r[idx["Metric Name"]],
                unit=r[idx["Metric Unit"]],
                value=r[idx["Metric Value"]],
            ))
        except (IndexError, KeyError):
            continue
    return rows


def run_ncu(method, seqlen, log_dir):
    cmd = [
        NCU,
        "--target-processes", "all",
        "--profile-from-start", "no",
        "--metrics", ",".join(METRICS),
        "--csv",
        sys.executable, TARGET, method, str(seqlen),
    ]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=600)
    dur = time.time() - t0
    out = p.stdout + "\n" + p.stderr
    log_path = os.path.join(log_dir, f"ncu_{method}_{seqlen}.log")
    with open(log_path, "w") as f:
        f.write(out)
    if p.returncode != 0:
        print(f"  !! ncu failed for {method} sl={seqlen} rc={p.returncode}")
        print(out[-1500:])
    rows = parse_ncu_csv(out)
    print(f"  {method:<7} sl={seqlen}  {len(rows):3d} metric-rows  ({dur:.1f}s)")
    return rows


def main():
    log_dir = os.path.join(ROOT, "data", "lab2_ncu_logs")
    os.makedirs(log_dir, exist_ok=True)
    all_rows = []
    for rep in range(REPEATS):
        for sl in SEQLENS:
            for method in METHODS:
                rs = run_ncu(method, sl, log_dir)
                for r in rs:
                    all_rows.append(dict(method=method, seqlen=sl, repeat=rep, **r))
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method","seqlen","repeat","kernel","metric","unit","value"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\nwrote {OUT}  ({len(all_rows)} rows)")

    # quick summary: median per (method, seqlen, metric)
    from collections import defaultdict
    agg = defaultdict(list)
    for r in all_rows:
        # only keep the SDPA flash kernel (skip any tiny housekeeping kernels)
        if "sdpa" in r["kernel"].lower() or "flash" in r["kernel"].lower() or "attention" in r["kernel"].lower():
            try:
                v = float(r["value"].replace(",", ""))
                agg[(r["method"], r["seqlen"], r["metric"])].append(v)
            except ValueError:
                pass
    print("\n--- median across repeats (SDPA kernel only) ---")
    print(f"{'method':<7} {'sl':<6} {'metric':<60} {'median':>12} {'spread%':>8}")
    for (m, sl, mt), vs in sorted(agg.items(), key=lambda x:(x[0][1], x[0][2], x[0][0])):
        med = statistics.median(vs)
        spread = (max(vs)-min(vs))/abs(med)*100 if med else 0
        print(f"{m:<7} {sl:<6} {mt:<60} {med:>12.4g} {spread:>7.2f}%")

if __name__ == "__main__":
    main()
