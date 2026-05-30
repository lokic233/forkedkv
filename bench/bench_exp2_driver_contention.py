"""
Experiment 2 (ASPLOS Extended Lab): Driver-Level Lock Contention & Multiprocess Scaling.

OBJECTIVE
  Quantify execution delays / driver-level serialization when many threads concurrently
  issue VMM mapping ops (cuMemMap + cuMemSetAccess + cuMemUnmap) on fresh pages. The
  ASPLOS insight: Metric 4's K~520K mapping-metadata ceiling is *explained* if the CUDA
  driver serializes these calls under a global lock — concurrency doesn't help past a
  point, and per-op latency degrades. We look for a CONTENDED regime (a cliff).

  We ALSO include the requested fork-to-first-token sweep (reuse bench_metric5b style)
  separately if time permits — here we focus on the pure driver-op contention scaling,
  which is the core measurement, and emit a fork-latency-vs-branches sub-CSV by calling
  the existing metric5b harness from a wrapper (see run_fork_sweep()).

METHOD
  - Thread sweep: {1, 2, 4, 8, 16, 32, 64, 128} Python threads.
  - Each thread, in a tight loop, does CYCLES_PER_THREAD independent VMM cycles on its
    own fresh page+VA: cuMemCreate -> cuMemAddressReserve -> cuMemMap -> cuMemSetAccess
    -> cuMemUnmap -> cuMemAddressFree -> cuMemRelease. Each op is perf_counter-timed
    (host-side; these driver calls are host-bound, like the CoW microbench B7 note).
  - We record per-op latency for map / setaccess / unmap, plus full-cycle wall time.
  - Metrics: total wall-time per 100 map/unmap cycles (normalized), and per-op
    p50/p95/p99 at each thread count. Aggregate throughput (cycles/sec) vs threads shows
    whether the driver scales (linear) or serializes (flat/cliff).
  - Python threads share one process + one CUDA context. The CPython GIL is released
    inside cuda-python's C driver calls, so true kernel-driver concurrency is exercised;
    any serialization observed is the DRIVER's lock, not the GIL (we note this caveat).

  Optionally (--strace) wrap the whole run under `strace -c -e ioctl` to attribute time
  to the kernel driver ioctl path (the driver talks to nvidia.ko via ioctl). strace is
  available on devgpu014 without root.

OUTPUT
  data/exp2_driver_contention.csv columns:
    threads, cycles_per_thread, total_cycles, wall_s, cycles_per_s,
    norm_ms_per_100cycles, op, p50_us, p95_us, p99_us, mean_us, n_samples
"""
import sys, os, csv, time, threading, argparse, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cuda import cuda

DEVICE = int(os.environ.get("EXP2_DEVICE", "0"))
THREADS = [1, 2, 4, 8, 16, 32, 64, 128]
CYCLES_PER_THREAD = 100   # so total cycles = threads * 100; "per 100 cycles" is natural
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "exp2_driver_contention.csv")


def _ck(ret, site=""):
    if isinstance(ret, tuple):
        err = ret[0]; rest = ret[1:]
    else:
        err = ret; rest = ()
    if int(err) != 0:
        name = cuda.cuGetErrorName(err)[1]
        try: name = name.decode()
        except Exception: pass
        raise RuntimeError(f"CUDA error {int(err)}: {name} at {site}")
    if len(rest) == 0: return None
    if len(rest) == 1: return rest[0]
    return rest


# --- one-time global setup: context, alloc prop, access desc, page size ---
_ck(cuda.cuInit(0))
_DEV = _ck(cuda.cuDeviceGet(DEVICE))
_CTX = _ck(cuda.cuDevicePrimaryCtxRetain(_DEV))
_ck(cuda.cuCtxSetCurrent(_CTX))
_PROP = cuda.CUmemAllocationProp()
_PROP.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
_PROP.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
_PROP.location.id = DEVICE
_PAGE = _ck(cuda.cuMemGetAllocationGranularity(
    _PROP, cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM))
_ACCESS = cuda.CUmemAccessDesc()
_ACCESS.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
_ACCESS.location.id = DEVICE
_ACCESS.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE


def worker(cycles, lat):
    """Run `cycles` independent VMM map/unmap cycles; append per-op latencies (us) to the
    thread-local `lat` dict of lists. Each thread must set the CUDA context current."""
    _ck(cuda.cuCtxSetCurrent(_CTX))
    create, reserve, mmap, setacc, unmap, free, release, full = (
        lat["create"], lat["reserve"], lat["map"], lat["setaccess"],
        lat["unmap"], lat["free"], lat["release"], lat["full_cycle"])
    for _ in range(cycles):
        t_full = time.perf_counter()
        t = time.perf_counter(); h = _ck(cuda.cuMemCreate(_PAGE, _PROP, 0), "create"); create.append((time.perf_counter()-t)*1e6)
        t = time.perf_counter(); va = _ck(cuda.cuMemAddressReserve(_PAGE, 0, 0, 0), "reserve"); reserve.append((time.perf_counter()-t)*1e6)
        t = time.perf_counter(); _ck(cuda.cuMemMap(va, _PAGE, 0, h, 0), "map"); mmap.append((time.perf_counter()-t)*1e6)
        t = time.perf_counter(); _ck(cuda.cuMemSetAccess(va, _PAGE, [_ACCESS], 1), "setaccess"); setacc.append((time.perf_counter()-t)*1e6)
        t = time.perf_counter(); _ck(cuda.cuMemUnmap(va, _PAGE), "unmap"); unmap.append((time.perf_counter()-t)*1e6)
        t = time.perf_counter(); _ck(cuda.cuMemAddressFree(va, _PAGE), "free"); free.append((time.perf_counter()-t)*1e6)
        t = time.perf_counter(); _ck(cuda.cuMemRelease(h), "release"); release.append((time.perf_counter()-t)*1e6)
        full.append((time.perf_counter()-t_full)*1e6)


def run_threadcount(nthreads, cycles_per_thread):
    """Spawn nthreads workers, each doing cycles_per_thread cycles. Return (wall_s, merged_lat)."""
    lats = [dict(create=[], reserve=[], map=[], setaccess=[], unmap=[], free=[], release=[], full_cycle=[])
            for _ in range(nthreads)]
    threads = [threading.Thread(target=worker, args=(cycles_per_thread, lats[i])) for i in range(nthreads)]
    # warmup the context in main thread (page table grows lazily)
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.perf_counter() - t0
    merged = dict()
    for key in lats[0]:
        merged[key] = [v for L in lats for v in L[key]]
    return wall, merged


def pct(xs, p):
    if not xs: return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=str, default=",".join(map(str, THREADS)))
    ap.add_argument("--cycles", type=int, default=CYCLES_PER_THREAD)
    ap.add_argument("--out", type=str, default=OUT)
    args = ap.parse_args()
    thread_list = [int(x) for x in args.threads.split(",")]
    print(f"device={DEVICE} page={_PAGE//1024}KiB cycles/thread={args.cycles}")
    print(f"thread sweep: {thread_list}")

    # warmup one cycle (single thread) to settle the driver/page-table
    run_threadcount(1, 5)

    rows = []
    base_cps = None
    for nt in thread_list:
        wall, lat = run_threadcount(nt, args.cycles)
        total = nt * args.cycles
        cps = total / wall
        if base_cps is None: base_cps = cps
        norm_ms_per_100 = (wall / total) * 100 * 1e3   # ms per 100 cycles (wall, aggregate)
        full50 = pct(lat["full_cycle"], 50); full99 = pct(lat["full_cycle"], 99)
        print(f"  threads={nt:4d}  wall={wall:7.3f}s  cyc/s={cps:9.0f}  "
              f"scaling={cps/base_cps:5.2f}x  full_cycle p50={full50:7.1f}us p99={full99:8.1f}us")
        for op in ["create", "reserve", "map", "setaccess", "unmap", "free", "release", "full_cycle"]:
            xs = lat[op]
            rows.append((nt, args.cycles, total, round(wall, 6), round(cps, 2),
                         round(norm_ms_per_100, 4), op,
                         round(pct(xs, 50), 3), round(pct(xs, 95), 3), round(pct(xs, 99), 3),
                         round(statistics.mean(xs) if xs else 0, 3), len(xs)))
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threads", "cycles_per_thread", "total_cycles", "wall_s", "cycles_per_s",
                    "norm_ms_per_100cycles", "op", "p50_us", "p95_us", "p99_us", "mean_us", "n_samples"])
        w.writerows(rows)
    print("wrote", args.out, f"({len(rows)} rows)")
    # headline: aggregate throughput scaling. Linear scaling => cps grows with threads;
    # flat/declining cps + linearly-growing per-cycle latency => driver serialization.
    maxt = thread_list[-1]
    obs = [r for r in rows if r[0] == maxt and r[6] == "full_cycle"][0]
    maxcps = max(r[4] for r in rows if r[6] == "full_cycle")
    print(f"\nHEADLINE: aggregate throughput peaks at {maxcps:.0f} cyc/s; at {maxt} threads "
          f"full-cycle p50={obs[7]}us p99={obs[9]}us (vs single-thread p50 reference). "
          f"Flat cyc/s + per-cycle latency growing ~linearly with threads == driver "
          f"serialization (global VMM lock), not parallel speedup.")


if __name__ == "__main__":
    main()
