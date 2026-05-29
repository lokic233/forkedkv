"""
bench_cow_overhead.py  (B5) — Decompose the cost of a single CoW event.

LIMITATIONS.md #2 noted (but did NOT measure) that our `_cow` uses a temporary scratch
VA window + cuMemcpyDtoD, adding 2 extra map ops vs an ideal in-place scheme. This
microbenchmark MEASURES that overhead so we can state it honestly.

A CoW event in KVBranchManager._cow does, per page:
  1. create_phys_page            (cuMemCreate)
  2. reserve scratch VA          (cuMemAddressReserve)         <- "extra" vs ideal
  3. map scratch                 (cuMemMap + cuMemSetAccess)   <- "extra" vs ideal
  4. D2D copy 2 MiB              (cuMemcpyDtoD)                 <- the real data movement
  5. unmap scratch               (cuMemUnmap)                  <- "extra" vs ideal
  6. free scratch VA             (cuMemAddressFree)            <- "extra" vs ideal
  7. unmap branch VA             (cuMemUnmap)
  8. map branch VA -> new page   (cuMemMap + cuMemSetAccess)

We time the WHOLE _cow path, and separately time JUST the D2D copy (cuMemcpyDtoD of one
page), to report what fraction of CoW latency is the scratch-VA bookkeeping vs the
unavoidable byte copy.

Output: data/cow_overhead.csv  columns: component, ns_median, ns_mean, ns_sd, reps
Microbenchmark, CUDA-event + perf_counter timed, n>=200.
"""
import sys, os, csv, statistics, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vmm_pool import VMMPool, _ck
from cuda import cuda

REPS = 300
WARMUP = 50
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "cow_overhead.csv")


def time_full_cow(pool, src_va):
    """One full _cow path (steps 1-8 above) against a fresh dst slot each rep."""
    # dst branch VA (the page we are 'CoW-ing'): reserve+map a page mapped to a shared-ish page
    dst_va, dst_sz = pool.reserve_va_range(1)
    base_pg = pool.create_phys_page()
    pool.map_page(dst_va, base_pg)
    pool.copy_page(dst_va, src_va)   # give it contents to copy (not timed region below)

    t0 = time.perf_counter()
    # --- mirror KVBranchManager._cow exactly ---
    new_pg = pool.create_phys_page()
    tmp_va, tmp_sz = pool.reserve_va_range(1)
    pool.map_page(tmp_va, new_pg)
    pool.copy_page(tmp_va, dst_va)
    pool.unmap_page(tmp_va)
    pool.free_va(tmp_va, tmp_sz)
    pool.unmap_page(dst_va)
    pool.map_page(dst_va, new_pg)
    pool.synchronize()
    dt = time.perf_counter() - t0
    # cleanup
    pool.unmap_page(dst_va); pool.free_va(dst_va, dst_sz)
    pool.decref(base_pg); pool.decref(new_pg)
    return dt * 1e9  # ns


def time_d2d_only(pool, src_va):
    """JUST the unavoidable 2 MiB D2D copy (step 4): cuMemcpyDtoD into a pre-mapped page."""
    dst_va, dst_sz = pool.reserve_va_range(1)
    pg = pool.create_phys_page()
    pool.map_page(dst_va, pg)
    pool.synchronize()
    t0 = time.perf_counter()
    pool.copy_page(dst_va, src_va)
    pool.synchronize()
    dt = time.perf_counter() - t0
    pool.unmap_page(dst_va); pool.free_va(dst_va, dst_sz); pool.decref(pg)
    return dt * 1e9


def time_scratch_only(pool):
    """JUST the scratch-VA bookkeeping (steps 2,3,5,6): reserve+map+unmap+free a page."""
    pg = pool.create_phys_page()
    pool.synchronize()
    t0 = time.perf_counter()
    tmp_va, tmp_sz = pool.reserve_va_range(1)
    pool.map_page(tmp_va, pg)
    pool.unmap_page(tmp_va)
    pool.free_va(tmp_va, tmp_sz)
    pool.synchronize()
    dt = time.perf_counter() - t0
    pool.decref(pg)
    return dt * 1e9


def measure(fn, *a):
    pool = VMMPool(device_id=0)
    src_pg = pool.create_phys_page()
    src_va, src_sz = pool.reserve_va_range(1)
    pool.map_page(src_va, src_pg)
    pool.memset_page(src_va, 7)
    pool.synchronize()
    samples = []
    for i in range(WARMUP + REPS):
        if fn is time_scratch_only:
            v = fn(pool)
        else:
            v = fn(pool, src_va)
        if i >= WARMUP:
            samples.append(v)
    pool.unmap_page(src_va); pool.free_va(src_va, src_sz)
    pool.destroy()
    return samples


def main():
    rows = []
    specs = [("full_cow", time_full_cow),
             ("d2d_copy_only", time_d2d_only),
             ("scratch_va_bookkeeping_only", time_scratch_only)]
    res = {}
    for name, fn in specs:
        s = measure(fn)
        med, mean, sd = statistics.median(s), statistics.mean(s), statistics.pstdev(s)
        res[name] = (med, mean, sd)
        rows.append((name, med, mean, sd, len(s)))
        print(f"{name:32s} median={med/1e3:9.2f}us  mean={mean/1e3:9.2f}us  sd={sd/1e3:7.2f}us  n={len(s)}")
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["component", "ns_median", "ns_mean", "ns_sd", "reps"])
        w.writerows(rows)
    print("wrote", OUT)
    full = res["full_cow"][0]; d2d = res["d2d_copy_only"][0]; scr = res["scratch_va_bookkeeping_only"][0]
    print(f"\nHEADLINE: full CoW median {full/1e3:.1f}us; unavoidable D2D copy {d2d/1e3:.1f}us "
          f"({100*d2d/full:.0f}% of CoW); scratch-VA bookkeeping {scr/1e3:.1f}us "
          f"({100*scr/full:.0f}% of CoW). The scratch path is the 'extra 2 map ops' overhead "
          f"from LIMITATIONS #2; it could be removed by mapping the new page directly at a "
          f"reusable scratch VA, but is NOT currently optimized.")


if __name__ == "__main__":
    main()
