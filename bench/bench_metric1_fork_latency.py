"""
Metric 1: Fork latency vs prefix length. CoW fork should be ~flat (O(#pages) map ops,
no byte copy); full-clone should grow linearly (copies every prefix byte).

Output: data/metric1_fork_latency.csv  (prefix_pages, prefix_mib, method, rep, latency_us)
N replications reported with mean/stddev. Microbenchmark.
"""
import sys, os, time, csv, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kv_branch_manager import KVBranchManager
from baseline_fullclone import FullCloneManager
from explog import log

PREFIX_PAGES = [1, 2, 4, 8, 16, 32, 64, 128]   # x 2MiB = up to 256 MiB prefix
REPS = 10
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "metric1_fork_latency.csv")

def time_cow(n):
    m = KVBranchManager(device_id=0)
    m.create_branch("p", n); m.fill_prefix("p", n, 7)
    snap = m.snapshot("p")
    lats = []
    for r in range(REPS):
        h = m.fork(snap, f"c{r}")
        lats.append(h.fork_latency_s * 1e6)
    return lats

def time_clone(n):
    m = FullCloneManager(device_id=0)
    m.create_filled_branch("p", n, 7)
    lats = []
    for r in range(REPS):
        dt = m.clone("p", f"c{r}")
        lats.append(dt * 1e6)
    return lats

def main():
    rows = []
    for n in PREFIX_PAGES:
        cow = time_cow(n)
        clone = time_clone(n)
        mib = n * 2
        for r,(a,b) in enumerate(zip(cow, clone)):
            rows.append((n, mib, "cow_fork", r, a))
            rows.append((n, mib, "full_clone", r, b))
        print(f"prefix={n:4d}p ({mib:4d}MiB)  cow={statistics.mean(cow):8.1f}us "
              f"(sd {statistics.pstdev(cow):6.1f})  clone={statistics.mean(clone):8.1f}us "
              f"(sd {statistics.pstdev(clone):6.1f})")
        log("metric1_fork_latency",
            dict(prefix_pages=n, prefix_mib=mib, reps=REPS, page_size_mib=2),
            dict(cow_us_mean=statistics.mean(cow), cow_us_sd=statistics.pstdev(cow),
                 clone_us_mean=statistics.mean(clone), clone_us_sd=statistics.pstdev(clone)))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["prefix_pages","prefix_mib","method","rep","latency_us"])
        w.writerows(rows)
    print("wrote", OUT)

if __name__ == "__main__":
    main()
