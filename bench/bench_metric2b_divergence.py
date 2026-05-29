"""
Metric 2b: Realistic bytes-written reduction when branches DIVERGE.
Each forked branch writes (diverges) a fraction of its prefix pages -> triggers CoW.
This is the honest, non-zero-divergence number. Full-clone always writes the full
prefix regardless. Reduction = 1 - cow_bytes/clone_bytes.

Output: data/metric2b_divergence.csv
columns: divergence_frac, fanout, prefix_pages, method, bytes_written, reduction_pct
"""
import sys, os, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kv_branch_manager import KVBranchManager
from baseline_fullclone import FullCloneManager
from explog import log

PREFIX_PAGES = 32
FANOUT = 16
DIVERGENCE = [0.0, 0.05, 0.10, 0.25, 0.50, 1.0]
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "metric2b_divergence.csv")

def run_cow(frac):
    m = KVBranchManager(device_id=0)
    m.create_branch("p", PREFIX_PAGES); m.fill_prefix("p", PREFIX_PAGES, 7)
    snap = m.snapshot("p")
    bc = m.pool.stat_bytes_created; bk = m.pool.stat_bytes_copied
    npages_write = int(round(frac * PREFIX_PAGES))
    for i in range(FANOUT):
        m.fork(snap, f"c{i}")
        for p in range(npages_write):
            m.write_page(f"c{i}", p, fill_value=i % 251 + 1)  # triggers CoW
    written = (m.pool.stat_bytes_created - bc) + (m.pool.stat_bytes_copied - bk)
    return written

def run_clone(frac):
    m = FullCloneManager(device_id=0)
    m.create_filled_branch("p", PREFIX_PAGES, 7)
    bc = m.pool.stat_bytes_created; bk = m.pool.stat_bytes_copied
    for i in range(FANOUT):
        m.clone("p", f"c{i}")   # full copy regardless of divergence
    written = (m.pool.stat_bytes_created - bc) + (m.pool.stat_bytes_copied - bk)
    return written

def main():
    rows = []
    print(f"prefix={PREFIX_PAGES}p fanout={FANOUT}")
    for frac in DIVERGENCE:
        cb = run_cow(frac); bb = run_clone(frac)
        red = 100*(1 - cb/bb)
        rows.append((frac, FANOUT, PREFIX_PAGES, "cow_fork", cb, red))
        rows.append((frac, FANOUT, PREFIX_PAGES, "full_clone", bb, 0.0))
        print(f"divergence={frac*100:5.1f}%  cow_written={cb/2**20:8.1f}MiB  "
              f"clone_written={bb/2**20:8.1f}MiB  reduction={red:5.1f}%")
        log("metric2b_divergence", dict(divergence_frac=frac, fanout=FANOUT, prefix_pages=PREFIX_PAGES),
            dict(cow_bytes_written=cb, clone_bytes_written=bb, reduction_pct=red))
    with open(OUT,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["divergence_frac","fanout","prefix_pages","method","bytes_written","reduction_pct"])
        w.writerows(rows)
    print("wrote", OUT)

if __name__=="__main__":
    main()
