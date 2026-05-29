"""
Metric 2: HBM bytes physically written per fork, and total HBM footprint, vs branch
fanout. CoW writes 0 KV bytes on fork (only page-table maps); full-clone writes
prefix_bytes per fork. Headline reduction claim derives from THIS file.

Output: data/metric2_bytes_written.csv
columns: fanout, prefix_pages, prefix_mib, method, bytes_written, live_phys_mib
"""
import sys, os, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kv_branch_manager import KVBranchManager
from baseline_fullclone import FullCloneManager
from explog import log

PREFIX_PAGES = 32              # 64 MiB prefix
FANOUTS = [1, 2, 4, 8, 16, 32]
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "metric2_bytes_written.csv")

def run_cow(fanout, n):
    m = KVBranchManager(device_id=0)
    m.create_branch("p", n); m.fill_prefix("p", n, 7)
    base_created = m.pool.stat_bytes_created
    base_copied = m.pool.stat_bytes_copied
    snap = m.snapshot("p")
    for i in range(fanout):
        m.fork(snap, f"c{i}")
    # bytes physically written DUE TO FORKS = new phys allocations + copies after base
    forked_bytes = (m.pool.stat_bytes_created - base_created) + (m.pool.stat_bytes_copied - base_copied)
    live_mib = len(m.pool.pages) * m.page_size / 2**20
    return forked_bytes, live_mib

def run_clone(fanout, n):
    m = FullCloneManager(device_id=0)
    m.create_filled_branch("p", n, 7)
    base_created = m.pool.stat_bytes_created
    base_copied = m.pool.stat_bytes_copied
    for i in range(fanout):
        m.clone("p", f"c{i}")
    forked_bytes = (m.pool.stat_bytes_created - base_created) + (m.pool.stat_bytes_copied - base_copied)
    live_mib = len(m.pool.pages) * m.page_size / 2**20
    return forked_bytes, live_mib

def main():
    rows = []
    print(f"prefix = {PREFIX_PAGES} pages ({PREFIX_PAGES*2} MiB)")
    for fo in FANOUTS:
        cb, cm = run_cow(fo, PREFIX_PAGES)
        bb, bm = run_clone(fo, PREFIX_PAGES)
        red = 100*(1 - cb/bb) if bb else float('nan')
        rows.append((fo, PREFIX_PAGES, PREFIX_PAGES*2, "cow_fork", cb, cm))
        rows.append((fo, PREFIX_PAGES, PREFIX_PAGES*2, "full_clone", bb, bm))
        print(f"fanout={fo:3d}  cow_written={cb/2**20:8.1f}MiB live={cm:8.1f}MiB | "
              f"clone_written={bb/2**20:8.1f}MiB live={bm:8.1f}MiB | "
              f"bytes-written reduction={red:5.1f}%")
        log("metric2_bytes_written",
            dict(fanout=fo, prefix_pages=PREFIX_PAGES, prefix_mib=PREFIX_PAGES*2),
            dict(cow_bytes_written=cb, clone_bytes_written=bb,
                 cow_live_mib=cm, clone_live_mib=bm,
                 bytes_written_reduction_pct=red))
    with open(OUT,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["fanout","prefix_pages","prefix_mib","method","bytes_written","live_phys_mib"])
        w.writerows(rows)
    print("wrote", OUT)

if __name__=="__main__":
    main()
