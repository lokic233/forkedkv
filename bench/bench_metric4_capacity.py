"""
Metric 4: Capacity — how many branches fit on ONE H100 before OOM.
We size the prefix to a realistic large agent context so full-clone OOMs at a small N,
then show CoW scales far past it (sharing the prefix).

We deliberately pick a prefix that makes full-clone OOM in the single-GPU budget.
H100 free ~95 GiB. We cap our VMM pool usage well under that and let cuMemCreate fail
(OOM) naturally; we catch the driver OOM and record the last successful N.

Output: data/metric4_capacity.csv  columns: method, prefix_gib, branches_succeeded, oom, note
"""
import sys, os, csv, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kv_branch_manager import KVBranchManager
from baseline_fullclone import FullCloneManager
from explog import log

# prefix sized so a single branch is ~12 GiB -> full clone OOMs after a handful.
PREFIX_PAGES = 6144            # 6144 * 2MiB = 12 GiB prefix
MAX_BRANCHES_CLONE = 64      # clone OOMs long before this
MAX_BRANCHES_COW = 4096      # P1-A: sweep CoW until it actually OOMs (VA / handle / HBM)
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "metric4_capacity.csv")

def is_oom(e):
    s = str(e).lower()
    return "out of memory" in s or "out_of_memory" in s or "error 2" in s

def cow_capacity():
    m = KVBranchManager(device_id=0, max_pages_per_branch=PREFIX_PAGES)
    m.create_branch("p", PREFIX_PAGES); m.fill_prefix("p", PREFIX_PAGES, 7)
    snap = m.snapshot("p")
    n = 0
    oom = False
    try:
        for i in range(MAX_BRANCHES_COW):
            m.fork(snap, f"c{i}")   # zero-copy alias
            n += 1
    except RuntimeError as e:
        oom = is_oom(e)
        if not oom: raise
    live = len(m.pool.pages) * m.page_size / 2**30
    return n, oom, live

def clone_capacity():
    m = FullCloneManager(device_id=0)
    m.create_filled_branch("p", PREFIX_PAGES, 7)
    n = 0
    oom = False
    try:
        for i in range(MAX_BRANCHES_CLONE):
            m.clone("p", f"c{i}")   # full 12 GiB copy each
            n += 1
    except RuntimeError as e:
        oom = is_oom(e)
        if not oom: raise
    live = len(m.pool.pages) * m.page_size / 2**30
    return n, oom, live

def _phase_clone():
    cn, coom, clive = clone_capacity()
    print(f"RESULT clone {cn} {coom} {clive:.2f}")
    log("metric4_capacity", dict(prefix_gib=PREFIX_PAGES*2/1024, max_branches_clone=MAX_BRANCHES_CLONE, max_branches_cow=MAX_BRANCHES_COW),
        dict(method="full_clone", branches_succeeded=cn, oom=coom, live_gib=clive))

def _phase_cow():
    fn, foom, flive = cow_capacity()
    print(f"RESULT cow {fn} {foom} {flive:.2f}")
    log("metric4_capacity", dict(prefix_gib=PREFIX_PAGES*2/1024, max_branches_clone=MAX_BRANCHES_CLONE, max_branches_cow=MAX_BRANCHES_COW),
        dict(method="cow_fork", branches_succeeded=fn, oom=foom, live_gib=flive))

def main():
    import subprocess, sys, csv, re
    gib = PREFIX_PAGES * 2 / 1024
    here = os.path.abspath(__file__)
    py = sys.executable
    print(f"prefix = {PREFIX_PAGES} pages = {gib:.1f} GiB per branch; clone cap={MAX_BRANCHES_CLONE} cow cap={MAX_BRANCHES_COW} (sweep to OOM)")
    out = {}
    for phase in ("clone", "cow"):
        r = subprocess.run([py, here, phase], capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
        if not line:
            print("PHASE", phase, "FAILED:\n", r.stdout[-500:], r.stderr[-800:]); raise SystemExit(1)
        _, meth, n, oom, live = line[0].split()
        out[meth] = (int(n), oom=="True", float(live))
        print(f"[{ 'full_clone' if meth=='clone' else 'cow_fork' }] succeeded {n} branches, OOM={oom}, live~{live} GiB")
    rows = [("full_clone", gib, out["clone"][0], out["clone"][1], f"live~{out['clone'][2]:.1f}GiB"),
            ("cow_fork", gib, out["cow"][0], out["cow"][1], f"live~{out['cow'][2]:.1f}GiB")]
    with open(OUT,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["method","prefix_gib","branches_succeeded","oom","note"]); w.writerows(rows)
    print("wrote", OUT)
    print(f"\nHEADLINE: full-clone reaches {out['clone'][0]} branches (OOM={out['clone'][1]}); "
          f"CoW reaches {out['cow'][0]} branches (OOM={out['cow'][1]}) on one H100.")

if __name__=="__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "clone":
        _phase_clone()
    elif len(sys.argv) > 1 and sys.argv[1] == "cow":
        _phase_cow()
    else:
        main()
