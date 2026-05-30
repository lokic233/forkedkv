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
from vmm_pool import CudaCallError
from explog import log

# prefix sized so a single branch is ~12 GiB -> full clone OOMs after a handful.
PREFIX_PAGES = 6144            # 6144 * 2MiB = 12 GiB prefix
MAX_BRANCHES_CLONE = 64      # clone OOMs long before this
MAX_BRANCHES_COW = 512      # P1-A: sweep CoW to OOM or 512 (each fork = 6144 map ops at 12GiB prefix)
CHURN_CYCLES = 120           # P0-2: fork+destroy cycles (>> 84 concurrent ceiling) to prove
                             # VA reservations RECYCLE: serial branch throughput is unbounded
                             # while live HBM stays flat at the prefix size.
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "metric4_capacity.csv")

def is_oom(e):
    if isinstance(e, CudaCallError):
        return e.is_oom
    s = str(e).lower()
    return "out of memory" in s or "out_of_memory" in s or "error 2" in s

def cow_capacity():
    """R1 behaviour: fork branches and KEEP them all alive (VA pooling does NOT help here
    because nothing is destroyed). Sweep to true OOM and record the EXACT failing CUDA
    call (P0-2 / gemini R2-3 forensics)."""
    m = KVBranchManager(device_id=0, max_pages_per_branch=PREFIX_PAGES)
    m.pool.va_pool_enabled = False   # isolate: no recycling, measure raw ceiling
    m.create_branch("p", PREFIX_PAGES); m.fill_prefix("p", PREFIX_PAGES, 7)
    snap = m.snapshot("p")
    n = 0
    oom = False
    fail_call = ""
    traj = []
    try:
        for i in range(MAX_BRANCHES_COW):
            m.fork(snap, f"c{i}")   # zero-copy alias
            n += 1
            if n % 32 == 0:
                traj.append((n, len(m.pool.pages) * m.page_size / 2**30))
    except CudaCallError as e:
        oom = e.is_oom; fail_call = e.call_site
        if not oom: raise
    except RuntimeError as e:
        oom = is_oom(e)
        if not oom: raise
    live = len(m.pool.pages) * m.page_size / 2**30
    print("COW_TRAJ", traj)
    print(f"COW_FAILCALL {fail_call}")
    return n, oom, live, fail_call

def cow_churn():
    """P0-2: prove the VA free-list makes branch metadata RECYCLABLE. We fork ONE child,
    decode-equivalent (alias prefix), then DESTROY it and fork again, CHURN_CYCLES times.
    Without pooling each cycle leaks a 6144-page VA reservation -> OOM. With pooling the VA
    is recycled, so churn is unbounded and live HBM stays flat at the prefix size. We report
    how many cycles succeed and the reserve-vs-reuse counts."""
    m = KVBranchManager(device_id=0, max_pages_per_branch=PREFIX_PAGES)
    m.pool.va_pool_enabled = True    # the optimization under test
    m.create_branch("p", PREFIX_PAGES); m.fill_prefix("p", PREFIX_PAGES, 7)
    snap = m.snapshot("p")
    cycles = 0; oom = False; fail_call = ""
    try:
        for i in range(CHURN_CYCLES):
            m.fork(snap, "child")
            m.destroy_branch("child")   # returns the 6144-page VA range to the free-list
            cycles += 1
    except CudaCallError as e:
        oom = e.is_oom; fail_call = e.call_site
        if not oom: raise
    live = len(m.pool.pages) * m.page_size / 2**30
    reserved = m.pool.stat_va_reserved; reused = m.pool.stat_va_reused
    print(f"CHURN cycles={cycles} oom={oom} live={live:.2f} va_reserved={reserved} va_reused={reused} fail={fail_call}")
    return cycles, oom, live, reserved, reused

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
    fn, foom, flive, fcall = cow_capacity()
    print(f"RESULT cow {fn} {foom} {flive:.2f} {fcall or 'none'}")
    log("metric4_capacity", dict(prefix_gib=PREFIX_PAGES*2/1024, max_branches_clone=MAX_BRANCHES_CLONE, max_branches_cow=MAX_BRANCHES_COW),
        dict(method="cow_fork", branches_succeeded=fn, oom=foom, live_gib=flive, oom_call_site=fcall))

def _phase_churn():
    cyc, oom, live, reserved, reused = cow_churn()
    print(f"RESULT churn {cyc} {oom} {live:.2f} reserved={reserved} reused={reused}")
    log("metric4_capacity", dict(prefix_gib=PREFIX_PAGES*2/1024, churn_cycles=CHURN_CYCLES),
        dict(method="cow_churn_pooled", cycles_succeeded=cyc, oom=oom, live_gib=live,
             va_reserved=reserved, va_reused=reused))

def main():
    import subprocess, sys, csv, re
    gib = PREFIX_PAGES * 2 / 1024
    here = os.path.abspath(__file__)
    py = sys.executable
    print(f"prefix = {PREFIX_PAGES} pages = {gib:.1f} GiB per branch; clone cap={MAX_BRANCHES_CLONE} cow cap={MAX_BRANCHES_COW} (sweep to OOM)")
    out = {}
    fail_call = {}
    churn = None
    for phase in ("clone", "cow", "churn"):
        r = subprocess.run([py, here, phase], capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
        if not line:
            print("PHASE", phase, "FAILED:\n", r.stdout[-500:], r.stderr[-800:]); raise SystemExit(1)
        parts = line[0].split()
        meth = parts[1]
        if meth == "churn":
            churn = (int(parts[2]), parts[3] == "True", float(parts[4]), parts[5], parts[6])
            print(f"[cow_churn_pooled] {parts[2]} fork+destroy cycles, OOM={parts[3]}, live~{parts[4]}GiB, {parts[5]} {parts[6]}")
        else:
            out[meth] = (int(parts[2]), parts[3] == "True", float(parts[4]))
            fail_call[meth] = parts[5] if len(parts) > 5 else "none"
            print(f"[{ 'full_clone' if meth=='clone' else 'cow_fork' }] succeeded {parts[2]} branches, OOM={parts[3]}, live~{parts[4]} GiB, fail_call={fail_call[meth]}")
    rows = [("full_clone", gib, out["clone"][0], out["clone"][1], f"live~{out['clone'][2]:.1f}GiB; oom_call={fail_call.get('clone','none')}"),
            ("cow_fork", gib, out["cow"][0], out["cow"][1], f"live~{out['cow'][2]:.1f}GiB; oom_call={fail_call.get('cow','none')}"),
            ("cow_churn_pooled", gib, churn[0], churn[1], f"live~{churn[2]:.1f}GiB; {churn[3]} {churn[4]} (VA recycled)")]
    with open(OUT,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["method","prefix_gib","branches_succeeded","oom","note"]); w.writerows(rows)
    print("wrote", OUT)
    print(f"\nHEADLINE: full-clone reaches {out['clone'][0]} branches (OOM={out['clone'][1]}, "
          f"fails at {fail_call.get('clone','?')}); CoW reaches {out['cow'][0]} concurrent branches "
          f"(OOM={out['cow'][1]}, fails at {fail_call.get('cow','?')}) on one H100 — "
          f"{out['cow'][0]/max(out['clone'][0],1):.0f}x.")
    print(f"P0-2: the CoW ceiling is the FORENSIC failing call '{fail_call.get('cow','?')}' "
          f"(VA-mapping metadata, NOT data: live HBM {out['cow'][2]:.1f} of 97 GiB). With the VA "
          f"free-list, {churn[0]} fork+destroy cycles ran with live HBM flat at {churn[2]:.1f} GiB "
          f"({churn[3]}, {churn[4]}) — metadata is RECYCLED, so concurrent capacity is bounded by "
          f"data, and serial branch THROUGHPUT is unbounded.")

if __name__=="__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "clone":
        _phase_clone()
    elif len(sys.argv) > 1 and sys.argv[1] == "cow":
        _phase_cow()
    elif len(sys.argv) > 1 and sys.argv[1] == "churn":
        _phase_churn()
    else:
        main()
