"""
Metric 4b (P0-2 R3): CONCURRENCY CEILING MODEL.

Metric 4 (R2) measured ONE prefix size (12 GiB -> 84 concurrent CoW branches before the
driver OOMs at cuMemSetAccess). Reviewer (metacode) R3 ask: prove the ceiling is
PREDICTABLE by sweeping prefix sizes and fitting the relationship between prefix size and
max concurrent branches.

Mechanism recap: each CoW fork aliases the prefix's physical pages by issuing one
cuMemMap + cuMemSetAccess PER PAGE into a fresh VA reservation. NO KV bytes are copied
(live HBM stays at the single prefix size). The ceiling is therefore NOT data -- it is the
driver's per-device MAPPING-TABLE capacity: the total number of (VA page -> phys handle)
access descriptors it will accept. With P prefix pages per branch and B branches, total
mappings = B * P. The driver caps total mappings at a constant K, so:

    max_branches(P) ~= K / P            (P = prefix pages per branch)

We sweep P over {1, 3, 6, 12} GiB (512, 1536, 3072, 6144 pages), measure max_branches
before the forensic OOM, and fit K = median(max_branches * prefix_pages). If the product
is roughly constant across prefix sizes, the ceiling is the predictable mapping-table
limit (and live HBM stays flat at one prefix the whole time).

Each prefix size runs in a FRESH subprocess: a driver OOM can leave the CUDA context in
an undefined state, so we never reuse a context after provoking OOM.

Output: data/metric4b_ceiling.csv
  columns: prefix_gib, prefix_pages, max_branches, oom, oom_call, live_gib, K_product
"""
import sys, os, csv, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "metric4b_ceiling.csv")
SWEEP_GIB = [1, 3, 6, 12]            # prefix sizes to sweep
PAGES_PER_GIB = 512                  # 2 MiB pages -> 512 pages per GiB
HARD_CAP_BRANCHES = 4096             # safety cap so a 1 GiB prefix doesn't run forever


def _child(prefix_pages):
    """Run in a fresh process: fork CoW branches off one prefix until the driver OOMs,
    print the max branches and the forensic failing call."""
    from kv_branch_manager import KVBranchManager
    from vmm_pool import CudaCallError
    m = KVBranchManager(device_id=0, max_pages_per_branch=prefix_pages)
    m.pool.va_pool_enabled = False          # isolate raw ceiling, no VA recycling
    m.create_branch("p", prefix_pages); m.fill_prefix("p", prefix_pages, 7)
    snap = m.snapshot("p")
    n = 0; oom = False; fail = "none"
    try:
        for i in range(HARD_CAP_BRANCHES):
            m.fork(snap, f"c{i}")           # zero-copy alias: prefix_pages map ops
            n += 1
    except CudaCallError as e:
        oom = e.is_oom; fail = e.call_site
        if not oom: raise
    except RuntimeError as e:
        s = str(e).lower()
        oom = ("out of memory" in s or "error 2" in s)
        if not oom: raise
    live = len(m.pool.pages) * m.page_size / 2**30
    print(f"CHILD {prefix_pages} {n} {oom} {fail} {live:.3f}")


def main():
    from explog import log
    here = os.path.abspath(__file__); py = sys.executable
    rows = []
    for gib in SWEEP_GIB:
        pages = gib * PAGES_PER_GIB
        r = subprocess.run([py, here, "child", str(pages)], capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("CHILD")]
        if not line:
            print(f"prefix {gib}GiB FAILED:\n", r.stdout[-400:], r.stderr[-800:]); raise SystemExit(1)
        _, p, n, oom, fail, live = line[0].split()
        pages = int(p); n = int(n); oom = (oom == "True"); live = float(live)
        K = n * pages
        print(f"[{gib:>2} GiB / {pages:>4} pages] max_branches={n:>4} oom={oom} "
              f"fail={fail} live={live:.2f}GiB  K=branches*pages={K}")
        rows.append((gib, pages, n, oom, fail, f"{live:.2f}", K))
        log("metric4b_ceiling", dict(prefix_gib=gib, prefix_pages=pages),
            dict(max_branches=n, oom=oom, oom_call=fail, live_gib=live, K_product=K))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prefix_gib", "prefix_pages", "max_branches", "oom", "oom_call",
                    "live_gib", "K_product"])
        w.writerows(rows)
    print("wrote", OUT)
    Ks = [r[6] for r in rows]
    import statistics
    Kmed = int(statistics.median(Ks)); Kmin = min(Ks); Kmax = max(Ks)
    spread = 100.0 * (Kmax - Kmin) / Kmed if Kmed else 0
    print(f"\nFIT: max_branches ~= K / prefix_pages")
    print(f"  K (branches*pages) per sweep point: {Ks}")
    print(f"  K median={Kmed}  range=[{Kmin},{Kmax}]  spread={spread:.0f}% of median")
    print(f"  => CEILING MODEL: max_branches ~= {Kmed} / prefix_pages")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "child":
        _child(int(sys.argv[2]))
    else:
        main()
