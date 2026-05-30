"""
Lab 1 — Linux VMA count vs driver-internal mapping ceiling.

QUESTION: Is the K ~= 520K branch ceiling (Metric 4b) gated by the kernel's
per-process VMA limit (vm.max_map_count), or by something inside the NVIDIA driver?

METHOD: Fork CoW branches off a 12-GiB prefix until cuMemSetAccess OOMs (the same
forensic OOM observed in Metric 4b). Every N branches, sample /proc/self/maps line
count (a proxy for the process's VMA count, which is what vm.max_map_count gates).
Record (branches, vma_count, live_gib).

PREDICTION: If the ceiling were the Linux VMA limit, vma_count at OOM would equal
~vm.max_map_count. If it is driver-internal, vma_count at OOM will sit far below
vm.max_map_count, and the failing call will be cuMemSetAccess (driver-side), not a
kernel mmap returning ENOMEM/ENOMEM-like.

OUTPUT: data/lab1_vmmap_count.csv
  columns: branches, vma_count, live_gib, oom (only last row True)
"""
import os, sys, csv, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "lab1_vmmap_count.csv")
PREFIX_GIB = 12
PAGES_PER_GIB = 512
HARD_CAP_BRANCHES = 200          # known ceiling ~84 for 12 GiB
SAMPLE_EVERY = 4

def vma_count():
    with open("/proc/self/maps") as f:
        return sum(1 for _ in f)

def sysctl_max_map_count():
    with open("/proc/sys/vm/max_map_count") as f:
        return int(f.read().strip())

def main():
    from kv_branch_manager import KVBranchManager
    from vmm_pool import CudaCallError

    pages = PREFIX_GIB * PAGES_PER_GIB
    max_map = sysctl_max_map_count()
    print(f"[lab1] vm.max_map_count = {max_map:,}")
    print(f"[lab1] prefix = {PREFIX_GIB} GiB ({pages} pages)")
    print(f"[lab1] vma_count BEFORE manager init: {vma_count():,}")

    m = KVBranchManager(device_id=0, max_pages_per_branch=pages)
    m.pool.va_pool_enabled = False
    m.create_branch("p", pages)
    m.fill_prefix("p", pages, 7)
    snap = m.snapshot("p")

    print(f"[lab1] vma_count AFTER prefix fill (pre-fork): {vma_count():,}")

    rows = [("branches", "vma_count", "live_gib", "oom", "oom_call")]
    n = 0
    oom = False
    fail = "none"
    # Initial sample
    live = len(m.pool.pages) * m.page_size / 2**30
    rows.append((0, vma_count(), f"{live:.3f}", False, "none"))

    try:
        for i in range(HARD_CAP_BRANCHES):
            m.fork(snap, f"c{i}")
            n += 1
            if n % SAMPLE_EVERY == 0:
                live = len(m.pool.pages) * m.page_size / 2**30
                vc = vma_count()
                rows.append((n, vc, f"{live:.3f}", False, "none"))
                print(f"[lab1] branches={n:>4} vma={vc:>8,} live={live:.2f}GiB")
    except CudaCallError as e:
        oom = e.is_oom
        fail = e.call_site
        if not oom:
            raise
    except RuntimeError as e:
        s = str(e).lower()
        oom = ("out of memory" in s or "error 2" in s)
        fail = "RuntimeError"
        if not oom:
            raise

    live = len(m.pool.pages) * m.page_size / 2**30
    vc_oom = vma_count()
    rows.append((n, vc_oom, f"{live:.3f}", oom, fail))
    print(f"[lab1] OOM at branches={n} vma={vc_oom:,} live={live:.2f}GiB call={fail}")
    print(f"[lab1] vm.max_map_count={max_map:,}  vma_at_oom={vc_oom:,}  "
          f"headroom={max_map - vc_oom:,} ({100*(max_map-vc_oom)/max_map:.2f}%)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[lab1] wrote {OUT}")

    # Also dump a tiny summary file for easy citation
    summ = os.path.join(os.path.dirname(OUT), "lab1_vmmap_summary.txt")
    with open(summ, "w") as f:
        f.write(f"vm.max_map_count={max_map}\n")
        f.write(f"prefix_gib={PREFIX_GIB}\n")
        f.write(f"prefix_pages={pages}\n")
        f.write(f"branches_at_oom={n}\n")
        f.write(f"vma_at_oom={vc_oom}\n")
        f.write(f"oom_call={fail}\n")
        f.write(f"oom={oom}\n")
        f.write(f"K_product={n*pages}\n")
        f.write(f"vma_headroom={max_map - vc_oom}\n")
        f.write(f"vma_utilization_pct={100.0*vc_oom/max_map:.4f}\n")
    print(f"[lab1] wrote {summ}")

if __name__ == "__main__":
    main()
