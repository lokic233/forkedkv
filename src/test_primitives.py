"""Unit test of Priority-1 primitives: Snapshot, Fork, CoW, refcount, divergence.
Run: python src/test_primitives.py  (from repo root, with .venv active)
Proves the mechanism touches the GPU MMU (handle aliasing verified via driver)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import ctypes, numpy as np
from cuda import cuda
from kv_branch_manager import KVBranchManager
from vmm_pool import _ck

def read_page_byte(va, n=8):
    out = (ctypes.c_ubyte * n)()
    _ck(cuda.cuMemcpyDtoH(out, va, n))
    return list(out)

def main():
    m = KVBranchManager(device_id=0)
    PS = m.page_size
    print(f"[cfg] page_size = {PS} bytes ({PS//1024//1024} MiB)")

    # parent branch with a 4-page 'prefix' filled with value 7
    m.create_branch("parent", num_pages=8)
    m.fill_prefix("parent", num_pages=4, fill_value=7)
    print(f"[setup] parent has 4 mapped pages, stats={m.stats()}")

    # SNAPSHOT
    snap = m.snapshot("parent")
    assert snap.num_pages == 4
    print(f"[snapshot] sid={snap.snapshot_id} pages={snap.num_pages}")

    # FORK three children — must be byte-copy-free
    bytes_copied_before = m.pool.stat_bytes_copied
    f1 = m.fork(snap, "child1")
    f2 = m.fork(snap, "child2")
    f3 = m.fork(snap, "child3")
    assert m.pool.stat_bytes_copied == bytes_copied_before, "fork must NOT copy bytes"
    print(f"[fork] 3 children forked, zero KV bytes copied. "
          f"f1 latency={f1.fork_latency_s*1e6:.1f}us pages_aliased={f1.pages_aliased}")

    # ALIASING: parent page 0 and child1 page 0 share the SAME physical handle
    assert m.shared_handle("parent", "child1", 0), "fork should alias physical pages"
    print("[alias] parent.page0 and child1.page0 share one physical handle (driver-verified)")

    # refcount: page0 physical should have refcount = parent(1)+snap(1)+3 children = 5
    pg0 = m.branches["parent"].page_phys[0]
    print(f"[refcount] parent.page0 refcount = {pg0.refcount} (expect 5)")
    assert pg0.refcount == 5, pg0.refcount

    # verify aliased read: child1 page0 reads value 7 written by parent
    va_c1_p0 = m.branches["child1"].va_of(0)
    assert read_page_byte(va_c1_p0)[0] == 7
    print("[alias] child1 reads parent's data (7) through aliased page")

    # COW: write to child1 page0 -> triggers copy, diverges only that page
    cow_events_before = m.pool.stat_cow_events
    did = m.write_page("child1", 0, fill_value=99)
    assert did, "write to shared page must trigger CoW"
    assert m.pool.stat_cow_events == cow_events_before + 1
    print(f"[cow] write to child1.page0 triggered CoW (1 copy = {PS} bytes)")

    # after CoW: child1.page0 != parent.page0 physically; child1 reads 99, parent still 7
    assert not m.shared_handle("parent", "child1", 0), "CoW must break the alias"
    assert read_page_byte(m.branches["child1"].va_of(0))[0] == 99
    assert read_page_byte(m.branches["parent"].va_of(0))[0] == 7
    print("[cow] child1.page0=99, parent.page0=7 (isolation preserved)")

    # parent page0 refcount now 4 (lost child1)
    print(f"[refcount] parent.page0 refcount after CoW = {pg0.refcount} (expect 4)")
    assert pg0.refcount == 4

    # other pages of child1 still aliased (only the written page diverged)
    assert m.shared_handle("parent", "child1", 1)
    print("[cow] child1.page1 still shared with parent (only touched page diverged)")

    # DIVERGENCE DETECTOR
    div = m.diverged_pages("parent", "child1")
    print(f"[divergence] diverged pages parent vs child1: {div} (expect [0])")
    assert div == [0]
    assert m.diverged_pages("parent", "child2") == []
    print("[divergence] child2 (untouched) has zero diverged pages")

    print("\nALL PRIMITIVE TESTS PASSED")
    print("final stats:", m.stats())

if __name__ == "__main__":
    main()
