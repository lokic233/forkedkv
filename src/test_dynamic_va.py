"""
test_dynamic_va.py — P0-C: prove a branch can grow its KV tail dynamically AFTER fork,
independently of siblings, while preserving CoW sharing of the shared prefix.

Scenario (models real agentic decode):
  1. Parent fills a 4-page shared prefix, snapshots.
  2. Fork two children A and B from the snapshot (alias the prefix, zero copy).
  3. A and B each append_page() 3 times (decode 3 more "token pages") with DIFFERENT
     content -> their tails diverge while their prefix stays aliased.
Assertions:
  - prefix pages (0..3) of A and B alias the SAME physical handle (still shared).
  - appended tail pages are PRIVATE: A's tail handle != B's tail handle.
  - A and B grew to different lengths independently (A->7, B->6).
  - zero bytes copied for the prefix (CoW only fires on writes to shared pages, and
    appends create fresh private pages -> no copy).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from kv_branch_manager import KVBranchManager

def main():
    m = KVBranchManager(device_id=0)
    PREFIX = 4
    m.create_branch("parent", PREFIX, headroom_pages=8)
    m.fill_prefix("parent", PREFIX, fill_value=7)
    snap = m.snapshot("parent")

    bc0, bk0 = m.pool.stat_bytes_created, m.pool.stat_bytes_copied
    # fork with headroom so children can grow their tail
    m.fork(snap, "A", headroom_pages=8)
    m.fork(snap, "B", headroom_pages=8)

    # prefix is aliased (shared) right after fork -> zero copy
    for i in range(PREFIX):
        assert m.shared_handle("A", "B", i), f"prefix page {i} should be shared A/B"
        assert m.shared_handle("parent", "A", i), f"prefix page {i} should be shared parent/A"
    assert m.pool.stat_bytes_copied == bk0, "fork must copy zero KV bytes"

    # A appends 3 tail pages, B appends 2 -> independent growth
    a_idx = [m.append_page("A", fill_value=100 + k) for k in range(3)]
    b_idx = [m.append_page("B", fill_value=200 + k) for k in range(2)]

    assert m.branches["A"].num_pages == PREFIX + 3 == 7, m.branches["A"].num_pages
    assert m.branches["B"].num_pages == PREFIX + 2 == 6, m.branches["B"].num_pages

    # appended tails are PRIVATE: A page 4 handle != B page 4 handle
    ha = m.pool.retained_handle_at(m.branches["A"].va_of(4))
    hb = m.pool.retained_handle_at(m.branches["B"].va_of(4))
    assert ha != hb, "appended tail pages must be private (different handles)"

    # prefix STILL shared after growth (growth did not disturb the prefix)
    for i in range(PREFIX):
        assert m.shared_handle("A", "B", i), f"prefix page {i} should STILL be shared after growth"

    # Now write into a SHARED prefix page of A -> must trigger CoW (tail growth did not)
    bk1 = m.pool.stat_bytes_copied
    did = m.write_page("A", 0, fill_value=42)
    assert did, "writing a shared prefix page should trigger CoW"
    assert m.pool.stat_bytes_copied == bk1 + m.page_size, "CoW copies exactly one page"
    assert not m.shared_handle("A", "B", 0), "after CoW, A.page0 must diverge from B.page0"

    print("PASS: dynamic VA growth (append_page) preserves CoW prefix sharing")
    print(f"  A grew to {m.branches['A'].num_pages} pages, B to {m.branches['B'].num_pages}; "
          f"prefix {PREFIX} pages shared until write; CoW fired only on explicit write.")
    m.pool.destroy()

if __name__ == "__main__":
    main()
