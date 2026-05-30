"""
kv_branch_manager.py — Branch-aware copy-on-write KV-cache manager over CUDA VMM.

A *branch* is one agent-execution path. Its KV cache is a list of VA pages, each
mapped (cuMemMap) to a physical page (PhysPage). Branches share physical pages by
mapping their VA pages to the same handle (refcount tracked in VMMPool).

Operations
----------
Snapshot(branch_id) -> SnapshotHandle
    Atomic checkpoint of a branch's KV state at a causal boundary. Records, per page,
    the physical page it currently maps and the logical token-length filled. Marks all
    those pages as shared (snapshot holds a ref). O(#pages), no byte copy.

Fork(snapshot, new_branch_id) -> ForkHandle
    Reserve a new VA range for the child, map each child VA page to the SAME physical
    page as the snapshot (refcount++). No KV bytes copied. O(#pages).

write(branch_id, page_index, ...) -> triggers CoW if the target page is shared
    Software page-fault-on-write: if the physical page backing (branch, page_index)
    has refcount>1, allocate a private page, copy 2 MiB, remap that one VA page,
    decref the shared page. Only the touched page diverges.

Divergence detector: compares two branches page-by-page by retained driver handle;
pages with different handles have physically diverged.
"""
import time
import numpy as np
from cuda import cuda
from vmm_pool import VMMPool, _ck


class Branch:
    __slots__ = ("branch_id", "va_base", "va_size", "num_pages", "capacity",
                 "page_phys", "page_va", "tokens_filled", "parent")
    def __init__(self, branch_id, va_base, va_size, num_pages, capacity=None):
        self.branch_id = branch_id
        self.va_base = va_base
        self.va_size = va_size
        # capacity = number of VA page slots reserved (headroom for dynamic growth).
        # num_pages = number of slots currently considered "active" (mapped or to-be).
        self.capacity = capacity if capacity is not None else num_pages
        self.num_pages = num_pages
        self.page_phys = [None] * self.capacity   # PhysPage per slot (None = unmapped)
        self.page_va = [va_base + i * self.page_size_unit() for i in range(self.capacity)]
        self.tokens_filled = 0
        self.parent = None

    def page_size_unit(self):
        # all slots are page_size; va_size = capacity * page_size
        return self.va_size // self.capacity

    def va_of(self, page_index):
        return self.page_va[page_index]


class SnapshotHandle:
    __slots__ = ("snapshot_id", "branch_id", "page_phys", "tokens_filled", "num_pages")
    def __init__(self, snapshot_id, branch_id, page_phys, tokens_filled, num_pages):
        self.snapshot_id = snapshot_id
        self.branch_id = branch_id
        self.page_phys = page_phys           # tuple of PhysPage (refs held)
        self.tokens_filled = tokens_filled
        self.num_pages = num_pages


class ForkHandle:
    __slots__ = ("branch_id", "snapshot_id", "fork_latency_s", "pages_aliased")
    def __init__(self, branch_id, snapshot_id, fork_latency_s, pages_aliased):
        self.branch_id = branch_id
        self.snapshot_id = snapshot_id
        self.fork_latency_s = fork_latency_s
        self.pages_aliased = pages_aliased


class KVBranchManager:
    def __init__(self, device_id=0, page_size=None, max_pages_per_branch=4096,
                 scratch_pool_size=8):
        self.pool = VMMPool(device_id=device_id, page_size=page_size)
        self.page_size = self.pool.page_size
        self.max_pages_per_branch = max_pages_per_branch
        self.branches = {}
        self.snapshots = {}
        self._next_snap_id = 0
        # B8 / R2-D3: pre-reserve a pool of reusable 1-page scratch VA windows. _cow()
        # borrows one to stage the private copy instead of reserve+free'ing a fresh VA
        # range every time. This removes the scratch-VA bookkeeping that B5 measured as
        # ~47% of CoW latency (175.7us -> the reserve/free pair is amortized away).
        self.scratch_pool_size = scratch_pool_size
        self._scratch_free = []   # list of (va, size) one-page windows, unmapped & idle
        for _ in range(scratch_pool_size):
            va, size = self.pool.reserve_va(1)
            self._scratch_free.append((va, size))
        self.stat_scratch_exhausted = 0  # times the pool was empty and we fell back

    def _borrow_scratch(self):
        if self._scratch_free:
            return self._scratch_free.pop(), True
        # pool exhausted (more concurrent CoW than scratch_pool_size): fall back to a
        # fresh reserve (the old path) so correctness never depends on pool size.
        self.stat_scratch_exhausted += 1
        return self.pool.reserve_va(1), False

    def _return_scratch(self, slot, from_pool):
        if from_pool and len(self._scratch_free) < self.scratch_pool_size:
            self._scratch_free.append(slot)
        else:
            va, size = slot
            # not a pooled slot (fallback) -> genuinely release it
            self.pool.va_pool_enabled = False
            self.pool.free_va(va, size)
            self.pool.va_pool_enabled = True

    # ---- branch lifecycle ----
    def create_branch(self, branch_id, num_pages, headroom_pages=0):
        """Create a branch. Reserves VA for (num_pages + headroom_pages) slots so the
        branch can grow its tail dynamically via append_page() without re-reserving VA
        (P0-C). VA reservation costs NO HBM; only mapped pages commit physical memory."""
        assert branch_id not in self.branches
        capacity = num_pages + headroom_pages
        va_base, va_size = self.pool.reserve_va_range(max(capacity, 1))
        br = Branch(branch_id, va_base, va_size, num_pages, capacity=max(capacity, 1))
        self.branches[branch_id] = br
        return br

    def append_page(self, branch_id, fill_value=None):
        """P0-C: grow a branch's KV tail by one fresh private page (real agents grow KV
        as decode emits tokens). Maps the next unmapped slot inside the branch's reserved
        VA headroom. Returns the new page_index. Raises if headroom is exhausted.
        CoW semantics are preserved: appended pages are private (refcount 1), so a child
        forked before the append never sees them; a child forked AFTER snapshotting the
        grown branch aliases them."""
        br = self.branches[branch_id]
        idx = br.num_pages
        if idx >= br.capacity:
            raise RuntimeError(
                f"branch {branch_id} VA headroom exhausted (capacity={br.capacity}); "
                f"reserve more headroom_pages at create/fork time")
        pg = self.pool.create_phys_page()
        self.pool.map_page(br.va_of(idx), pg)
        br.page_phys[idx] = pg
        if fill_value is not None:
            self.pool.memset_page(br.va_of(idx), fill_value)
        br.num_pages += 1
        return idx

    def alloc_page(self, branch_id, page_index, fill_value=None):
        """Allocate a fresh private physical page and map it into the branch."""
        br = self.branches[branch_id]
        assert br.page_phys[page_index] is None, "page already mapped"
        pg = self.pool.create_phys_page()
        self.pool.map_page(br.va_of(page_index), pg)
        br.page_phys[page_index] = pg
        if fill_value is not None:
            self.pool.memset_page(br.va_of(page_index), fill_value)
        return pg

    def fill_prefix(self, branch_id, num_pages, fill_value=7):
        """Convenience: allocate+fill the first num_pages of a branch (a 'prefix')."""
        for i in range(num_pages):
            self.alloc_page(branch_id, i, fill_value=fill_value)
        self.branches[branch_id].tokens_filled = num_pages

    # ---- Snapshot ----
    def snapshot(self, branch_id):
        br = self.branches[branch_id]
        mapped = [pg for pg in br.page_phys if pg is not None]
        # hold a ref on each mapped page so it survives even if the branch overwrites
        for pg in mapped:
            self.pool.incref(pg)
        sid = self._next_snap_id; self._next_snap_id += 1
        snap = SnapshotHandle(sid, branch_id, tuple(mapped), br.tokens_filled, len(mapped))
        self.snapshots[sid] = snap
        return snap

    # ---- Fork ----
    def fork(self, snapshot, new_branch_id, headroom_pages=0):
        t0 = time.perf_counter()
        n = snapshot.num_pages
        br = self.create_branch(new_branch_id, max(n, 1), headroom_pages=headroom_pages)
        for i in range(n):
            pg = snapshot.page_phys[i]
            self.pool.map_page(br.va_of(i), pg)   # alias same physical handle
            self.pool.incref(pg)
            br.page_phys[i] = pg
        br.num_pages = n if n > 0 else 1
        br.tokens_filled = snapshot.tokens_filled
        br.parent = snapshot.branch_id
        self.pool.synchronize()
        dt = time.perf_counter() - t0
        return ForkHandle(new_branch_id, snapshot.snapshot_id, dt, n)

    def destroy_branch(self, branch_id):
        """Tear down a branch: unmap every mapped page (decref its physical handle, freeing
        HBM at refcount 0), then return the branch's VA reservation to the process-wide VA
        free-list (P0-2) so a later fork of the SAME size reuses it instead of issuing a
        fresh cuMemAddressReserve. This is what turns Metric 4's mapping-metadata ceiling
        into the true data ceiling."""
        br = self.branches.pop(branch_id)
        for i in range(br.capacity):
            pg = br.page_phys[i]
            if pg is not None:
                self.pool.unmap_page(br.va_of(i))
                self.pool.decref(pg)
                br.page_phys[i] = None
        self.pool.free_va(br.va_base, br.va_size, num_pages=br.capacity)

    # ---- CoW write (software page-fault-on-write) ----
    def write_page(self, branch_id, page_index, host_bytes=None, fill_value=None):
        """Write to a page. If the page is shared (refcount>1) perform CoW first.
        Returns True if a CoW copy happened."""
        br = self.branches[branch_id]
        pg = br.page_phys[page_index]
        assert pg is not None, "page not mapped"
        did_cow = False
        if pg.refcount > 1:
            did_cow = self._cow(br, page_index, pg)
            pg = br.page_phys[page_index]
        va = br.va_of(page_index)
        if host_bytes is not None:
            buf = (np.frombuffer(host_bytes, dtype=np.uint8))
            _ck(cuda.cuMemcpyHtoD(va, buf.ctypes.data, min(len(host_bytes), self.page_size)))
        elif fill_value is not None:
            self.pool.memset_page(va, fill_value)
        return did_cow

    def _cow(self, br, page_index, shared_pg):
        """Allocate private page, copy contents, remap one VA page, decref shared.
        B8 / R2-D3: stages the copy through a REUSABLE scratch VA window borrowed from a
        pre-reserved pool, eliminating the per-CoW cuMemAddressReserve+cuMemAddressFree
        pair that B5 measured as ~47% of CoW latency."""
        va = br.va_of(page_index)
        new_pg = self.pool.create_phys_page()
        new_pg.born_from = shared_pg.page_id
        # copy old contents into the new physical page via a reusable scratch VA window
        (tmp_va, tmp_size), from_pool = self._borrow_scratch()
        self.pool.map_page(tmp_va, new_pg)
        self.pool.copy_page(tmp_va, va)              # D2D copy of 2 MiB
        self.pool.unmap_page(tmp_va)                 # leave VA reserved & idle for reuse
        self._return_scratch((tmp_va, tmp_size), from_pool)
        # remap the branch's VA page to the private copy
        self.pool.unmap_page(va)
        self.pool.map_page(va, new_pg)
        br.page_phys[page_index] = new_pg
        self.pool.decref(shared_pg)
        self.pool.stat_cow_events += 1
        return True

    # ---- Divergence detector ----
    def diverged_pages(self, branch_a, branch_b):
        """Return list of page indices where the two branches map DIFFERENT physical
        handles (verified via the driver's retained allocation handle)."""
        a = self.branches[branch_a]; b = self.branches[branch_b]
        n = min(a.num_pages, b.num_pages)
        out = []
        for i in range(n):
            if a.page_phys[i] is None or b.page_phys[i] is None:
                continue
            ha = self.pool.retained_handle_at(a.va_of(i))
            hb = self.pool.retained_handle_at(b.va_of(i))
            if ha != hb:
                out.append(i)
        return out

    def shared_handle(self, branch_a, branch_b, page_index):
        a = self.branches[branch_a]; b = self.branches[branch_b]
        return (self.pool.retained_handle_at(a.va_of(page_index)) ==
                self.pool.retained_handle_at(b.va_of(page_index)))

    # ---- accounting ----
    def stats(self):
        p = self.pool
        return dict(phys_pages_created=p.stat_phys_pages_created,
                    bytes_created=p.stat_bytes_created,
                    bytes_copied=p.stat_bytes_copied,
                    cow_events=p.stat_cow_events,
                    map_ops=p.stat_map_ops,
                    live_phys_pages=len(p.pages),
                    page_size=self.page_size)
