"""
baseline_fullclone.py — Full-clone KV baseline (mimics naive vLLM branch cloning).

To branch an agent, the naive approach allocates a fresh KV region and copies every
KV byte of the prefix (no page sharing). This is what you get if you take a vLLM
sequence's KV blocks and deep-copy them into a new sequence. We implement it with the
SAME VMM pool so the comparison isolates the CoW mechanism (not the allocator).

A full-clone fork:
  - reserves a new VA range
  - creates a fresh physical page per prefix page (cuMemCreate)
  - copies prefix bytes D2D into each
So bytes_written_per_fork = prefix_pages * page_size, and HBM footprint grows linearly
with branch fanout.
"""
import time
from vmm_pool import VMMPool


class FullCloneManager:
    def __init__(self, device_id=0, page_size=None):
        self.pool = VMMPool(device_id=device_id, page_size=page_size)
        self.page_size = self.pool.page_size
        self.branches = {}   # branch_id -> dict(va_base, va_size, pages[list of PhysPage], num_pages)

    def create_filled_branch(self, branch_id, num_pages, fill_value=7):
        va, size = self.pool.reserve_va(num_pages)
        pages = []
        for i in range(num_pages):
            pg = self.pool.create_phys_page()
            self.pool.map_page(va + i*self.page_size, pg)
            self.pool.memset_page(va + i*self.page_size, fill_value)
            pages.append(pg)
        self.branches[branch_id] = dict(va=va, size=size, pages=pages, num_pages=num_pages)
        return self.branches[branch_id]

    def clone(self, src_branch_id, new_branch_id):
        """Full deep copy of the source branch's KV pages."""
        t0 = time.perf_counter()
        src = self.branches[src_branch_id]
        n = src["num_pages"]
        va, size = self.pool.reserve_va(n)
        pages = []
        for i in range(n):
            pg = self.pool.create_phys_page()                 # fresh physical page
            dst_va = va + i*self.page_size
            self.pool.map_page(dst_va, pg)
            self.pool.copy_page(dst_va, src["va"] + i*self.page_size)  # copy 2 MiB
            pages.append(pg)
        self.branches[new_branch_id] = dict(va=va, size=size, pages=pages, num_pages=n)
        self.pool.synchronize()
        dt = time.perf_counter() - t0
        return dt

    def stats(self):
        p = self.pool
        return dict(phys_pages_created=p.stat_phys_pages_created,
                    bytes_created=p.stat_bytes_created,
                    bytes_copied=p.stat_bytes_copied,
                    live_phys_pages=len(p.pages),
                    page_size=self.page_size)

    def destroy(self):
        """Unmap every VA page, release every physical handle, free VA ranges."""
        for b in self.branches.values():
            for i in range(b["num_pages"]):
                try: self.pool.unmap_page(b["va"] + i*self.page_size)
                except Exception: pass
            try: self.pool.free_va(b["va"], b["size"])
            except Exception: pass
        self.branches.clear()
        self.pool.destroy()
