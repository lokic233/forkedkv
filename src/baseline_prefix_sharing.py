"""
baseline_prefix_sharing.py — vLLM-style software prefix sharing (block table + refcounts).

This is the STRONG software baseline that committee reviewer Gemini correctly identified:
vLLM Automatic Prefix Caching (APC) and SGLang RadixAttention already provide zero-copy
prefix sharing in software, with FINER granularity than our 2 MiB CUDA-VMM page.

Mechanism (mirroring vLLM v0.6 PagedAttention / APC):
  - ONE pre-allocated torch tensor pool of physical "blocks" of fixed size in tokens
    (vLLM default: 16 tokens/block; we parameterise this).
  - Each sequence (= our "branch") owns a block_table: list[block_id], one entry per
    logical block. To attend, the kernel gathers KV via this indirection.
  - SHARED prefix = two block_tables point to the same block_id (refcount > 1).
    Fork = copy the block_table (a Python list of ints) and refcount++ each entry.
    Cost per fork = O(prefix_blocks) cheap pointer ops; NO HBM allocation.
  - WRITE on a shared block = COPY-ON-WRITE at BLOCK granularity (16 tokens, not 2 MiB):
    allocate a free block from the pool, memcpy the 16-token block, repoint the writer's
    block_table entry, decref the shared block. This is finer than our VMM-page CoW.

What this baseline INTENTIONALLY does not model:
  - Custom PagedAttention / FlashAttention-with-block-table kernels: those are the
    mandatory cost vLLM pays to use this allocator. Standard FlashAttention does NOT
    accept a block table; it requires contiguous K/V tensors. We measure that
    asymmetry analytically (WRITEUP §"Comparison vs Software Prefix Sharing").
  - vLLM hash-based prefix detection: we hand the parent block_table to fork() so the
    sharing is exact, which is the strictly-best case for the software baseline.

Capacity model:
  Pool = N_blocks_total physical blocks of `block_tokens` each. With pool_size_gib=12 GiB,
  block_tokens=16, layers=64, heads=8, head_dim=128, dtype=bf16:
    bytes_per_token  = 2 (K/V) * layers * heads * head_dim * 2 = 64*8*128*4 = 262,144 B
    bytes_per_block  = bytes_per_token * block_tokens = 4,194,304 B  (= 4 MiB)
    blocks_in_12GiB  = 12 GiB / 4 MiB = 3,072
  This is identical to our ForkedKV 12 GiB / 2 MiB = 6,144 page pool, scaled for KV bytes.
  The fair comparison is: at a fixed prefix-bytes budget, how many branches can each
  mechanism alias? With N branches sharing a P-byte prefix and zero divergence:
    SOFTWARE: N is bounded only by RAM (block_table is ~8 B/block); pool stays constant.
    HARDWARE (us): N is bounded by the per-context driver mapping ceiling K ~= 520K (Lab 1).

So on PURE capacity, software wins. The hardware delta is NOT capacity. It is:
  (1) contiguous virtual address per branch  -> unmodified FlashAttention works OOB
  (2) MMU-level page aliasing                -> no per-token block-table lookup in hot path

We measure (a) fork latency, (b) bytes copied per diverging block (CoW granularity),
(c) capacity vs ForkedKV at matched prefix size. Result intentionally favours software
on (a) and (b); the headline of the paper is therefore the asymmetry on (1)+(2),
not capacity.
"""

import time
import numpy as np
from collections import defaultdict


class SoftwareBlockPool:
    """Refcounted block pool. Each block is a fixed-size byte buffer; we use numpy
    on-host because the SOFTWARE baseline's cost model is dominated by pointer ops,
    not memory bandwidth, and we want CoW byte counts to be GPU-arch-independent.
    For wall-clock fork latency we measure pure block-table operations (the way
    vLLM's scheduler does it) so the result is bandwidth-free."""

    def __init__(self, n_blocks, block_bytes):
        self.n_blocks = n_blocks
        self.block_bytes = block_bytes
        self.free_blocks = list(range(n_blocks))
        self.refcount = [0] * n_blocks
        # Stats
        self.stat_blocks_allocated = 0
        self.stat_bytes_copied = 0
        self.stat_cow_events = 0

    def alloc(self):
        if not self.free_blocks:
            raise MemoryError("software block pool exhausted")
        b = self.free_blocks.pop()
        self.refcount[b] = 1
        self.stat_blocks_allocated += 1
        return b

    def incref(self, b):
        assert self.refcount[b] > 0
        self.refcount[b] += 1

    def decref(self, b):
        assert self.refcount[b] > 0
        self.refcount[b] -= 1
        if self.refcount[b] == 0:
            self.free_blocks.append(b)

    @property
    def in_use(self):
        return sum(1 for c in self.refcount if c > 0)


class SoftwarePrefixSharingManager:
    """vLLM-APC-equivalent block-table allocator.
    Each branch is identified by branch_id; we keep a list[block_id] block_table per branch.
    Fork is a list copy + refcount++ per block.
    Write to a shared block triggers CoW: alloc a fresh block, simulate a 16-token memcpy,
    repoint the writer's table entry, decref the shared block.
    """

    def __init__(self, n_blocks, block_bytes):
        self.pool = SoftwareBlockPool(n_blocks, block_bytes)
        self.block_bytes = block_bytes
        self.branches = {}   # branch_id -> list[block_id]

    def create_filled_branch(self, branch_id, num_blocks):
        bt = []
        for _ in range(num_blocks):
            bt.append(self.pool.alloc())
        self.branches[branch_id] = bt
        return bt

    def fork(self, src_branch_id, new_branch_id):
        """vLLM-style fork: copy the block table, refcount++ each shared block.
        Zero HBM bytes copied; cost is ~num_blocks * (list append + refcount inc)."""
        t0 = time.perf_counter()
        src = self.branches[src_branch_id]
        new_bt = list(src)                         # shallow Python list copy
        for b in new_bt:
            self.pool.incref(b)
        self.branches[new_branch_id] = new_bt
        dt = time.perf_counter() - t0
        return dt

    def write_block(self, branch_id, logical_index, tokens_in_block=None):
        """Write to a logical block. If shared (refcount>1), CoW at BLOCK granularity:
        allocate a fresh physical block, simulate 16-token KV copy, decref the shared
        block. tokens_in_block defaults to "full block" (=block_bytes copied)."""
        bt = self.branches[branch_id]
        old = bt[logical_index]
        if self.pool.refcount[old] > 1:
            new = self.pool.alloc()
            # bytes copied = the full block (vLLM has to copy the whole block; the
            # 16-token granularity is what gives finer-grained sharing, NOT
            # finer-grained copy-on-write within a block)
            self.pool.stat_bytes_copied += self.block_bytes
            self.pool.stat_cow_events += 1
            self.pool.decref(old)
            bt[logical_index] = new

    def append_block(self, branch_id):
        """Decode appends a fresh private block (refcount=1). Mirrors our append_page()."""
        b = self.pool.alloc()
        self.branches[branch_id].append(b)
        return b

    def stats(self):
        return dict(
            blocks_allocated=self.pool.stat_blocks_allocated,
            bytes_copied=self.pool.stat_bytes_copied,
            cow_events=self.pool.stat_cow_events,
            blocks_in_use=self.pool.in_use,
            n_blocks_total=self.pool.n_blocks,
            block_bytes=self.block_bytes,
            n_branches=len(self.branches),
        )


# vLLM-style KV-byte sizing (matches Llama-3-8B-ish: 32 layers, 8 KV-heads, 128 head_dim, bf16)
# Kept as a default; experiments override.
def kv_bytes_per_token(n_layers=32, n_kv_heads=8, head_dim=128, dtype_bytes=2):
    return 2 * n_layers * n_kv_heads * head_dim * dtype_bytes  # K + V

# Llama-3-8B-ish: 32 layers, 8 KV-heads, 128 head_dim, bf16 -> 128 KiB/token
DEFAULT_KV_BYTES_PER_TOKEN = 131072
# vLLM default block size = 16 tokens
DEFAULT_BLOCK_TOKENS = 16
DEFAULT_BLOCK_BYTES = DEFAULT_KV_BYTES_PER_TOKEN * DEFAULT_BLOCK_TOKENS  # 2 MiB at the defaults above
