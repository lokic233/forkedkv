"""
bench_software_baseline.py — head-to-head: ForkedKV (CUDA-VMM, 2 MiB pages)
                              vs SoftwarePrefixSharing (vLLM-APC-style, 16-token blocks)

Addresses Gemini's RED Attack 1 ("strawman baseline"). The honest comparison is NOT
against a naive full-clone; it is against vLLM's existing block-table prefix sharing.

Three measurements (all on H100, single process):
  M1. fork_latency vs prefix length (pages / blocks).
      Software fork = list copy + refcount++ per block. Hardware fork = cuMemMap +
      cuMemSetAccess per page. Software is expected to win on raw fork latency.

  M2. CoW granularity = bytes copied per single-token write into a shared prefix.
      Software CoW unit = block_bytes (default 2 MiB at our token-equivalent sizing,
      or 256 KiB if we use vLLM's exact 16-token block at Llama-3-8B sizing).
      Hardware (us) CoW unit = page_bytes = 2 MiB.
      The interesting axis: at smaller block sizes (16-token blocks under realistic
      sizing) software is FINER-grained than us.

  M3. capacity / max-branches at a fixed prefix size.
      Software: bounded only by pool size; refcounts cost ~8 B/branch. Should easily
      reach ~10^4 branches in CPU memory at any prefix. Hardware: bounded by the
      driver mapping ceiling K~520K (Lab 1).

Output: data/baseline_compare_*.csv
"""
import sys, os, time, csv, statistics, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from baseline_prefix_sharing import (
    SoftwarePrefixSharingManager, DEFAULT_BLOCK_BYTES, DEFAULT_KV_BYTES_PER_TOKEN, DEFAULT_BLOCK_TOKENS
)
from kv_branch_manager import KVBranchManager

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# At our default sizing (Llama-3-8B-ish, 32L, 8KV, 128hd, bf16), 2 MiB CUDA-VMM page
# == 16 tokens of KV. So the comparison is exactly fair:
#   Hardware page = 2 MiB = 16 tokens of KV (one vLLM block at this sizing)
#   Software block = 16 tokens of KV = 2 MiB (one vLLM block at this sizing)
# With these defaults the SHARING granularity is identical; the real differences are
# elsewhere (kernel compatibility, fork-latency mechanics).
HW_PAGE_BYTES = 2 * 1024 * 1024
assert DEFAULT_BLOCK_BYTES == HW_PAGE_BYTES, (
    f"block_bytes ({DEFAULT_BLOCK_BYTES}) must match HW page size for an apples-to-apples test"
)


# ---------- M1: fork latency ----------
def m1_fork_latency():
    PREFIX_BLOCKS = [1, 2, 4, 8, 16, 32, 64, 128]
    REPS = 16
    out = os.path.join(DATA_DIR, "baseline_compare_m1_fork_latency.csv")
    rows = []

    # Software path
    pool_blocks = max(PREFIX_BLOCKS) + REPS * max(PREFIX_BLOCKS) + 16
    for n in PREFIX_BLOCKS:
        sw = SoftwarePrefixSharingManager(n_blocks=pool_blocks, block_bytes=DEFAULT_BLOCK_BYTES)
        sw.create_filled_branch("p", n)
        lats = []
        for r in range(REPS):
            dt = sw.fork("p", f"c{r}")
            lats.append(dt * 1e6)
        rows.append(("software", n, statistics.mean(lats), statistics.stdev(lats) if len(lats)>1 else 0.0,
                     min(lats), max(lats)))
        del sw; gc.collect()

    # Hardware (ForkedKV) path
    for n in PREFIX_BLOCKS:
        m = KVBranchManager(device_id=0)
        m.create_branch("p", n); m.fill_prefix("p", n, 7)
        snap = m.snapshot("p")
        lats = []
        for r in range(REPS):
            h = m.fork(snap, f"c{r}")
            lats.append(h.fork_latency_s * 1e6)
        rows.append(("hardware_vmm", n, statistics.mean(lats), statistics.stdev(lats) if len(lats)>1 else 0.0,
                     min(lats), max(lats)))
        # cleanup
        for r in range(REPS):
            try: m.destroy_branch(f"c{r}")
            except Exception: pass
        try: m.destroy_branch("p")
        except Exception: pass
        try: m.pool.destroy()
        except Exception: pass
        del m; gc.collect()

    with open(out, "w") as f:
        w = csv.writer(f)
        w.writerow(["method","prefix_blocks","mean_us","stddev_us","min_us","max_us"])
        w.writerows(rows)
    print("[M1]", out)
    for r in rows:
        print(f"  {r[0]:14s} prefix={r[1]:4d} mean={r[2]:8.2f}us stddev={r[3]:7.2f}us min={r[4]:8.2f} max={r[5]:8.2f}")
    return rows


# ---------- M2: CoW granularity ----------
def m2_cow_granularity():
    """Single-token write into a shared 32-block prefix. Bytes copied per write event."""
    out = os.path.join(DATA_DIR, "baseline_compare_m2_cow_granularity.csv")
    rows = []

    # Software CoW: copies one block (= block_bytes)
    sw = SoftwarePrefixSharingManager(n_blocks=256, block_bytes=DEFAULT_BLOCK_BYTES)
    sw.create_filled_branch("p", 32)
    sw.fork("p", "c")
    sw.write_block("c", 0)   # write to first shared block
    s_stats = sw.stats()
    rows.append(("software", "1-token write -> 1-block CoW",
                 s_stats["bytes_copied"], DEFAULT_BLOCK_BYTES, DEFAULT_BLOCK_TOKENS))

    # Hardware (us): copies one PAGE = 2 MiB
    m = KVBranchManager(device_id=0)
    m.create_branch("p", 32); m.fill_prefix("p", 32, 7)
    snap = m.snapshot("p")
    m.fork(snap, "c")
    bytes_before = m.pool.stat_bytes_copied
    m.write_page("c", 0, fill_value=0xAB)
    bytes_after = m.pool.stat_bytes_copied
    hw_bytes = bytes_after - bytes_before
    rows.append(("hardware_vmm", "1-token write -> 1-page CoW",
                 hw_bytes, HW_PAGE_BYTES, DEFAULT_BLOCK_TOKENS))

    with open(out, "w") as f:
        w = csv.writer(f)
        w.writerow(["method","scenario","bytes_copied","cow_unit_bytes","tokens_per_unit"])
        w.writerows(rows)
    print("[M2]", out)
    for r in rows:
        print(f"  {r[0]:14s} {r[1]:32s} bytes={r[2]:10d} unit={r[3]:10d} tokens/unit={r[4]}")
    return rows


# ---------- M3: capacity / max branches ----------
def m3_capacity():
    """How many branches at prefix=32 blocks before each system fails?

    Software: pure CPU bookkeeping. We simulate up to 100,000 branches.
    Hardware: gated by driver mapping ceiling. We use the previously-measured
    capacity model rather than re-running the destructive sweep.
    """
    out = os.path.join(DATA_DIR, "baseline_compare_m3_capacity.csv")
    rows = []
    PREFIX_BLOCKS = 32   # 64 MiB / 512 tokens of KV at default sizing

    # Software: fork until the block POOL refcount overhead is the only constraint.
    # vLLM in production sizes its pool to all available HBM; with prefix sharing
    # the pool grows by O(1) per branch (just the refcount table on host).
    # We measure: (a) wall-clock per fork at scale, (b) host RAM per branch.
    sw = SoftwarePrefixSharingManager(n_blocks=PREFIX_BLOCKS + 16, block_bytes=DEFAULT_BLOCK_BYTES)
    sw.create_filled_branch("p", PREFIX_BLOCKS)
    targets = [10, 100, 1000, 10000, 100000]
    import resource
    rss_pre = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    n_done = 0
    t_start = time.perf_counter()
    for k in range(1, max(targets) + 1):
        sw.fork("p", f"c{k}")
        n_done = k
        if k in targets:
            t_now = time.perf_counter()
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            rows.append(("software", PREFIX_BLOCKS, k,
                         (t_now - t_start), (rss - rss_pre) * 1024,
                         "host RAM (block_table refcounts)"))

    # Hardware: ceiling model from Lab 1 (driver mapping budget K ≈ 520,000
    # entries per context; max_branches ≈ K / prefix_pages).
    # At prefix=32 pages: ~ 520000/32 = ~16,250 branches before driver OOM
    # on cuMemSetAccess.
    K = 520000
    hw_cap = K // PREFIX_BLOCKS
    rows.append(("hardware_vmm", PREFIX_BLOCKS, hw_cap, None, None,
                 "driver mapping ceiling (Lab 1: K~520K / prefix_pages)"))

    with open(out, "w") as f:
        w = csv.writer(f)
        w.writerow(["method","prefix_blocks","branches","wall_s","extra_bytes_rss","note"])
        w.writerows(rows)
    print("[M3]", out)
    for r in rows:
        print(f"  {r[0]:14s} prefix={r[1]:3d} branches={r[2]:7d} t={r[3]} rss_delta={r[4]} {r[5]}")
    return rows


def main():
    print("="*72)
    print("Software Prefix Sharing (vLLM-APC equivalent) vs ForkedKV (CUDA-VMM)")
    print("Block/page size:", DEFAULT_BLOCK_BYTES // 1024 // 1024, "MiB")
    print("="*72)
    m1_fork_latency()
    print()
    m2_cow_granularity()
    print()
    m3_capacity()


if __name__ == "__main__":
    main()
