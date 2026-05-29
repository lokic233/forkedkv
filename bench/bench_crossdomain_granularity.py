"""
Priority-3 cross-domain granularity demo (v0.1, partially simulated).

Branchable state spans 3 domains with very different natural granularities:
  - KV    : attention cache. Granularity = GPU VMM page = 2 MiB (REAL, on H100).
  - RNG   : sampler RNG state. Granularity = tens of bytes (seed+counter). SIMULATED
            in host memory (RNG state is tiny and lives on host/cheap to copy).
  - TOOL  : external tool-call log. Granularity = one log record (variable). SIMULATED
            in host memory.

Claim under test: a single uniform page size is wrong for forking agent state. KV wants
coarse 2 MiB GPU pages (CoW via MMU); RNG/TOOL want byte-granular CoW (cheap host copy).
We measure bytes copied per fork+divergence under (a) uniform 2 MiB granularity applied
to ALL domains vs (b) per-domain granularity.

HONEST: KV numbers are REAL GPU VMM measurements; RNG/TOOL are host-side simulation of
state sizes representative of a sampler + tool log. Labeled accordingly in output CSV.

Output: data/crossdomain_granularity.csv
"""
import sys, os, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kv_branch_manager import KVBranchManager
from explog import log

PAGE = 2 * 1024 * 1024
FANOUT = 16
# representative per-fork divergent state sizes
KV_DIVERGENT_PAGES = 3                # 3 x 2MiB of KV actually rewritten per branch
RNG_STATE_BYTES = 32                  # philox seed+counter+offset
TOOL_LOG_BYTES = 4096                 # one appended tool record per branch
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "crossdomain_granularity.csv")


def real_kv_cow_bytes():
    """REAL: measure GPU bytes copied when FANOUT branches each rewrite KV_DIVERGENT_PAGES."""
    m = KVBranchManager(device_id=0)
    prefix = 32
    m.create_branch("p", prefix); m.fill_prefix("p", prefix, 7)
    snap = m.snapshot("p")
    bk = m.pool.stat_bytes_copied; bc = m.pool.stat_bytes_created
    for i in range(FANOUT):
        m.fork(snap, "c%d" % i)
        for pidx in range(KV_DIVERGENT_PAGES):
            m.write_page("c%d" % i, pidx, fill_value=i % 251 + 1)
    real = (m.pool.stat_bytes_copied - bk) + (m.pool.stat_bytes_created - bc)
    m.pool.destroy()
    return real


def main():
    rows = []
    kv_real = real_kv_cow_bytes()   # GPU-measured

    # Uniform 2MiB granularity: every domain's divergence rounds UP to a 2MiB page.
    uniform = {
        "KV":   kv_real,                                   # already page-granular (real)
        "RNG":  FANOUT * PAGE,                             # 32 B rounded to 2 MiB
        "TOOL": FANOUT * PAGE,                             # 4 KB rounded to 2 MiB
    }
    # Per-domain granularity:
    perdomain = {
        "KV":   kv_real,                                   # 2 MiB pages (correct for KV)
        "RNG":  FANOUT * RNG_STATE_BYTES,                  # byte-granular
        "TOOL": FANOUT * TOOL_LOG_BYTES,                   # record-granular
    }
    for dom in ["KV", "RNG", "TOOL"]:
        src = "GPU-measured" if dom == "KV" else "host-simulated"
        rows.append((dom, "uniform_2MiB", uniform[dom], src))
        rows.append((dom, "per_domain", perdomain[dom], src))
        print(f"{dom:5s} uniform={uniform[dom]/2**20:8.2f}MiB  per-domain={perdomain[dom]/2**20:10.5f}MiB  ({src})")

    tot_u = sum(uniform.values()); tot_p = sum(perdomain.values())
    red = 100*(1 - tot_p/tot_u)
    print(f"\nTOTAL bytes copied/fork-batch: uniform={tot_u/2**20:.2f}MiB  "
          f"per-domain={tot_p/2**20:.2f}MiB  reduction={red:.1f}%")
    with open(OUT,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["domain","scheme","bytes_copied","source"]); w.writerows(rows)
    log("crossdomain_granularity",
        dict(fanout=FANOUT, kv_divergent_pages=KV_DIVERGENT_PAGES,
             rng_state_bytes=RNG_STATE_BYTES, tool_log_bytes=TOOL_LOG_BYTES),
        dict(uniform_total_bytes=tot_u, perdomain_total_bytes=tot_p,
             reduction_pct=red, kv_gpu_measured_bytes=kv_real))
    print("wrote", OUT)

if __name__=="__main__":
    main()
