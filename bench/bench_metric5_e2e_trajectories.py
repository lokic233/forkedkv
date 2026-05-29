"""
Metric 5: End-to-end on real SWE-Bench-Verified trajectories (representative subset).

Uses real SWE-bench_Verified instances (data/swe_selected_instances.csv) sampled across
the problem-statement size distribution. For each instance we model a realistic agentic
run: a long shared PREFIX (system prompt + retrieved repo context + problem statement,
sized from real instance text) followed by FANOUT candidate-fix BRANCHES that each
diverge in their tail (write DIVERGENCE_FRAC of prefix pages). We compare CoW-fork vs
full-clone on wall time, peak HBM footprint, and KV bytes physically written.

All sizing constants are in bench/m5_config.json.

HONEST CAVEAT: NOT a full LLM-agent token-generation loop. KV pages are sized from real
instance text but filled synthetically; per-token attention compute is Metric 3 (run
separately). This isolates the MEMORY mechanism end-to-end over real workload shapes.
See LIMITATIONS.md.

Output: data/metric5_e2e.csv
"""
import sys, os, csv, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pandas as pd
from kv_branch_manager import KVBranchManager
from baseline_fullclone import FullCloneManager
from explog import log

CFG = json.load(open(os.path.join(os.path.dirname(__file__), "m5_config.json")))
KV_BYTES_PER_TOKEN = CFG["kv_bytes_per_token"]
CHARS_PER_TOKEN = CFG["chars_per_token"]
SYS_PROMPT_TOKENS = CFG["sys_prompt_tokens"]
REPO_CONTEXT_TOKENS = CFG["repo_context_tokens"]
FANOUT = CFG["fanout"]
DIVERGENCE_FRAC = CFG["divergence_frac"]
PAGE_SIZE = CFG["page_size"]
MIB = 1024 * 1024
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "metric5_e2e.csv")
SEL = os.path.join(os.path.dirname(__file__), "..", "data", "swe_selected_instances.csv")


def prefix_pages_for(ps_chars):
    prob_tokens = ps_chars / CHARS_PER_TOKEN
    total_tokens = SYS_PROMPT_TOKENS + REPO_CONTEXT_TOKENS + prob_tokens
    prefix_bytes = total_tokens * KV_BYTES_PER_TOKEN
    return max(1, int((prefix_bytes + PAGE_SIZE - 1) // PAGE_SIZE)), int(total_tokens)


def run_cow(npages):
    m = KVBranchManager(device_id=0)
    m.create_branch("p", npages); m.fill_prefix("p", npages, 7)
    snap = m.snapshot("p")
    bc, bk = m.pool.stat_bytes_created, m.pool.stat_bytes_copied
    t0 = time.perf_counter()
    nwrite = int(round(DIVERGENCE_FRAC * npages))
    for i in range(FANOUT):
        m.fork(snap, "c%d" % i)
        for p in range(nwrite):
            m.write_page("c%d" % i, p, fill_value=i % 251 + 1)
    m.pool.synchronize()
    dt = time.perf_counter() - t0
    written = (m.pool.stat_bytes_created - bc) + (m.pool.stat_bytes_copied - bk)
    live = len(m.pool.pages) * PAGE_SIZE
    m.pool.destroy()
    return dt, written, live


def run_clone(npages):
    m = FullCloneManager(device_id=0)
    m.create_filled_branch("p", npages, 7)
    bc, bk = m.pool.stat_bytes_created, m.pool.stat_bytes_copied
    t0 = time.perf_counter()
    for i in range(FANOUT):
        m.clone("p", "c%d" % i)
    m.pool.synchronize()
    dt = time.perf_counter() - t0
    written = (m.pool.stat_bytes_created - bc) + (m.pool.stat_bytes_copied - bk)
    live = len(m.pool.pages) * PAGE_SIZE
    m.destroy()
    return dt, written, live


def main():
    sel = pd.read_csv(SEL)
    rows = []
    print("KV/token=%dKiB fanout=%d divergence=%d%%" %
          (KV_BYTES_PER_TOKEN // 1024, FANOUT, int(DIVERGENCE_FRAC * 100)))
    print("%-34s %7s %6s  %8s %9s %5s %8s %9s %7s %8s" % (
        "instance", "pfx_pgs", "tok", "cow_ms", "clone_ms", "spd",
        "cow_MiB", "clone_MiB", "wr_red%", "mem_red%"))
    for _, r in sel.iterrows():
        npages, toks = prefix_pages_for(r["ps_chars"])
        cdt, cwr, cliv = run_cow(npages)
        bdt, bwr, bliv = run_clone(npages)
        spd = bdt / cdt if cdt else float("nan")
        wr_red = 100 * (1 - cwr / bwr) if bwr else 0
        mem_red = 100 * (1 - cliv / bliv) if bliv else 0
        print("%-34s %7d %6d  %8.2f %9.2f %5.1f %8.1f %9.1f %7.1f %8.1f" % (
            r["instance_id"], npages, toks, cdt * 1e3, bdt * 1e3, spd,
            cwr / MIB, bwr / MIB, wr_red, mem_red))
        rows.append((r["instance_id"], r["repo"], r["ps_chars"], npages, toks,
                     cdt, bdt, spd, cwr, bwr, cliv, bliv, wr_red, mem_red))
        log("metric5_e2e",
            dict(instance=r["instance_id"], repo=r["repo"], ps_chars=int(r["ps_chars"]),
                 prefix_pages=npages, prefix_tokens=toks, fanout=FANOUT,
                 divergence_frac=DIVERGENCE_FRAC),
            dict(cow_s=cdt, clone_s=bdt, speedup=spd, cow_bytes_written=cwr,
                 clone_bytes_written=bwr, cow_live_bytes=cliv, clone_live_bytes=bliv,
                 bytes_written_reduction_pct=wr_red, mem_reduction_pct=mem_red))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "repo", "ps_chars", "prefix_pages", "prefix_tokens",
                    "cow_s", "clone_s", "speedup", "cow_bytes_written", "clone_bytes_written",
                    "cow_live_bytes", "clone_live_bytes", "bytes_written_reduction_pct",
                    "mem_reduction_pct"])
        w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
