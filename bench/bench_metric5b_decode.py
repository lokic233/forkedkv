"""
Metric 5b (P0-A): End-to-end single-layer autoregressive DECODE over CoW-backed KV.

This is the real-token-generation companion to Metric 5. It runs an ACTUAL
autoregressive decode loop using ONE real transformer layer (layer 0 of
Qwen2.5-7B-Instruct: 28 q-heads / 4 kv-heads GQA, head_dim 128, RoPE, RMSNorm; see
src/decode_layer.py and prototype_status R1-D1/D2) with the per-branch KV cache
physically backed by the VMM copy-on-write pages.

Workload (models N agent branches exploring from a shared context):
  1. Build a shared PREFIX of `prefix_tokens` by running real layer-0 forward over a
     prompt; its K/V live in VMM pages (one "parent" branch).
  2. Snapshot the parent. Two regimes:
       - cow:   fork N children from the snapshot (alias prefix KV, zero copy), then
                each child decodes `decode_tokens` NEW tokens, appending K/V to its own
                CoW-grown tail (writing the shared prefix pages triggers CoW only if a
                child overwrites them; pure append does not).
       - clone: each child gets a FULL deep copy of the prefix K/V (FullClone), then
                decodes the same `decode_tokens`.
  3. Measure, for each regime: total tokens/sec, peak live HBM (GPU pages * 2 MiB),
     total KV bytes physically copied across all branches.

HONEST CAVEATS (see LIMITATIONS.md):
  - ONE layer, not 28. Proves CoW pages support REAL attention compute with REAL
    weights; does NOT model full-model throughput or generation quality.
  - Greedy single-layer decode produces degenerate token sequences (no LM signal from
    one layer). We report SYSTEMS quantities (tok/s, HBM, bytes copied), NOT text.
  - tok/s here is single-layer; a full 28-layer model is ~28x more attention+MLP work.

Output: data/metric5b_decode.csv
  columns: regime, n_branches, prefix_tokens, decode_tokens, total_tokens,
           wall_s, tokens_per_s, peak_live_mib, kv_bytes_copied, peak_live_per_branch_mib
"""
import sys, os, csv, time, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from kv_branch_manager import KVBranchManager
from baseline_fullclone import FullCloneManager
from explog import log

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "metric5b_decode.csv")
MIB = 1024 * 1024


def _init_torch():
    torch.cuda.init(); torch.cuda.set_device(0)
    _ = torch.zeros(8, device="cuda"); torch.cuda.synchronize()


def build_prefix_kv(L, mgr, kid, vid, prefix_tokens, headroom):
    """Run real layer-0 forward over `prefix_tokens` tokens, storing K/V in VMM pages.
    Returns a BranchKV for the prefix."""
    from decode_layer import BranchKV
    mgr.create_branch(kid, 1, headroom_pages=headroom)
    mgr.create_branch(vid, 1, headroom_pages=headroom)
    bkv = BranchKV(mgr, L.n_kv, L.hd, kid, vid)
    tok = 1234
    for pos in range(prefix_tokens):
        h, q, k, v = L.project(tok, pos)
        bkv.append_token(k, v)
        logits = L.attend_and_logits(h, q, bkv.k_view(), bkv.v_view())
        tok = int(logits.argmax())
    return bkv


def decode_into(L, bkv, start_tok, n_tokens):
    """Decode n_tokens new tokens into bkv (appending K/V). Returns last token."""
    tok = start_tok
    base_pos = bkv.seq
    for i in range(n_tokens):
        h, q, k, v = L.project(tok, base_pos + i)
        bkv.append_token(k, v)
        logits = L.attend_and_logits(h, q, bkv.k_view(), bkv.v_view())
        tok = int(logits.argmax())
    return tok


def decode_seq(L, bkv, start_tok, n_tokens):
    """Like decode_into but also returns the produced token list (for correctness check)."""
    tok = start_tok; base = bkv.seq; out = []
    for i in range(n_tokens):
        h, q, k, v = L.project(tok, base + i)
        bkv.append_token(k, v)
        logits = L.attend_and_logits(h, q, bkv.k_view(), bkv.v_view())
        tok = int(logits.argmax()); out.append(tok)
    return out, tok


def run_cow(L, prefix_tokens, decode_tokens, n_branches):
    """Fork N children from a shared prefix snapshot, decode each. CoW-backed KV."""
    from decode_layer import BranchKV
    mgr = KVBranchManager(device_id=0)
    # headroom must cover prefix + decode pages
    bytes_per_tok = L.n_kv * L.hd * 2
    toks_per_page = mgr.page_size // bytes_per_tok
    total_pages = (prefix_tokens + decode_tokens) // toks_per_page + 2
    pbkv = build_prefix_kv(L, mgr, "pfxK", "pfxV", prefix_tokens, headroom=total_pages)
    torch.cuda.synchronize()
    snapK = mgr.snapshot("pfxK"); snapV = mgr.snapshot("pfxV")
    bytes_copied_0 = mgr.pool.stat_bytes_copied
    peak_pages = len(mgr.pool.pages)
    t0 = time.perf_counter()
    last = 4321; first_branch_tokens = []
    for b in range(n_branches):
        kid, vid = f"cK{b}", f"cV{b}"
        mgr.fork(snapK, kid, headroom_pages=total_pages)
        mgr.fork(snapV, vid, headroom_pages=total_pages)
        cbkv = BranchKV(mgr, L.n_kv, L.hd, kid, vid, reset=False)  # keep forked prefix pages
        cbkv.seq = prefix_tokens   # inherits prefix length
        toks, last = decode_seq(L, cbkv, last, decode_tokens)
        if b == 0: first_branch_tokens = toks
        peak_pages = max(peak_pages, len(mgr.pool.pages))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    bytes_copied = mgr.pool.stat_bytes_copied - bytes_copied_0
    peak_mib = peak_pages * mgr.page_size / MIB
    mgr.pool.destroy()
    return dt, bytes_copied, peak_mib, first_branch_tokens


def run_clone(L, prefix_tokens, decode_tokens, n_branches):
    """Each child gets a FULL deep copy of the prefix KV, then decodes. Full-clone."""
    from decode_layer import BranchKV
    mgr = KVBranchManager(device_id=0)
    bytes_per_tok = L.n_kv * L.hd * 2
    toks_per_page = mgr.page_size // bytes_per_tok
    total_pages = (prefix_tokens + decode_tokens) // toks_per_page + 2
    pbkv = build_prefix_kv(L, mgr, "pfxK", "pfxV", prefix_tokens, headroom=total_pages)
    torch.cuda.synchronize()
    prefix_pages = mgr.branches["pfxK"].num_pages
    bytes_copied_0 = mgr.pool.stat_bytes_copied
    peak_pages = len(mgr.pool.pages)
    t0 = time.perf_counter()
    last = 4321; first_branch_tokens = []
    for b in range(n_branches):
        kid, vid = f"cK{b}", f"cV{b}"
        # deep copy: create fresh pages and D2D-copy every prefix page (full clone)
        for src_bid, dst_bid in (("pfxK", kid), ("pfxV", vid)):
            mgr.create_branch(dst_bid, 1, headroom_pages=total_pages)
            mgr.branches[dst_bid].num_pages = 0
            src = mgr.branches[src_bid]
            for pi in range(src.num_pages):
                idx = mgr.append_page(dst_bid)            # fresh private page (cuMemCreate)
                mgr.pool.copy_page(mgr.branches[dst_bid].va_of(idx), src.va_of(pi))  # D2D
        cbkv = BranchKV(mgr, L.n_kv, L.hd, kid, vid, reset=False)
        cbkv.seq = prefix_tokens
        toks, last = decode_seq(L, cbkv, last, decode_tokens)
        if b == 0: first_branch_tokens = toks
        peak_pages = max(peak_pages, len(mgr.pool.pages))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    bytes_copied = mgr.pool.stat_bytes_copied - bytes_copied_0
    peak_mib = peak_pages * mgr.page_size / MIB
    mgr.pool.destroy()
    return dt, bytes_copied, peak_mib, first_branch_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix_tokens", type=int, default=4096)  # R1 committed headline config
    ap.add_argument("--decode_tokens", type=int, default=128)
    ap.add_argument("--branches", type=int, default=16)  # R1 committed headline config
    args = ap.parse_args()

    _init_torch()
    from decode_layer import QwenLayer0
    t0 = time.time(); L = QwenLayer0(); print(f"loaded Qwen2.5-7B layer-0 in {time.time()-t0:.1f}s "
          f"(n_q={L.n_q} n_kv={L.n_kv} hd={L.hd})")

    P, D, N = args.prefix_tokens, args.decode_tokens, args.branches
    total = N * D
    print(f"prefix_tokens={P} decode_tokens={D} branches={N} -> {total} generated tokens")
    rows = []; seqs = {}
    for regime, fn in (("cow_fork", run_cow), ("full_clone", run_clone)):
        dt, bcopy, peak, seq0 = fn(L, P, D, N)
        seqs[regime] = seq0
        tps = total / dt
        per_branch = peak / N
        print(f"[{regime:10s}] wall={dt:7.3f}s  {tps:8.1f} tok/s  peak_live={peak:9.1f} MiB "
              f"({per_branch:7.1f} MiB/branch)  kv_bytes_copied={bcopy/MIB:8.1f} MiB")
        rows.append((regime, N, P, D, total, dt, tps, peak, bcopy, per_branch))
        log("metric5b_decode",
            dict(regime=regime, n_branches=N, prefix_tokens=P, decode_tokens=D, model="Qwen2.5-7B-layer0"),
            dict(wall_s=dt, tokens_per_s=tps, peak_live_mib=peak, kv_bytes_copied=bcopy,
                 peak_live_per_branch_mib=per_branch))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "n_branches", "prefix_tokens", "decode_tokens", "total_tokens",
                    "wall_s", "tokens_per_s", "peak_live_mib", "kv_bytes_copied",
                    "peak_live_per_branch_mib"])
        w.writerows(rows)
    print("wrote", OUT)
    # correctness: CoW-backed attention must produce identical tokens to full-clone
    match = seqs["cow_fork"] == seqs["full_clone"]
    print(f"\ncorrectness: branch-0 decoded tokens identical CoW vs clone? {match} "
          f"(first 8: cow={seqs['cow_fork'][:8]} clone={seqs['full_clone'][:8]})")
    cow = [r for r in rows if r[0] == "cow_fork"][0]
    clo = [r for r in rows if r[0] == "full_clone"][0]
    print(f"\nHEADLINE (single-layer, N={N} branches, {P}-tok prefix, {D}-tok decode each):")
    print(f"  peak HBM: CoW {cow[7]:.0f} MiB vs clone {clo[7]:.0f} MiB "
          f"({100*(1-cow[7]/clo[7]):.0f}% reduction)")
    print(f"  KV bytes copied: CoW {cow[8]/MIB:.0f} MiB vs clone {clo[8]/MIB:.0f} MiB")
    print(f"  tok/s: CoW {cow[6]:.0f} vs clone {clo[6]:.0f} (compute-bound, ~parity)")


if __name__ == "__main__":
    main()
