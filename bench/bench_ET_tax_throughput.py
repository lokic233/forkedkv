"""
E-T: Contiguous-VA VMM Tax / high-fanout throughput collapse.

ONE decisive end-to-end experiment for the committee. A high-fanout tree-of-thought /
branching-decode workload measuring SYSTEM THROUGHPUT (tokens/sec) and peak feasible
fanout, two ways on the SAME workload using a REAL Qwen2.5-7B layer-0:

  ARM-HW : KVBranchManager VMM CoW (cuMemMap/cuMemSetAccess per page). Each branch forks
           off a shared prefix snapshot (zero-copy alias), then decodes `decode_tokens`
           tokens with REAL attention; the first decode-token write into the shared tail
           prefix page triggers a real hardware CoW remap. Sweeping fanout B upward toward
           and PAST the predicted per-context mapping ceiling K/(maps_per_fork). At the
           ceiling cuMemSetAccess OOMs -> we record the crash gracefully (watchdog).

  ARM-SW : SoftwarePrefixSharingManager (vLLM-APC style block-table + refcounts) backing
           the SAME real-attention decode. KV bytes live in ONE shared contiguous torch
           tensor pool; fork = block-table copy + refcount++ (no HBM, no mapping). CoW on
           the shared tail block copies one block. No per-context mapping ceiling: bounded
           only by the (large) block pool / RAM. Runs the SAME B sweep, INCLUDING B values
           where HW has already crashed.

THE QUESTION: does increasing fanout produce a measured THROUGHPUT COLLAPSE / hard
capacity wall for ARM-HW that ARM-SW does not suffer (HW crashes at K while SW scales on),
and across all measured B does HW EVER beat SW on throughput (is the VMM-CoW win-region
empty end-to-end)?

SAFETY: small bounded prefix; watchdog stops at the mapping ceiling (catches CudaCallError
OOM at cuMemSetAccess) instead of thrashing HBM. Each HW fanout point runs in a FRESH
subprocess so a driver OOM never poisons a later context. HBM is capped well within H100
headroom (live <= a few GiB at the ceiling).

Output: data/et_tax_throughput.csv
  arm, prefix_pages, fanout_B, throughput_tok_s, peak_hbm_mib, crashed, crash_call, reps, notes
"""
import sys, os, csv, time, json, argparse, subprocess, statistics

# ---- cuda-python v13 compat shim: repo uses `from cuda import cuda` (v12 layout) ----
from cuda.bindings import driver as _drv
import cuda as _cudapkg
sys.modules['cuda.cuda'] = _drv
setattr(_cudapkg, 'cuda', _drv)

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, SRC)
OUT = os.path.join(HERE, "..", "data", "et_tax_throughput.csv")
MIB = 1024 * 1024

# ---- workload config (SMALL + BOUNDED) ----
# DESIGN: the per-context VMM mapping ceiling K is driven by the NUMBER OF PAGES aliased
# per fork, NOT by how many tokens those pages logically hold. So we map PREFIX_PAGES_PER_
# RANGE physical pages per K/V range (drives the ceiling) but keep the LOGICAL prefix
# sequence small (PREFIX_SEQ tokens, all inside page 0) so attention stays cheap & bounded.
# The first decode token writes into page 0 (a SHARED aliased prefix page) -> real HW CoW.
PREFIX_PAGES_PER_RANGE = 256     # phys pages mapped per K/V range (ceiling driver)
PREFIX_SEQ = 64                  # logical prefix tokens (live in page 0) -> cheap attention
DECODE_TOKENS = 8                # short decode per branch
REPS = 3
HARD_HBM_CAP_MIB = 60000         # watchdog: never let live HBM exceed this (~60 GiB of 97)
PAGE = 2 * 1024 * 1024
# K ~= 523404 measured. maps per HW fork = 2 ranges * PREFIX_PAGES_PER_RANGE (+1 tail each
# when the first decode token grows a fresh page; we keep prefix page-aligned so the tail
# stays inside the shared prefix page -> forces CoW, the realistic agentic case).
K_CEILING = 523404


def maps_per_fork(prefix_pages_per_range):
    return 2 * prefix_pages_per_range  # K + V


def ceiling_B(prefix_pages_per_range):
    return K_CEILING // maps_per_fork(prefix_pages_per_range)


def peak_hbm_mib():
    import torch
    free, total = torch.cuda.mem_get_info()
    return (total - free) / MIB


# =====================================================================================
# ARM-HW: VMM CoW
# =====================================================================================
def run_hw(prefix_pages, fanout_B, decode_tokens, rep_idx):
    """Returns dict(throughput_tok_s, peak_hbm_mib, crashed, crash_call, decoded_tokens)."""
    import torch
    from kv_branch_manager import KVBranchManager
    from decode_layer import QwenLayer0, BranchKV
    from vmm_pool import CudaCallError

    L = QwenLayer0()
    torch.cuda.synchronize()
    mgr = KVBranchManager(device_id=0)
    mgr.pool.va_pool_enabled = False  # isolate the raw per-context mapping ceiling

    # build a real prefix (page-aligned to PREFIX_PAGES_PER_RANGE so the last prefix token
    # sits inside the final shared page -> first decode write triggers a real CoW).
    bkv = _build_prefix(mgr, L, prefix_pages)
    snapK = mgr.snapshot("pK"); snapV = mgr.snapshot("pV")
    prefix_seq = bkv.seq

    crashed = False; crash_call = "none"
    decoded = 0
    branches_done = 0
    last_hbm = peak_hbm_mib()
    AccErr = getattr(torch, "AcceleratorError", None)
    cuda_excs = tuple(x for x in (AccErr,) if x is not None) + (RuntimeError,)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        for b in range(fanout_B):
            if (b & 31) == 0:
                last_hbm = peak_hbm_mib()      # last-known-good HBM (pre-crash)
                if last_hbm > HARD_HBM_CAP_MIB:
                    crashed = True; crash_call = "hbm_watchdog"; break
            # FORK = the mapping op that drives the per-context ceiling. mgr.fork ends with a
            # cuCtxSynchronize, so a clean driver OOM at cuMemSetAccess surfaces HERE as a
            # CudaCallError (the A* ceiling signature). Past the descriptor budget the table
            # goes inconsistent -> a later kernel reads an unmapped VA (illegal access).
            mgr.fork(snapK, f"c{b}K"); mgr.fork(snapV, f"c{b}V")
            ck = BranchKV(mgr, L.n_kv, L.hd, f"c{b}K", f"c{b}V", reset=False)
            ck.seq = prefix_seq
            tok = 1
            for step in range(decode_tokens):
                pos = ck.seq
                h, q, k, v = L.project(tok, pos)
                ck.append_token(k, v)          # CoW fires on shared tail prefix page
                K_all = ck.k_view(); V_all = ck.v_view()
                logits = L.attend_and_logits(h, q, K_all, V_all)
                tok = int(logits.argmax())
                decoded += 1
            torch.cuda.synchronize()           # localize a ceiling fault to THIS branch
            branches_done = b + 1
    except CudaCallError as e:
        crashed = True
        crash_call = (e.call_site if e.is_oom else f"NONOOM:{e.call_site}")
    except cuda_excs as e:
        sm = str(e).lower()
        crashed = True
        if "out of memory" in sm or "error 2" in sm:
            crash_call = "cuMemSetAccess(OOM)"
        elif "illegal" in sm or "cudaerrorillegal" in sm:
            crash_call = "cuMemSetAccess_ceiling(illegal_access)"
        else:
            crash_call = f"RT:{sm[:48]}"
    dt = time.perf_counter() - t0
    # POST-CRASH: the context may be poisoned -> NO further CUDA calls. Use last-known HBM.
    if not crashed:
        try:
            torch.cuda.synchronize(); last_hbm = peak_hbm_mib()
        except Exception:
            pass
        try:
            mgr.pool.destroy()
        except Exception:
            pass
    thr = decoded / dt if dt > 0 else 0.0
    res = dict(throughput_tok_s=thr, peak_hbm_mib=last_hbm, crashed=crashed,
               crash_call=crash_call, decoded_tokens=decoded,
               branches_completed=branches_done, wall_s=dt)
    return res


def _build_prefix(mgr, L, prefix_pages):
    """Map `prefix_pages` physical pages per K/V range (this drives the per-context
    mapping ceiling), but only fill the FIRST PREFIX_SEQ tokens (all inside page 0) with
    real projected K/V. The remaining pages are mapped-but-logically-unused: they still
    cost one (VA->phys) access descriptor each at fork time (the mapping tax) while keeping
    attention compute bounded to PREFIX_SEQ tokens. This isolates the MAPPING tax."""
    from decode_layer import BranchKV
    # reserve VA + map prefix_pages physical pages on each range
    mgr.create_branch("pK", prefix_pages); mgr.branches["pK"].num_pages = 0
    mgr.create_branch("pV", prefix_pages); mgr.branches["pV"].num_pages = 0
    for _ in range(prefix_pages):
        mgr.append_page("pK"); mgr.append_page("pV")   # maps a fresh phys page each
    bkv = BranchKV(mgr, L.n_kv, L.hd, "pK", "pV", reset=False)
    # fill only the first PREFIX_SEQ tokens with REAL projected K/V (page 0)
    tok = 1
    for pos in range(PREFIX_SEQ):
        h, q, k, v = L.project(tok, pos)
        # write directly into the page-0 views without growing pages (already mapped)
        Kv = bkv._kv_views(bkv.kid); Vv = bkv._kv_views(bkv.vid)
        Kv[:, pos, :] = k; Vv[:, pos, :] = v
    bkv.seq = PREFIX_SEQ
    return bkv


# =====================================================================================
# ARM-SW: software prefix sharing (vLLM-APC style) backing the SAME real attention decode
# =====================================================================================
def run_sw(prefix_pages, fanout_B, decode_tokens, rep_idx):
    """Software prefix sharing (vLLM-APC style). KV bytes live in ONE shared contiguous
    torch tensor (prefix), shared by all branches via a refcounted block table (RAM, no
    GPU mapping). Fork = block-table copy + refcount++. CoW on the shared tail block copies
    one block. SAME real Qwen attention decode as ARM-HW. No per-context mapping ceiling:
    bounded only by the (large) RAM-side refcount table. We keep the logical prefix to
    PREFIX_SEQ tokens (same attention work as HW) so the ONLY difference vs HW is the KV
    memory mechanism (block table vs cuMemMap aliasing)."""
    import torch
    from decode_layer import QwenLayer0

    L = QwenLayer0()
    torch.cuda.synchronize()

    # ONE shared contiguous prefix KV (PREFIX_SEQ tokens), computed once -> zero per-branch
    # prefix HBM; branches alias it via the refcount table.
    Kpre = torch.empty(L.n_kv, PREFIX_SEQ, L.hd, dtype=torch.float16, device="cuda")
    Vpre = torch.empty(L.n_kv, PREFIX_SEQ, L.hd, dtype=torch.float16, device="cuda")
    tok = 1
    for pos in range(PREFIX_SEQ):
        h, q, k, v = L.project(tok, pos)
        Kpre[:, pos, :] = k; Vpre[:, pos, :] = v
    torch.cuda.synchronize()

    decoded = 0
    crashed = False; crash_call = "none"
    # vLLM refcount table: one entry per prefix block (RAM bound, no mapping ceiling).
    n_prefix_blocks = prefix_pages
    refcounts = {bid: 1 for bid in range(n_prefix_blocks)}
    base_table = list(range(n_prefix_blocks))

    t0 = time.perf_counter()
    try:
        for b in range(fanout_B):
            if (b & 31) == 0 and peak_hbm_mib() > HARD_HBM_CAP_MIB:
                crashed = True; crash_call = "hbm_watchdog"; break
            # software fork: copy block table + refcount++ (pure RAM; no GPU mapping)
            child_table = list(base_table)
            for bid in child_table:
                refcounts[bid] += 1
            # Software block CoW: the active prefix block (where the first decode token
            # lands) is SHARED, so we privatize exactly THAT one block — mirroring HW's
            # single-page CoW. The rest of the prefix stays shared zero-copy (Kpre/Vpre).
            cow_k = Kpre[:, :, :].clone() if PREFIX_SEQ <= 0 else Kpre.narrow(1, 0, PREFIX_SEQ).clone()
            cow_v = Vpre.narrow(1, 0, PREFIX_SEQ).clone()
            # per-branch private decode tail
            Ktail = torch.empty(L.n_kv, decode_tokens, L.hd, dtype=torch.float16, device="cuda")
            Vtail = torch.empty(L.n_kv, decode_tokens, L.hd, dtype=torch.float16, device="cuda")
            tok = 1
            for step in range(decode_tokens):
                pos = PREFIX_SEQ + step
                h, q, k, v = L.project(tok, pos)
                Ktail[:, step, :] = k; Vtail[:, step, :] = v
                K_all = torch.cat([cow_k, Ktail[:, :step+1, :]], dim=1)
                V_all = torch.cat([cow_v, Vtail[:, :step+1, :]], dim=1)
                logits = L.attend_and_logits(h, q, K_all, V_all)
                tok = int(logits.argmax())
                decoded += 1
            for bid in child_table:
                refcounts[bid] -= 1
            del Ktail, Vtail, cow_k, cow_v
    except RuntimeError as e:
        sm = str(e).lower()
        crashed = True
        crash_call = "OOM_runtime" if ("out of memory" in sm or "error 2" in sm) else f"RT:{sm[:40]}"
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    hbm = peak_hbm_mib()
    thr = decoded / dt if dt > 0 else 0.0
    return dict(throughput_tok_s=thr, peak_hbm_mib=hbm, crashed=crashed,
                crash_call=crash_call, decoded_tokens=decoded,
                branches_completed=fanout_B if not crashed else 0, wall_s=dt)


def _emit(d):
    print("RESULT " + json.dumps(d))


# =====================================================================================
# child entrypoint: run ONE (arm, B, rep) in a fresh process
# =====================================================================================
def _child(arm, prefix_pages, fanout_B, decode_tokens, rep_idx):
    import torch
    torch.cuda.init(); torch.cuda.set_device(0)
    if arm == "hw":
        res = run_hw(prefix_pages, fanout_B, decode_tokens, rep_idx)
    else:
        res = run_sw(prefix_pages, fanout_B, decode_tokens, rep_idx)
    _emit(res)


# =====================================================================================
# orchestrator
# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix_pages", type=int, default=PREFIX_PAGES_PER_RANGE)
    ap.add_argument("--decode_tokens", type=int, default=DECODE_TOKENS)
    ap.add_argument("--reps", type=int, default=REPS)
    args = ap.parse_args()

    P = args.prefix_pages
    Bceil = ceiling_B(P)
    # decision-relevant sweep: dense at low B (where HW survives) + at/around the wall.
    # includes 0.5x, 0.9x, 1.1x of the predicted alias-only ceiling Bceil.
    sweep = sorted(set([4, 16, 64, 128, 256,
                        int(0.5 * Bceil), int(0.9 * Bceil), int(1.1 * Bceil)]))
    print(f"prefix_pages/range={P}  maps_per_fork={maps_per_fork(P)}  "
          f"predicted ceiling B≈{Bceil}  sweep={sweep}")

    py = sys.executable
    rows = []
    for arm in ["hw", "sw"]:
        for B in sweep:
            thrs = []; hbms = []; crash_calls = []; branches_at_crash = []
            any_crash = False; notes_parts = []
            for rep in range(args.reps):
                r = subprocess.run(
                    [py, os.path.abspath(__file__), "child", arm, str(P), str(B),
                     str(args.decode_tokens), str(rep)],
                    capture_output=True, text=True, timeout=1800)
                line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
                if not line:
                    any_crash = True; crash_calls.append("child_died")
                    notes_parts.append(f"rep{rep}:NO_RESULT")
                    print(f"[{arm} B={B} rep{rep}] CHILD FAILED: {r.stderr[-200:]}")
                    continue
                d = json.loads(line[-1][len("RESULT "):])
                thrs.append(d["throughput_tok_s"]); hbms.append(d["peak_hbm_mib"])
                if d["crashed"]:
                    any_crash = True; crash_calls.append(d["crash_call"])
                    branches_at_crash.append(d.get("branches_completed", 0))
                    notes_parts.append(f"rep{rep}:bdone={d.get('branches_completed')}")
                else:
                    notes_parts.append(f"rep{rep}:ok")
            thr = round(statistics.median(thrs), 2) if thrs else 0.0
            hbm = round(max(hbms), 1) if hbms else 0.0
            reps_done = len(thrs)
            # pick the most common crash call; record median crash branch
            crash_call = "none"
            if any_crash:
                crash_call = statistics.mode(crash_calls) if crash_calls else "crash"
                if branches_at_crash:
                    crash_call += f"@B~{int(statistics.median(branches_at_crash))}"
            notes = "; ".join(notes_parts)
            print(f"[{arm:2}] B={B:>5}  thr={thr:9.1f} tok/s  hbm={hbm:7.0f} MiB  "
                  f"crashed={any_crash} call={crash_call} reps={reps_done}  {notes}")
            rows.append(dict(arm=arm, prefix_pages=P, fanout_B=B,
                             throughput_tok_s=thr, peak_hbm_mib=hbm,
                             crashed=any_crash, crash_call=crash_call, reps=reps_done,
                             notes=notes))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "prefix_pages", "fanout_B", "throughput_tok_s", "peak_hbm_mib",
                    "crashed", "crash_call", "reps", "notes"])
        for r in rows:
            w.writerow([r["arm"], r["prefix_pages"], r["fanout_B"], r["throughput_tok_s"],
                        r["peak_hbm_mib"], r["crashed"], r["crash_call"], r["reps"], r["notes"]])
    print("wrote", OUT)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        _, _, arm, P, B, dt, rep = sys.argv
        _child(arm, int(P), int(B), int(dt), int(rep))
    else:
        main()
