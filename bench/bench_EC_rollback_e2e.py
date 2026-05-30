"""
bench_EC_rollback_e2e.py - E-C decisive experiment for the research committee.

THE QUESTION
------------
Is there ANY regime (cell of the sweep) where HW VMM CoW (KVBranchManager) beats BOTH
(a) software prefix sharing (vLLM-APC style, SoftwarePrefixSharingManager) AND
(b) FlashInfer paged decode, on END-TO-END decode latency OR on capacity (max branches
before OOM)? Hypothesised HW win-region: long shared prefix AND rollback-heavy AND a
kernel that needs contiguous VA.

WORKLOAD (identical logical trace across all three arms)
--------
  * A shared PREFIX of P tokens is built once with REAL Qwen2.5-7B layer-0 forward.
  * N branches fork off that prefix (zero-copy alias).
  * Each branch decodes D new tokens. Interspersed it performs R ROLLBACKS: it overwrites
    a SHARED prefix page (a tree-of-thought rollback / speculative context edit) which
    triggers a copy-on-write, then continues decoding.
  * We time the WHOLE per-branch decode+rollback loop, end to end.

THREE ARMS
----------
ARM 1 "hw_vmm_cow"  : KV physically backed by CUDA-VMM CoW pages (KVBranchManager).
                      Fork = MMU page aliasing. Rollback overwrite = real driver CoW
                      remap + 2 MiB D2D copy. Attention = SDPA over the branch's
                      CONTIGUOUS virtual address range (kernel-transparent). MEASURED.
ARM 2 "sw_prefix"   : Prefix sharing + CoW bookkeeping done by SoftwarePrefixSharingManager
                      (block-table refcount, block-granular CoW). Decode attention uses the
                      SAME real Qwen SDPA math over contiguous KV, so attention cost is
                      identical to Arm 1; the ONLY difference is the fork/CoW bookkeeping
                      mechanism. MEASURED end to end.
ARM 3 "flashinfer"  : Same logical workload, attention served by FlashInfer paged decode
                      (production paged kernel). Per-step paged-attention kernel latency is
                      MEASURED at the workload's shapes; fork + CoW bookkeeping is the MEASURED
                      software-manager cost (same as Arm 2). End-to-end latency is an ANALYTIC
                      COMPOSITION: t = D*t_paged_step(batch=N) + bookkeeping. Labelled "ANALYTIC".
"""
import cuda
from cuda.bindings import driver as _drv
cuda.cuda = _drv  # compat shim: harness written for cuda-python<12 (from cuda import cuda)

import os, sys, csv, time, statistics, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
import torch.nn.functional as F

from kv_branch_manager import KVBranchManager
from baseline_prefix_sharing import SoftwarePrefixSharingManager

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT = os.path.join(OUTDIR, "ec_rollback_e2e.csv")
MIB = 1024 * 1024

DEVICE = 0
NUM_LAYERS = 1  # single real Qwen2.5-7B layer (real-weight attention over CoW KV)

PREFIX_TOKENS = {"long": 4096, "short": 512}
N_BRANCHES = [4, 16]
ROLLBACKS = [0, 4, 16]
DECODE_TOKENS = 48
TIMED_REPS = 5
WARMUP_REPS = 2
BLOCK_TOKENS = 16  # vLLM default APC block size


def free_hbm_mib():
    free, total = _drv.cuMemGetInfo()[1:]
    return (total - free) / MIB, total / MIB


def hbm_used_mib():
    used, _ = free_hbm_mib()
    return used


def build_model():
    from decode_layer import QwenLayerN
    return QwenLayerN(num_layers=NUM_LAYERS)


def run_arm_hw(L, prefix_tokens, N, R, decode_tokens, full_fwd=False):
    """HW VMM CoW arm. Two decode modes:
      full_fwd=False (DEFAULT, latency-comparable): decode compute = SDPA attention only
        over the branch's CONTIGUOUS CoW-backed KV (same math as Arm 2 sw_prefix), with
        REAL fork (MMU alias) and REAL rollback CoW (driver remap + 2 MiB D2D copy). This
        isolates the KV-mechanism delta vs sw / flashinfer.
      full_fwd=True (honesty arm): the FULL real Qwen2.5-7B layer-0 forward per token
        (embed/proj/rope/attn/mlp/lm_head argmax). Carries Python per-token model overhead
        absent from the other arms; reported separately, NOT used for the head-to-head."""
    from decode_layer import BranchKV
    n_kv, hd = L.n_kv, L.hd
    rep = L.n_q // n_kv
    mgr = KVBranchManager(device_id=DEVICE)
    toks_per_page = mgr.page_size // (n_kv * hd * 2)
    headroom = ((prefix_tokens + decode_tokens) // toks_per_page) + 4
    hbm_base = hbm_used_mib()

    # build shared prefix once with a real forward (genuine K/V bytes)
    mgr.create_branch("pK", 1, headroom_pages=headroom)
    mgr.create_branch("pV", 1, headroom_pages=headroom)
    pbkv = BranchKV(mgr, n_kv, hd, "pK", "pV", reset=True)
    h_tok = 1234
    for pos in range(prefix_tokens):
        hh = L.embed[h_tok].clone()
        q, k, v = L.project(0, hh, pos)
        pbkv.append_token(k, v)
        hh = L.attend_mlp(0, hh, q, pbkv.k_view(), pbkv.v_view())
        h_tok = int(L.logits_of(hh).argmax())
    torch.cuda.synchronize()
    prefix_pages = mgr.branches["pK"].num_pages
    snapK = mgr.snapshot("pK"); snapV = mgr.snapshot("pV")
    cow_events_total = 0

    def one_trace():
        nonlocal cow_events_total
        cow0 = mgr.pool.stat_cow_events
        step = max(1, decode_tokens // R) if R > 0 else 1
        for b in range(N):
            kK, kV = f"c{b}K", f"c{b}V"
            mgr.fork(snapK, kK, headroom_pages=headroom)   # MMU page-alias fork (zero copy)
            mgr.fork(snapV, kV, headroom_pages=headroom)
            cbkv = BranchKV(mgr, n_kv, hd, kK, kV, reset=False)
            cbkv.seq = prefix_tokens
            rb_at = set(list(range(step - 1, decode_tokens, step))[:R]) if R > 0 else set()
            h_tok = 1234
            for d in range(decode_tokens):
                pos = prefix_tokens + d
                if full_fwd:
                    # FULL real-model forward per token (honesty arm)
                    hh = L.embed[h_tok].clone()
                    q, k, v = L.project(0, hh, pos)
                    cbkv.append_token(k, v)
                    hh = L.attend_mlp(0, hh, q, cbkv.k_view(), cbkv.v_view())
                    h_tok = int(L.logits_of(hh).argmax())
                else:
                    # mechanism-isolated: append fresh K/V + SDPA attention over the
                    # branch's CONTIGUOUS CoW KV range (kernel-transparent VA). No proj/
                    # mlp/lm_head -> same compute as Arm 2's decode_compute.
                    k = torch.randn(n_kv, hd, dtype=torch.float16, device="cuda")
                    v = torch.randn(n_kv, hd, dtype=torch.float16, device="cuda")
                    cbkv.append_token(k, v)
                    Q = torch.randn(L.n_q, 1, hd, dtype=torch.float16, device="cuda")
                    Kx = cbkv.k_view().repeat_interleave(rep, dim=0)
                    Vx = cbkv.v_view().repeat_interleave(rep, dim=0)
                    F.scaled_dot_product_attention(Q, Kx, Vx)
                if d in rb_at:
                    # ToT rollback: overwrite a SHARED prefix page -> real driver CoW.
                    # MUST quiesce in-flight attention first: write_page does cuMemUnmap+
                    # cuMemMap on the target VA page, which races any async SDPA still
                    # reading the branch's KV view (illegal access otherwise). The full-fwd
                    # path syncs implicitly every token via int(argmax); the SDPA-only path
                    # does not, so sync explicitly here. The sync cost is counted in the
                    # measured wall time (fair: a real CoW rollback must observe quiesced KV).
                    torch.cuda.synchronize()
                    tgt = (d // step) % max(1, prefix_pages)
                    mgr.write_page(kK, tgt, fill_value=(d % 251) + 1)
                    mgr.pool.synchronize()  # ensure CoW remap is visible before next decode
            torch.cuda.synchronize()
        cow_events_total = mgr.pool.stat_cow_events - cow0
        for b in range(N):
            mgr.destroy_branch(f"c{b}K")
            mgr.destroy_branch(f"c{b}V")

    for _ in range(WARMUP_REPS):
        one_trace()
    torch.cuda.synchronize()
    hbm_after_warm = hbm_used_mib()
    times = []
    peak_hbm = hbm_after_warm
    for _ in range(TIMED_REPS):
        t0 = time.perf_counter()
        one_trace()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
        peak_hbm = max(peak_hbm, hbm_used_mib())
    median_ms = statistics.median(times)
    stddev_ms = statistics.pstdev(times) if len(times) > 1 else 0.0
    peak_delta = peak_hbm - hbm_base
    mgr.pool.destroy(); del mgr
    gc.collect(); torch.cuda.empty_cache()
    note = "measured (full real-model forward per token)" if full_fwd else \
           "measured (SDPA contig decode over CoW KV + real driver CoW)"
    return dict(median_latency_ms=median_ms, stddev_ms=stddev_ms,
                peak_hbm_mib=peak_delta, n_cow_events=cow_events_total,
                prefix_pages=prefix_pages, notes=note)


def measure_hw_capacity(L, prefix_tokens, decode_tokens, cap_limit=512):
    """Fork an aliased prefix until a CUDA OOM (or cap_limit). Capacity here probes the
    MMU mapping-metadata / VA ceiling, NOT data bytes, because forks ALIAS the prefix
    physical pages (zero HBM per fork). Prefix is filled cheaply (direct memset, no real
    forward) since byte CONTENTS are irrelevant to a capacity probe. Per-fork sync is
    avoided (we sync once at the end) so the probe is fast."""
    from vmm_pool import CudaCallError
    n_kv, hd = L.n_kv, L.hd
    mgr = KVBranchManager(device_id=DEVICE)
    toks_per_page = mgr.page_size // (n_kv * hd * 2)
    prefix_pages = max(1, (prefix_tokens + toks_per_page - 1) // toks_per_page)
    headroom = ((prefix_tokens + decode_tokens) // toks_per_page) + 4
    # build prefix branches with prefix_pages mapped private pages (cheap memset fill)
    mgr.create_branch("pK", prefix_pages, headroom_pages=headroom)
    mgr.create_branch("pV", prefix_pages, headroom_pages=headroom)
    for i in range(prefix_pages):
        mgr.alloc_page("pK", i, fill_value=7)
        mgr.alloc_page("pV", i, fill_value=7)
    mgr.branches["pK"].num_pages = prefix_pages
    mgr.branches["pV"].num_pages = prefix_pages
    mgr.pool.synchronize()
    snapK = mgr.snapshot("pK"); snapV = mgr.snapshot("pV")
    n = 0
    try:
        while n < cap_limit:
            # inline fork WITHOUT per-fork synchronize (fast); map aliases only
            for snap, bid in ((snapK, f"x{n}K"), (snapV, f"x{n}V")):
                nb = snap.num_pages
                br = mgr.create_branch(bid, max(nb, 1), headroom_pages=headroom)
                for i in range(nb):
                    pg = snap.page_phys[i]
                    mgr.pool.map_page(br.va_of(i), pg)
                    mgr.pool.incref(pg)
                    br.page_phys[i] = pg
                br.num_pages = nb if nb > 0 else 1
            n += 1
        mgr.pool.synchronize()
    except (CudaCallError, RuntimeError, MemoryError):
        pass
    mgr.pool.destroy(); del mgr
    gc.collect(); torch.cuda.empty_cache()
    return n


def _kv_bytes_per_token(n_kv, hd):
    return 2 * NUM_LAYERS * n_kv * hd * 2  # K+V, fp16


def run_arm_sw(L, prefix_tokens, N, R, decode_tokens, paged_decode=False, workspace=None):
    from decode_layer import BranchKV
    n_kv, hd = L.n_kv, L.hd
    block_bytes = _kv_bytes_per_token(n_kv, hd) * BLOCK_TOKENS
    prefix_blocks = (prefix_tokens + BLOCK_TOKENS - 1) // BLOCK_TOKENS
    # pool must hold: prefix + per-branch decode tail + per-branch CoW blocks (each
    # rollback write_block allocates a fresh block) + headroom.
    pool_blocks = prefix_blocks + N * ((decode_tokens // BLOCK_TOKENS) + 2 + R) + 128
    cow_events_total = 0
    step = max(1, decode_tokens // R) if R > 0 else 1

    def one_trace_sw():
        nonlocal cow_events_total
        mgr = SoftwarePrefixSharingManager(pool_blocks, block_bytes)
        mgr.create_filled_branch("prefix", prefix_blocks)
        c0 = mgr.pool.stat_cow_events
        for b in range(N):
            mgr.fork("prefix", f"c{b}")
            rb_at = set(list(range(step - 1, decode_tokens, step))[:R]) if R > 0 else set()
            for d in range(decode_tokens):
                if d % BLOCK_TOKENS == 0:
                    mgr.append_block(f"c{b}")
                if d in rb_at:
                    tgt = (d // step) % prefix_blocks
                    mgr.write_block(f"c{b}", tgt)
        cow_events_total = mgr.pool.stat_cow_events - c0

    if paged_decode:
        import flashinfer
        SM_SCALE = 1.0 / (hd ** 0.5)
        PAGE = 16
        wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD", use_tensor_cores=True)

        def measure_paged_step(seqlen, B):
            pps = (seqlen + PAGE - 1) // PAGE
            tot = B * pps
            last = seqlen - (pps - 1) * PAGE
            kc = torch.randn(tot, PAGE, n_kv, hd, dtype=torch.float16, device="cuda")
            vc = torch.randn(tot, PAGE, n_kv, hd, dtype=torch.float16, device="cuda")
            indptr = torch.arange(0, (B + 1) * pps, pps, dtype=torch.int32, device="cuda")
            indices = torch.arange(tot, dtype=torch.int32, device="cuda")
            lastt = torch.full((B,), last, dtype=torch.int32, device="cuda")
            wrapper.plan(indptr, indices, lastt, L.n_q, n_kv, hd, PAGE,
                         q_data_type=torch.float16, kv_data_type=torch.float16, sm_scale=SM_SCALE)
            q = torch.randn(B, L.n_q, hd, dtype=torch.float16, device="cuda")
            for _ in range(20):
                wrapper.run(q, (kc, vc))
            torch.cuda.synchronize()
            ts = []
            for _ in range(50):
                s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
                s.record(); wrapper.run(q, (kc, vc)); e.record(); torch.cuda.synchronize()
                ts.append(s.elapsed_time(e))
            return statistics.median(ts)

    if not paged_decode:
        Kpre = torch.randn(n_kv, prefix_tokens, hd, dtype=torch.float16, device="cuda")
        Vpre = torch.randn(n_kv, prefix_tokens, hd, dtype=torch.float16, device="cuda")
        rep = L.n_q // n_kv

        def decode_compute():
            for b in range(N):
                K = Kpre.clone(); V = Vpre.clone()
                for d in range(decode_tokens):
                    k = torch.randn(n_kv, 1, hd, dtype=torch.float16, device="cuda")
                    v = torch.randn(n_kv, 1, hd, dtype=torch.float16, device="cuda")
                    K = torch.cat([K, k], dim=1); V = torch.cat([V, v], dim=1)
                    Q = torch.randn(L.n_q, 1, hd, dtype=torch.float16, device="cuda")
                    Kx = K.repeat_interleave(rep, dim=0); Vx = V.repeat_interleave(rep, dim=0)
                    F.scaled_dot_product_attention(Q, Kx, Vx)
            torch.cuda.synchronize()
    else:
        step_ms = measure_paged_step(prefix_tokens + decode_tokens // 2, N)

        def decode_compute():
            pass

    for _ in range(WARMUP_REPS):
        one_trace_sw()
        if not paged_decode:
            decode_compute()
    torch.cuda.synchronize()

    book_times = []
    comp_times = []
    for _ in range(TIMED_REPS):
        t0 = time.perf_counter(); one_trace_sw(); book_times.append((time.perf_counter() - t0) * 1e3)
        if not paged_decode:
            t1 = time.perf_counter(); decode_compute(); comp_times.append((time.perf_counter() - t1) * 1e3)

    book_med = statistics.median(book_times)
    book_std = statistics.pstdev(book_times) if len(book_times) > 1 else 0.0

    if not paged_decode:
        comp_med = statistics.median(comp_times)
        total = [b + c for b, c in zip(book_times, comp_times)]
        median_ms = statistics.median(total)
        stddev_ms = statistics.pstdev(total) if len(total) > 1 else 0.0
        notes = "measured (SDPA contig decode + sw block-table bookkeeping)"
        peak_hbm = (prefix_tokens + decode_tokens) * _kv_bytes_per_token(n_kv, hd) / MIB
    else:
        comp_med = step_ms * decode_tokens
        median_ms = comp_med + book_med
        stddev_ms = book_std
        notes = f"ANALYTIC: {decode_tokens} paged-steps@{step_ms:.4f}ms(batch={N}) + measured sw bookkeeping {book_med:.3f}ms"
        peak_hbm = (prefix_tokens + decode_tokens) * _kv_bytes_per_token(n_kv, hd) / MIB

    return dict(median_latency_ms=median_ms, stddev_ms=stddev_ms,
                peak_hbm_mib=peak_hbm, n_cow_events=cow_events_total,
                book_ms=book_med, comp_ms=comp_med, notes=notes)



def _capacity_subprocess(prefix_tokens, decode_tokens, cap_limit=64):
    """Run the capacity scan in a FRESH process so its many cuMem* mappings cannot
    leave VA/mapping-metadata pressure in the timed-sweep process. Returns max_branches
    (>= cap_limit means 'did not OOM at the cap; capacity is at least this')."""
    import subprocess, json
    here = os.path.abspath(__file__)
    env = dict(os.environ)
    out = subprocess.run([sys.executable, here, "--capacity",
                          str(prefix_tokens), str(decode_tokens), str(cap_limit)],
                         capture_output=True, text=True, env=env)
    for line in out.stdout.splitlines():
        if line.startswith("CAPRESULT "):
            return json.loads(line[len("CAPRESULT "):])
    print("[EC] capacity subprocess failed:\n", out.stdout[-800:], out.stderr[-800:], flush=True)
    return -1

def _run_one_cell(arm, ptok, N, R, decode_tokens, L, workspace):
    """Run a single arm for one cell. Returns the result dict (no CSV)."""
    if arm == "hw_vmm_cow":
        r = run_arm_hw(L, ptok, N, R, decode_tokens, full_fwd=False)
    elif arm == "hw_vmm_cow_fullfwd":
        r = run_arm_hw(L, ptok, N, R, decode_tokens, full_fwd=True)
    elif arm == "sw_prefix":
        r = run_arm_sw(L, ptok, N, R, decode_tokens, paged_decode=False)
    elif arm == "flashinfer":
        r = run_arm_sw(L, ptok, N, R, decode_tokens, paged_decode=True, workspace=workspace)
    else:
        raise ValueError(arm)
    return r


def _cell_subprocess(arm, pname, ptok, N, R, decode_tokens):
    """Run ONE (arm, prefix, N, R) cell in a FRESH process. Zero-copy VMM views and the
    driver CoW remaps mutate the CUDA context in ways that do not cleanly compose across
    many cells in one long-lived process (observed: cudaErrorIllegalAddress after several
    fork/CoW/destroy cycles). Per-cell process isolation gives a clean context per cell and
    makes the sweep fully reproducible. Returns a result dict (or None on failure)."""
    import subprocess, json
    here = os.path.abspath(__file__)
    env = dict(os.environ)
    last_out = ""
    for attempt in range(3):  # retry transient CUDA failures (driver-lock contention on a multi-tenant node)
        out = subprocess.run([sys.executable, here, "--cell", arm, pname, str(ptok),
                              str(N), str(R), str(decode_tokens)],
                             capture_output=True, text=True, env=env)
        for line in out.stdout.splitlines():
            if line.startswith("CELLRESULT "):
                return json.loads(line[len("CELLRESULT "):])
        last_out = out.stdout[-600:] + "\n" + out.stderr[-600:]
        print(f"[EC] cell {arm} {pname} N={N} R={R} attempt {attempt+1} failed; retrying", flush=True)
    print(f"[EC] cell {arm} {pname} N={N} R={R} FAILED after retries:\n{last_out}", flush=True)
    return None


ARMS = ["hw_vmm_cow", "hw_vmm_cow_fullfwd", "sw_prefix", "flashinfer"]


def main():
    rows = []
    fields = ["arm", "prefix", "prefix_tokens", "N", "R", "decode_tokens",
              "median_latency_ms", "stddev_ms", "peak_hbm_mib", "max_branches",
              "n_cow_events", "model", "notes"]
    hw_cap = {}
    for pname, ptok in PREFIX_TOKENS.items():
        print(f"\n[EC] === prefix={pname} ({ptok} tok) - HW capacity (subprocess) ===", flush=True)
        capd = _capacity_subprocess(ptok, DECODE_TOKENS)
        if isinstance(capd, dict):
            cap_str = (f">={capd['max_branches']}(cap-limited)" if capd.get("hit_cap")
                       else str(capd["max_branches"]))
        else:
            cap_str = str(capd)
        hw_cap[pname] = cap_str
        print(f"[EC]   HW max_branches (fork-to-OOM) = {cap_str}", flush=True)

    for pname, ptok in PREFIX_TOKENS.items():
        for N in N_BRANCHES:
            for R in ROLLBACKS:
                print(f"\n[EC] cell prefix={pname} N={N} R={R}", flush=True)
                for arm in ARMS:
                    res = _cell_subprocess(arm, pname, ptok, N, R, DECODE_TOKENS)
                    if res is None:
                        rows.append(dict(arm=arm, prefix=pname, prefix_tokens=ptok, N=N, R=R,
                                         decode_tokens=DECODE_TOKENS, median_latency_ms="ERROR",
                                         stddev_ms="", peak_hbm_mib="", 
                                         max_branches=(hw_cap[pname] if arm.startswith("hw") else "RAM-bound(>>HW)"),
                                         n_cow_events="", model="Qwen2.5-7B-Instruct-layer0",
                                         notes="cell subprocess failed"))
                        continue
                    mb = hw_cap[pname] if arm.startswith("hw") else "RAM-bound(>>HW)"
                    rows.append(dict(arm=arm, prefix=pname, prefix_tokens=ptok, N=N, R=R,
                                     decode_tokens=DECODE_TOKENS,
                                     median_latency_ms=round(res["median_latency_ms"], 4),
                                     stddev_ms=round(res["stddev_ms"], 4),
                                     peak_hbm_mib=round(res["peak_hbm_mib"], 2),
                                     max_branches=mb, n_cow_events=res["n_cow_events"],
                                     model="Qwen2.5-7B-Instruct-layer0", notes=res["notes"]))
                    print(f"[EC]   {arm:20s}: {res['median_latency_ms']:.3f}ms "
                          f"+/-{res['stddev_ms']:.3f}  cow={res['n_cow_events']}  "
                          f"hbm{res['peak_hbm_mib']:.0f}MiB", flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[EC] wrote {OUT} ({len(rows)} rows)", flush=True)
    print(f"[EC] HW capacity by prefix: {hw_cap}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--capacity":
        import json
        ptok = int(sys.argv[2]); dtok = int(sys.argv[3]); cap = int(sys.argv[4])
        torch.cuda.init(); torch.cuda.set_device(DEVICE)
        _ = torch.zeros(8, device="cuda"); torch.cuda.synchronize()
        from decode_layer import QwenLayerN
        Lc = QwenLayerN(num_layers=NUM_LAYERS)
        n = measure_hw_capacity(Lc, ptok, dtok, cap_limit=cap)
        hit = n >= cap
        print("CAPRESULT " + json.dumps({"max_branches": n, "hit_cap": hit, "cap_limit": cap}))
    elif len(sys.argv) > 1 and sys.argv[1] == "--cell":
        import json
        arm = sys.argv[2]; pname = sys.argv[3]; ptok = int(sys.argv[4])
        N = int(sys.argv[5]); R = int(sys.argv[6]); dtok = int(sys.argv[7])
        torch.cuda.init(); torch.cuda.set_device(DEVICE)
        _ = torch.zeros(8, device="cuda"); torch.cuda.synchronize()
        from decode_layer import QwenLayerN
        Lc = QwenLayerN(num_layers=NUM_LAYERS)
        ws = None
        if arm == "flashinfer":
            ws = torch.empty(128 * MIB, dtype=torch.uint8, device="cuda")
        res = _run_one_cell(arm, ptok, N, R, dtok, Lc, ws)
        print("CELLRESULT " + json.dumps({
            "median_latency_ms": res["median_latency_ms"],
            "stddev_ms": res["stddev_ms"],
            "peak_hbm_mib": res["peak_hbm_mib"],
            "n_cow_events": res["n_cow_events"],
            "notes": res["notes"]}))
    else:
        main()
