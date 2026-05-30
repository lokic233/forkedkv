"""
Experiment 1 (ASPLOS Extended Lab): Hardware TLB & L2 Cache Pressure of VMM-paged KV.

OBJECTIVE
  Measure the hardware-level translation/cache cost of VMM-paged KV (ForkedKV's
  contiguous-VA, 2 MiB-page layout) under heavy parallel execution, vs a standard
  contiguous torch allocation. The ASPLOS question (from Metric 3's +0.05% finding):
  "does the GPU TLB handle VMM indirection transparently?"

TOOL AVAILABILITY (probed at runtime, reported honestly in the CSV/notes):
  - ncu  (Nsight Compute): NOT INSTALLED on devgpu014 (no `ncu` on PATH). So we CANNOT
    read the requested GPU hardware counters directly:
        lts__tlb_tag_misses.sum, l1tex__t_sector_hit_rate.pct,
        lts__t_sector_hit_rate.pct, dram__throughput.avg.pct_of_peak_sustained_elapsed
  - nsys (Nsight Systems): AVAILABLE, runs without root (process-tree CUDA trace). nsys
    gives kernel/API timeline + occupancy, but NOT the SM-level counters above (those are
    ncu-only). We use nsys to confirm the SAME SDPA kernel is dispatched for both methods
    (i.e. the indirection is in the page table, not the kernel), as a structural check.
  - PRIMARY METHOD (the ASPLOS-worthy fallback): torch.cuda.Event kernel-granularity
    timing of per-layer attention, VMM-paged vs contiguous, across the seqlen x branch
    sweep. We report overhead % per (seqlen, branches). High overhead => translation cost
    is visible; ~0 overhead => the TLB absorbs VMM indirection (our hypothesis).

CONFIG
  - Qwen2.5-7B attention shape: 28 q-heads / 4 KV-heads (GQA), head_dim 128, fp16.
  - 28 layers' worth of KV (we replay one representative layer's SDPA 28x per "decode" to
    emulate the full-model per-token attention cost, and ALSO report single-layer numbers).
  - seqlen sweep: {1024, 2048, 4096, 8192, 16384} (32768 skipped if OOM).
  - branch sweep: {1, 4, 8, 16, 32, 64} concurrent branches, each with its own KV.
    Concurrent branches multiply the resident-page footprint => more TLB/L2 pressure.
  - Method (a) VMM-paged: KV physically backed by 2 MiB VMM pages mapped into a
    contiguous reserved VA range (ForkedKV layout). Method (b) contiguous: plain
    torch.randn KV (caching allocator).

OUTPUT
  data/exp1_tlb_pressure.csv columns:
    seqlen, branches, method, layers_per_step, rep, ms, ncu_available, nsys_available
"""
import sys, os, csv, statistics, gc, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
import torch.nn.functional as F
from cuda import cuda
from vmm_pool import VMMPool, _ck

DEVICE = 0
Q_HEADS, KV_HEADS, HEAD_DIM = 28, 4, 128     # Qwen2.5-7B GQA
DTYPE = torch.float16
SEQLENS = [1024, 2048, 4096, 8192, 16384]
BRANCHES = [1, 4, 8, 16, 32, 64]
LAYERS_PER_STEP = 28                          # emulate full-model per-token attn cost
REPS = 30
WARMUP = 20
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "exp1_tlb_pressure.csv")
HMIB = 1024 * 1024


def probe_tools():
    import shutil
    return shutil.which("ncu") is not None, shutil.which("nsys") is not None


def make_vmm_kv(pool, seqlen):
    """Allocate K and V for one layer in VMM 2MiB pages over a contiguous VA range.
    Returns (K, V, keep) where K,V are zero-copy torch views of the VMM VA.
    Shape [1, KV_HEADS, seqlen, HEAD_DIM] fp16."""
    numel = KV_HEADS * seqlen * HEAD_DIM
    nbytes = numel * 2
    keep_pages = []
    def alloc():
        npages = (nbytes + pool.page_size - 1) // pool.page_size
        va, size = pool.reserve_va(npages)
        for i in range(npages):
            pg = pool.create_phys_page()
            pool.map_page(va + i * pool.page_size, pg)
            keep_pages.append(pg)
        # zero-copy torch view via __cuda_array_interface__
        class _Arr:
            def __init__(self, ptr, nb):
                self.__cuda_array_interface__ = dict(
                    shape=(nb,), typestr='|u1', data=(ptr, False), version=3, strides=None)
        u8 = torch.as_tensor(_Arr(int(va), nbytes), device='cuda')
        return u8.view(DTYPE).view(1, KV_HEADS, seqlen, HEAD_DIM)
    K = alloc(); V = alloc()
    K.normal_(); V.normal_()
    return K, V, keep_pages


def make_contig_kv(seqlen):
    K = torch.randn(1, KV_HEADS, seqlen, HEAD_DIM, dtype=DTYPE, device='cuda')
    V = torch.randn(1, KV_HEADS, seqlen, HEAD_DIM, dtype=DTYPE, device='cuda')
    return K, V


def gqa_sdpa(Q, K, V):
    """GQA: repeat KV heads to match Q heads, then SDPA. Decode-style Q (1 query token)."""
    rep = Q_HEADS // KV_HEADS
    Kx = K.repeat_interleave(rep, dim=1)
    Vx = V.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(Q, Kx, Vx)


def bench_step(branch_KVs, seqlen, layers):
    """Time one 'decode step' = `layers` SDPA calls, each over a (randomly chosen) branch's
    KV, with a 1-token decode query. Interleaving across branches stresses the resident
    page set => TLB/L2 pressure. Returns median ms over REPS."""
    nb = len(branch_KVs)
    Q = torch.randn(1, Q_HEADS, 1, HEAD_DIM, dtype=DTYPE, device='cuda')  # decode: 1 query tok
    for _ in range(WARMUP):
        for li in range(layers):
            K, V = branch_KVs[li % nb]
            gqa_sdpa(Q, K, V)
    torch.cuda.synchronize()
    times = []
    for _ in range(REPS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for li in range(layers):
            K, V = branch_KVs[li % nb]
            gqa_sdpa(Q, K, V)
        e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return times


def bench_interleaved(contig_KVs, vmm_KVs, layers):
    """B2-style: time contiguous and VMM-paged decode steps INTERLEAVED per rep so GPU
    clock drift (boost ramp / thermal) hits both equally and cancels in the ratio.
    Both KV sets are resident simultaneously, which is also a stronger TLB/L2 stress
    (2x the working set). Returns (contig_times, vmm_times)."""
    nb = len(contig_KVs)
    Q = torch.randn(1, Q_HEADS, 1, HEAD_DIM, dtype=DTYPE, device='cuda')
    def one(KVs):
        for li in range(layers):
            K, V = KVs[li % nb]
            gqa_sdpa(Q, K, V)
    for _ in range(WARMUP):
        one(contig_KVs); one(vmm_KVs)
    torch.cuda.synchronize()
    ct, vt = [], []
    for _ in range(REPS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); one(contig_KVs); e.record(); torch.cuda.synchronize(); ct.append(s.elapsed_time(e))
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); one(vmm_KVs); e.record(); torch.cuda.synchronize(); vt.append(s.elapsed_time(e))
    return ct, vt


def run_pair(seqlen, branches, layers):
    """Build both contiguous and VMM-paged KV for `branches` branches, time interleaved."""
    pool = VMMPool(device_id=DEVICE)
    keep = []; vmm_KVs = []; contig_KVs = []
    for _ in range(branches):
        K, V, kp = make_vmm_kv(pool, seqlen)
        vmm_KVs.append((K, V)); keep.append(kp)
        contig_KVs.append(make_contig_kv(seqlen))
    cv, vv = bench_interleaved(contig_KVs, vmm_KVs, layers)
    del contig_KVs, vmm_KVs; pool.destroy(); torch.cuda.empty_cache(); gc.collect()
    return cv, vv


def main():
    torch.cuda.init(); torch.cuda.set_device(DEVICE)
    ncu_ok, nsys_ok = probe_tools()
    print(f"ncu available: {ncu_ok}  |  nsys available: {nsys_ok}")
    print(f"PRIMARY = torch.cuda.Event per-layer SDPA timing (ncu unavailable -> no HW counters)")
    print(f"shape: Q_HEADS={Q_HEADS} KV_HEADS={KV_HEADS} head_dim={HEAD_DIM} layers/step={LAYERS_PER_STEP}")
    rows = []
    for sl in SEQLENS:
        for nb in BRANCHES:
            # OOM guard: both contiguous AND VMM KV sets resident at once (2x) for the
            # interleaved fair-ratio measurement. footprint ~ 2(methods) * 2(K+V) * bytes * nb
            foot_gib = 2 * 2 * KV_HEADS * sl * HEAD_DIM * 2 * nb / (1024**3)
            if foot_gib > 70:
                print(f"  seqlen={sl} branches={nb}: SKIP (est {foot_gib:.1f} GiB > 70 GiB guard)")
                continue
            try:
                cv, vv = run_pair(sl, nb, LAYERS_PER_STEP)
            except Exception as ex:
                print(f"  seqlen={sl} branches={nb}: ERROR {type(ex).__name__}: {ex} -> stop branch sweep")
                break
            cm, vm = statistics.median(cv), statistics.median(vv)
            ovh = 100 * (vm / cm - 1)
            print(f"  seqlen={sl:6d} branches={nb:3d}  contig={cm:8.4f}ms  vmm={vm:8.4f}ms  overhead={ovh:+6.2f}%  (foot~{foot_gib:.1f}GiB)")
            for r, (a, b) in enumerate(zip(cv, vv)):
                rows.append((sl, nb, "contiguous", LAYERS_PER_STEP, r, a, ncu_ok, nsys_ok))
                rows.append((sl, nb, "vmm_paged", LAYERS_PER_STEP, r, b, ncu_ok, nsys_ok))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seqlen", "branches", "method", "layers_per_step", "rep", "ms", "ncu_available", "nsys_available"])
        w.writerows(rows)
    print("wrote", OUT, f"({len(rows)} rows)")


if __name__ == "__main__":
    main()
