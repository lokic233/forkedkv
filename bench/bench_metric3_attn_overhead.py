"""
Metric 3: Attention kernel overhead from VMM page indirection.

We compare attention over KV stored in:
  (A) a standard contiguous torch tensor (baseline), and
  (B) KV physically backed by VMM pages mapped into a reserved VA range, exposed to
      torch via a tensor whose storage points at the VMM VA.

Because our VMM VA range is CONTIGUOUS (cuMemAddressReserve gives one VA range, each
page mapped in order), the attention kernel sees ordinary contiguous memory — the
indirection is in the *page table*, not in the kernel's address arithmetic. So we
expect ~0 kernel overhead. We MEASURE it rather than assume.

We build a torch tensor aliasing the VMM VA via torch.cuda caching allocator bypass:
we cudaMemcpy KV into the VMM VA, then wrap with torch.frombuffer-style construction
using a tensor created from the raw device pointer.

Output: data/metric3_attn_overhead.csv columns: seqlen, method, rep, ms
Microbenchmark (single-head SDPA, fp16).
"""
import sys, os, csv, ctypes, statistics, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from cuda import cuda
from vmm_pool import VMMPool, _ck
from explog import log

HEADS, HEAD_DIM = 32, 128       # ~Qwen2.5-7B-ish single layer K/V shape
DTYPE = torch.float16
SEQLENS = [512, 1024, 2048, 4096, 8192]
REPS = 50
WARMUP = 50   # B2: was 10; clock-drift artifact at 8192 needs longer warmup
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "metric3_attn_overhead.csv")


def make_vmm_tensor(pool, nbytes):
    """Reserve+map enough VMM pages to hold nbytes, return (va, torch_tensor_uint8, keep)."""
    npages = (nbytes + pool.page_size - 1) // pool.page_size
    va, size = pool.reserve_va(npages)
    pages = []
    for i in range(npages):
        pg = pool.create_phys_page()
        pool.map_page(va + i*pool.page_size, pg)
        pages.append(pg)
    # wrap VMM VA as a torch tensor via UntypedStorage from the raw pointer
    t = torch.empty(0, dtype=torch.uint8, device="cuda")
    # build tensor from external pointer using torch.cuda's from_blob equivalent
    return va, size, pages, npages


def tensor_from_va(va, numel, dtype, shape):
    # Use torch's __cuda_array_interface__ consumer: wrap with cupy-like dict via torch
    import torch
    # torch.as_strided needs a storage; we create a tensor that views external memory
    # through torch.frombuffer is host-only, so use the DLPack-free path:
    # create a uint8 tensor over the VA using torch.Tensor with a custom storage.
    elem_size = torch.tensor([], dtype=dtype).element_size()
    nbytes = numel * elem_size
    # Build via __cuda_array_interface__
    class _Arr:
        def __init__(self, ptr, nbytes):
            self.__cuda_array_interface__ = dict(
                shape=(nbytes,), typestr='|u1', data=(ptr, False), version=3, strides=None)
    arr = _Arr(int(va), nbytes)
    u8 = torch.as_tensor(arr, device='cuda')   # uint8 view, zero-copy
    return u8.view(dtype).view(*shape)


def run_method_vmm(seqlen):
    pool = VMMPool(device_id=0)
    # K and V tensors: [1, HEADS, seqlen, HEAD_DIM]
    numel = HEADS * seqlen * HEAD_DIM
    nbytes = numel * 2  # fp16
    vak, *_ = make_vmm_tensor(pool, nbytes)
    vav, *_ = make_vmm_tensor(pool, nbytes)
    K = tensor_from_va(vak, numel, DTYPE, (1, HEADS, seqlen, HEAD_DIM))
    V = tensor_from_va(vav, numel, DTYPE, (1, HEADS, seqlen, HEAD_DIM))
    K.normal_(); V.normal_()
    Q = torch.randn(1, HEADS, seqlen, HEAD_DIM, dtype=DTYPE, device='cuda')
    return _bench_sdpa(Q, K, V)


def run_method_contig(seqlen):
    Q = torch.randn(1, HEADS, seqlen, HEAD_DIM, dtype=DTYPE, device='cuda')
    K = torch.randn(1, HEADS, seqlen, HEAD_DIM, dtype=DTYPE, device='cuda')
    V = torch.randn(1, HEADS, seqlen, HEAD_DIM, dtype=DTYPE, device='cuda')
    return _bench_sdpa(Q, K, V)


def _bench_sdpa(Q, K, V):
    import torch.nn.functional as F
    for _ in range(WARMUP):
        o = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()
    times = []
    for _ in range(REPS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        o = F.scaled_dot_product_attention(Q, K, V)
        e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return times


def _bench_sdpa_interleaved(Qc, Kc, Vc, Qv, Kv, Vv):
    """B2: measure contiguous and VMM-paged SDPA INTERLEAVED within each rep so any
    GPU clock drift (boost ramp / thermal) affects both methods equally and cancels in
    the ratio. Returns (contig_times, vmm_times)."""
    import torch.nn.functional as F
    for _ in range(WARMUP):
        F.scaled_dot_product_attention(Qc, Kc, Vc)
        F.scaled_dot_product_attention(Qv, Kv, Vv)
    torch.cuda.synchronize()
    ct, vt = [], []
    for _ in range(REPS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); F.scaled_dot_product_attention(Qc, Kc, Vc); e.record(); torch.cuda.synchronize()
        ct.append(s.elapsed_time(e))
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); F.scaled_dot_product_attention(Qv, Kv, Vv); e.record(); torch.cuda.synchronize()
        vt.append(s.elapsed_time(e))
    return ct, vt


def main():
    rows = []
    print(f"WARMUP={WARMUP} REPS={REPS} (interleaved contig/vmm per rep -> clock drift cancels)")
    for sl in SEQLENS:
        # build BOTH tensor sets, then interleave timing
        pool = VMMPool(device_id=0)
        numel = HEADS * sl * HEAD_DIM; nbytes = numel * 2
        vak, *_ = make_vmm_tensor(pool, nbytes); vav, *_ = make_vmm_tensor(pool, nbytes)
        Kv = tensor_from_va(vak, numel, DTYPE, (1, HEADS, sl, HEAD_DIM))
        Vv = tensor_from_va(vav, numel, DTYPE, (1, HEADS, sl, HEAD_DIM))
        Kv.normal_(); Vv.normal_()
        Qv = torch.randn(1, HEADS, sl, HEAD_DIM, dtype=DTYPE, device='cuda')
        Qc = torch.randn(1, HEADS, sl, HEAD_DIM, dtype=DTYPE, device='cuda')
        Kc = torch.randn(1, HEADS, sl, HEAD_DIM, dtype=DTYPE, device='cuda')
        Vc = torch.randn(1, HEADS, sl, HEAD_DIM, dtype=DTYPE, device='cuda')
        c, v = _bench_sdpa_interleaved(Qc, Kc, Vc, Qv, Kv, Vv)
        # use MEDIAN (robust to the occasional boost-clock outlier) for the headline ratio
        cm, vm = statistics.median(c), statistics.median(v)
        ovh = 100*(vm/cm - 1)
        for r,(a,b) in enumerate(zip(c, v)):
            rows.append((sl, "contiguous", r, a))
            rows.append((sl, "vmm_paged", r, b))
        print(f"seqlen={sl:5d}  contig_med={cm:7.4f}ms (sd {statistics.pstdev(c):.4f})  "
              f"vmm_med={vm:7.4f}ms (sd {statistics.pstdev(v):.4f})  overhead={ovh:+5.1f}%")
        log("metric3_attn_overhead", dict(seqlen=sl, heads=HEADS, head_dim=HEAD_DIM, reps=REPS, warmup=WARMUP,
                dtype="fp16", method="interleaved", stat="median"),
            dict(contig_ms_median=cm, contig_ms_sd=statistics.pstdev(c),
                 vmm_ms_median=vm, vmm_ms_sd=statistics.pstdev(v), overhead_pct=ovh))
        del Kv, Vv, Qv, Qc, Kc, Vc; pool.destroy(); torch.cuda.empty_cache()
    with open(OUT,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["seqlen","method","rep","ms"]); w.writerows(rows)
    print("wrote", OUT)

if __name__=="__main__":
    main()
