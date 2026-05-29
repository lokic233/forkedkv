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
REPS = 30
WARMUP = 10
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


def main():
    rows = []
    for sl in SEQLENS:
        c = run_method_contig(sl)
        v = run_method_vmm(sl)
        cm, vm = statistics.mean(c), statistics.mean(v)
        ovh = 100*(vm/cm - 1)
        for r,(a,b) in enumerate(zip(c, v)):
            rows.append((sl, "contiguous", r, a))
            rows.append((sl, "vmm_paged", r, b))
        print(f"seqlen={sl:5d}  contig={cm:7.4f}ms (sd {statistics.pstdev(c):.4f})  "
              f"vmm={vm:7.4f}ms (sd {statistics.pstdev(v):.4f})  overhead={ovh:+5.1f}%")
        log("metric3_attn_overhead", dict(seqlen=sl, heads=HEADS, head_dim=HEAD_DIM, reps=REPS, dtype="fp16"),
            dict(contig_ms_mean=cm, contig_ms_sd=statistics.pstdev(c),
                 vmm_ms_mean=vm, vmm_ms_sd=statistics.pstdev(v), overhead_pct=ovh))
    with open(OUT,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["seqlen","method","rep","ms"]); w.writerows(rows)
    print("wrote", OUT)

if __name__=="__main__":
    main()
