"""
Lab 2 ncu target: a single Python invocation that builds Q/K/V tensors with a chosen
backing (contiguous or VMM-paged) and a chosen seqlen, then issues exactly ONE SDPA
call inside a cudaProfilerStart/Stop region. Used by bench_lab2_ncu_counters.py:
ncu attaches with --profile-from-start no, captures only the kernels inside the region.

Usage:  python _lab2_target.py <method> <seqlen>
where  <method> in {contig, vmm}.
"""
import os, sys, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F

method = sys.argv[1]; seqlen = int(sys.argv[2])
import sys as _sys
def _dbg(msg):
    _sys.stderr.write(f"[lab2_target {method}/{seqlen}] {msg}\n"); _sys.stderr.flush()
_dbg("imported torch")

HEADS, HEAD_DIM = 32, 128
DTYPE = torch.float16

# build Q (always contig — this is the query, identical in both methods)
Q = torch.randn(1, HEADS, seqlen, HEAD_DIM, dtype=DTYPE, device='cuda')
_dbg("built Q")

if method == "contig":
    K = torch.randn(1, HEADS, seqlen, HEAD_DIM, dtype=DTYPE, device='cuda')
    V = torch.randn(1, HEADS, seqlen, HEAD_DIM, dtype=DTYPE, device='cuda')
    keep_alive = ()
else:
    from vmm_pool import VMMPool, _ck
    from cuda import cuda
    pool = VMMPool(device_id=0)

    def make_vmm_tensor(numel, shape):
        nbytes = numel * 2  # fp16
        npages = (nbytes + pool.page_size - 1) // pool.page_size
        va, size = pool.reserve_va(npages)
        pages = []
        for i in range(npages):
            pg = pool.create_phys_page()
            pool.map_page(va + i*pool.page_size, pg)
            pages.append(pg)
        class _Arr:
            def __init__(self, ptr, n):
                self.__cuda_array_interface__ = dict(
                    shape=(n,), typestr='|u1', data=(ptr, False), version=3, strides=None)
        u8 = torch.as_tensor(_Arr(int(va), nbytes), device='cuda')
        return u8.view(DTYPE).view(*shape), (va, size, pages)

    numel = HEADS*seqlen*HEAD_DIM
    K, kk = make_vmm_tensor(numel, (1, HEADS, seqlen, HEAD_DIM))
    V, vv = make_vmm_tensor(numel, (1, HEADS, seqlen, HEAD_DIM))
    K.normal_(); V.normal_()
    keep_alive = (pool, kk, vv)
_dbg("built K,V")

# warm: ensure cuBLAS/SDPA backend is selected and JIT done OUTSIDE the profiled region
for _ in range(20):
    o = F.scaled_dot_product_attention(Q, K, V)
torch.cuda.synchronize()
_dbg("warmed")

# ---- profiled region: exactly one SDPA call ----
torch.cuda.cudart().cudaProfilerStart()
_dbg("profilerStart returned")
o = F.scaled_dot_product_attention(Q, K, V)
torch.cuda.synchronize()
_dbg("sdpa+sync done")
torch.cuda.cudart().cudaProfilerStop()
_dbg("profilerStop returned")
# -------------------------------------------------

# keep alive until exit so ncu doesn't see weird unmaps mid-profile
del o
_dbg("done")
