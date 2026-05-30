"""
vmm_pool.py — CUDA Virtual Memory Management (VMM) backed physical-page pool.

Owns physical GPU memory in fixed-size pages (one cuMemCreate handle per page) and
the virtual address space (cuMemAddressReserve). Exposes primitives for branch-aware
copy-on-write of KV-cache pages:
  - create_phys_page  (cuMemCreate)            -> PhysPage(refcount)
  - reserve_va        (cuMemAddressReserve)
  - map_page          (cuMemMap + cuMemSetAccess) -> aliasing (shared handle)
  - unmap_page / copy_page                      -> CoW remap
  - incref/decref reference counting on physical pages

CoW unit = one physical page = CU_MEM_ALLOC_GRANULARITY_MINIMUM (2 MiB on H100).

Forking a branch does NOT copy KV bytes: reserve new VA, map each new VA page to the
SAME physical handle as the parent (refcount++). A write to a shared page triggers a
software page-fault (KV manager) that allocates a private handle, copies 2 MiB, and
remaps that single VA page. OS-style CoW over the GPU MMU.
"""
import ctypes
from cuda import cuda


class CudaCallError(RuntimeError):
    """A CUDA driver error annotated with the call site (P0-2 / gemini R2-3 forensics)."""
    def __init__(self, errcode, errname, call_site):
        self.errcode = int(errcode)
        self.errname = errname
        self.call_site = call_site
        super().__init__(f"CUDA driver error {int(errcode)}: {errname} at call_site={call_site!r}")

    @property
    def is_oom(self):
        return self.errcode == 2  # CUDA_ERROR_OUT_OF_MEMORY


def _ck(ret, call_site=None):
    """Check a cuda-python return. If `call_site` is given, OOM/errors are annotated with
    the exact driver call that failed (forensic OOM evidence for Metric 4)."""
    if isinstance(ret, tuple):
        err = ret[0]; rest = ret[1:]
    else:
        err = ret; rest = ()
    if int(err) != 0:
        name = cuda.cuGetErrorName(err)[1]
        try: name = name.decode()
        except Exception: pass
        if call_site is not None:
            raise CudaCallError(err, name, call_site)
        raise RuntimeError(f"CUDA driver error {int(err)}: {name}")
    if len(rest) == 0: return None
    if len(rest) == 1: return rest[0]
    return rest


class PhysPage:
    """A single physical GPU allocation handle (cuMemCreate) with a refcount."""
    __slots__ = ("handle", "refcount", "page_id", "born_from")
    def __init__(self, handle, page_id):
        self.handle = handle
        self.refcount = 1
        self.page_id = page_id
        self.born_from = None   # parent page_id if created by a CoW copy


class VMMPool:
    """Physical-page pool + VA manager for KV cache on one device."""
    def __init__(self, device_id=0, page_size=None):
        _ck(cuda.cuInit(0))
        self.dev = _ck(cuda.cuDeviceGet(device_id))
        self.ctx = _ck(cuda.cuDevicePrimaryCtxRetain(self.dev))
        _ck(cuda.cuCtxSetCurrent(self.ctx))
        self.device_id = device_id

        self._prop = cuda.CUmemAllocationProp()
        self._prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        self._prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        self._prop.location.id = device_id
        min_gran = _ck(cuda.cuMemGetAllocationGranularity(
            self._prop, cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM))
        self.page_size = page_size or min_gran
        assert self.page_size % min_gran == 0

        self._access = cuda.CUmemAccessDesc()
        self._access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        self._access.location.id = device_id
        self._access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        self._next_page_id = 0
        self.pages = {}
        self.stat_phys_pages_created = 0
        self.stat_bytes_created = 0
        self.stat_bytes_copied = 0
        self.stat_cow_events = 0
        self.stat_map_ops = 0

        # P0-2: process-wide VA free-list keyed by num_pages. On release, a VA range is
        # NOT cuMemAddressFree'd; it is parked here and re-handed-out for a same-size
        # reservation. This turns Metric 4's 84-branch "mapping-metadata ceiling" into a
        # measurement of the TRUE (physical-data) ceiling, because freed branches' VA
        # reservations are recycled instead of accumulating.
        self.va_pool = {}                 # num_pages -> [ (va, size), ... ]
        self.va_pool_enabled = True
        self.stat_va_reserved = 0         # cuMemAddressReserve calls actually issued
        self.stat_va_reused = 0           # reservations served from the free-list
        self.stat_va_freed = 0            # cuMemAddressFree calls actually issued
        self.stat_va_pooled = 0           # ranges parked in the free-list
        self.last_oom_call_site = None    # forensic: which cuMem* call OOM'd (Metric 4)

    def create_phys_page(self):
        h = _ck(cuda.cuMemCreate(self.page_size, self._prop, 0), call_site="cuMemCreate")
        pid = self._next_page_id; self._next_page_id += 1
        pg = PhysPage(h, pid)
        self.pages[pid] = pg
        self.stat_phys_pages_created += 1
        self.stat_bytes_created += self.page_size
        return pg

    def _release_phys_page(self, pg):
        _ck(cuda.cuMemRelease(pg.handle))
        self.pages.pop(pg.page_id, None)

    def incref(self, pg):
        pg.refcount += 1

    def decref(self, pg):
        pg.refcount -= 1
        if pg.refcount <= 0:
            self._release_phys_page(pg)

    def reserve_va(self, num_pages):
        size = num_pages * self.page_size
        va = _ck(cuda.cuMemAddressReserve(size, 0, 0, 0), call_site="cuMemAddressReserve")
        self.stat_va_reserved += 1
        return int(va), size

    def reserve_va_range(self, num_pages, fixed_addr=0):
        """Reserve a VA range, optionally at a fixed address hint (for contiguous
        extension). Returns (va, size). VA reservation consumes NO physical HBM; only
        cuMemMap of a page commits HBM. This lets a branch reserve generous VA headroom
        upfront and grow its mapped region lazily as decode proceeds (P0-C dynamic VA).

        P0-2: if a same-size range is parked in the process-wide VA free-list, reuse it
        (no cuMemAddressReserve). This recycles freed branches' reservations so capacity
        is bounded by physical data, not by ever-growing VA-mapping metadata."""
        if fixed_addr == 0 and self.va_pool_enabled:
            bucket = self.va_pool.get(num_pages)
            if bucket:
                va, size = bucket.pop()
                self.stat_va_reused += 1
                return va, size
        size = num_pages * self.page_size
        va = _ck(cuda.cuMemAddressReserve(size, 0, fixed_addr, 0),
                 call_site="cuMemAddressReserve")
        self.stat_va_reserved += 1
        return int(va), size

    def free_va(self, va, size, num_pages=None):
        """Return a VA range. P0-2: with pooling enabled, park it in the free-list keyed
        by size instead of cuMemAddressFree, so it can be re-handed-out. The caller MUST
        have unmapped all pages in the range first (cuMemAddressFree requires unmapped)."""
        if num_pages is None:
            num_pages = size // self.page_size
        if self.va_pool_enabled:
            self.va_pool.setdefault(num_pages, []).append((int(va), size))
            self.stat_va_pooled += 1
            return
        _ck(cuda.cuMemAddressFree(va, size), call_site="cuMemAddressFree")
        self.stat_va_freed += 1

    def map_page(self, va_addr, pg):
        _ck(cuda.cuMemMap(va_addr, self.page_size, 0, pg.handle, 0), call_site="cuMemMap")
        _ck(cuda.cuMemSetAccess(va_addr, self.page_size, [self._access], 1),
            call_site="cuMemSetAccess")
        self.stat_map_ops += 1

    def unmap_page(self, va_addr):
        _ck(cuda.cuMemUnmap(va_addr, self.page_size), call_site="cuMemUnmap")

    def copy_page(self, dst_va, src_va):
        _ck(cuda.cuMemcpyDtoD(dst_va, src_va, self.page_size))
        self.stat_bytes_copied += self.page_size

    def memset_page(self, va_addr, value=0):
        _ck(cuda.cuMemsetD8(va_addr, value, self.page_size))

    def retained_handle_at(self, va_addr):
        """Driver handle currently backing this VA page. Two VA pages with the same
        retained handle are PHYSICALLY aliased (proves CoW sharing at the MMU)."""
        h = _ck(cuda.cuMemRetainAllocationHandle(va_addr))
        ident = int(h)
        _ck(cuda.cuMemRelease(h))
        return ident

    def synchronize(self):
        _ck(cuda.cuCtxSynchronize(), call_site="cuCtxSynchronize")

    def drain_va_pool(self):
        """Actually cuMemAddressFree every parked VA range (real teardown)."""
        for bucket in self.va_pool.values():
            for va, size in bucket:
                try:
                    _ck(cuda.cuMemAddressFree(va, size))
                    self.stat_va_freed += 1
                except Exception:
                    pass
        self.va_pool.clear()

    def destroy(self):
        """Release every live physical handle. Call before discarding a pool to free HBM."""
        for pg in list(self.pages.values()):
            try:
                _ck(cuda.cuMemRelease(pg.handle))
            except Exception:
                pass
        self.pages.clear()
        self.drain_va_pool()
