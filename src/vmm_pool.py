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


def _ck(ret):
    if isinstance(ret, tuple):
        err = ret[0]; rest = ret[1:]
    else:
        err = ret; rest = ()
    if int(err) != 0:
        name = cuda.cuGetErrorName(err)[1]
        try: name = name.decode()
        except Exception: pass
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

    def create_phys_page(self):
        h = _ck(cuda.cuMemCreate(self.page_size, self._prop, 0))
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
        va = _ck(cuda.cuMemAddressReserve(size, 0, 0, 0))
        return int(va), size

    def free_va(self, va, size):
        _ck(cuda.cuMemAddressFree(va, size))

    def map_page(self, va_addr, pg):
        _ck(cuda.cuMemMap(va_addr, self.page_size, 0, pg.handle, 0))
        _ck(cuda.cuMemSetAccess(va_addr, self.page_size, [self._access], 1))
        self.stat_map_ops += 1

    def unmap_page(self, va_addr):
        _ck(cuda.cuMemUnmap(va_addr, self.page_size))

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
        _ck(cuda.cuCtxSynchronize())
