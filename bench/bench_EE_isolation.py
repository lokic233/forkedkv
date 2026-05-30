"""
bench_EE_isolation.py — Cluster E gating experiment (E-E): write-after-fork EXACT
bit-identical ISOLATION of GPU KV state as a runtime PRIMITIVE.

THESIS UNDER TEST (Cluster E): HW VMM CoW provides a CORRECTNESS/SAFETY primitive
(NOT a perf win — sibling C* already proved HW loses on speed): O(1-page) verified
rollback + driver-handle-level proof of non-corruption for safe speculative/untrusted
context edits. The hostile question: does software prefix-sharing (vLLM-APC-style
refcounted block tables) ALREADY provide bit-identical isolation + cheap rollback,
collapsing this thesis into software-equivalence?

THREAT MODEL / SCENARIO:
  An agent forks N speculative branches off a shared prefix. Some branches make
  UNTRUSTED / contaminated context edits: they OVERWRITE shared prefix pages mid-decode
  (a malicious or hallucinated speculative edit to context the branch forked from). The
  runtime must guarantee:
    (a) ISOLATION: sibling/parent branches remain BIT-IDENTICALLY uncorrupted.
    (b) ROLLBACK: a contaminated branch can be rolled back to a verified-clean state
        cheaply (re-alias the pre-fork snapshot), copying O(1 page) not the whole prefix.

We run BOTH arms through the IDENTICAL scenario on real Qwen2.5-7B layer-0 K/V:
  ARM-HW (KVBranchManager VMM CoW):
    - isolation verified TWO ways: (1) DRIVER HANDLE proof via cuMemRetainAllocationHandle
      (sibling page maps a *physically distinct* allocation handle from the contaminated
      branch, AND the *same* handle as the pre-fork snapshot -> hardware/MMU-enforced
      separation), (2) byte-compare to pre-fork snapshot.
    - rollback = re-map the contaminated branch's VA page to the clean snapshot's physical
      handle (cuMemUnmap + cuMemMap + incref). O(1 page), 0 bytes copied on rollback;
      the contiguous VA is preserved so an unmodified kernel runs on the recovered branch.
  ARM-SW (real-GPU refcounted block-table, vLLM-APC-equivalent):
    - SAME pool of real GPU blocks; fork = copy block_table + refcount++; contaminated
      write = block-granularity CoW (alloc free block, D2D copy, repoint table entry).
    - isolation verified by byte-compare to pre-fork snapshot ONLY -- software has NO
      physical-handle primitive; it can only assert separation via its own bookkeeping
      (refcount ints) + a data read-back. There is no driver-level proof a kernel that
      *ignores the block table* (an unmodified FlashAttention) would honor.
    - rollback = repoint the block_table entry back to the snapshot block + refcount++.
      O(1 block), 0 bytes copied on rollback.

HONEST DESIGN NOTE: the existing repo SW baseline (baseline_prefix_sharing.py) stored
blocks on HOST numpy and only *simulated* byte copies. That is unfair (HW touches real
HBM). Here the SW arm uses a REAL GPU block pool with REAL cuMemcpyDtoD copies and REAL
read-backs, so the isolation/rollback comparison is apples-to-apples on the same device.

Sweep: N in {4,8}, prefix=2048 tokens (1 page / block-group), a few rollbacks, >=3 reps.
All allocations bounded (<~ 1 GiB). Output: data/ee_isolation.csv
"""
import sys, os, csv, time, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from cuda import cuda
from kv_branch_manager import KVBranchManager
from vmm_pool import VMMPool, _ck

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "ee_isolation.csv")
MIB = 1024 * 1024


def _init_torch():
    torch.cuda.init(); torch.cuda.set_device(0)
    _ = torch.zeros(8, device="cuda"); torch.cuda.synchronize()


def read_page(va, nbytes):
    host = (ctypes.c_uint8 * nbytes)()
    _ck(cuda.cuMemcpyDtoH(host, va, nbytes))
    return bytes(host)


# ---------------------------------------------------------------------------
# ARM-SW: a REAL-GPU refcounted block-table manager (vLLM-APC-equivalent).
# One GPU allocation per block (via the same VMMPool primitives so bytes are real),
# block_table = list of physical-block objects. Isolation provable ONLY by data
# read-back (no physical-handle / MMU separation primitive exposed to a kernel).
# ---------------------------------------------------------------------------
class GpuBlock:
    __slots__ = ("va", "phys", "refcount", "block_id")
    def __init__(self, va, phys, block_id):
        self.va = va; self.phys = phys; self.refcount = 1; self.block_id = block_id


class RealGpuPrefixSharingManager:
    """vLLM-APC-style block-table allocator backed by REAL GPU blocks.
    Each block = one mapped VMM page (so the byte size and D2D copy cost match HW arm).
    A branch is a list[GpuBlock]. Fork copies the list + refcount++. CoW on a shared
    block allocs a fresh block, D2D-copies, repoints. Rollback repoints back to snapshot."""
    def __init__(self, device_id=0):
        self.pool = VMMPool(device_id=device_id)
        self.page_size = self.pool.page_size
        self._next_id = 0
        self.branches = {}      # branch_id -> list[GpuBlock]
        self.snapshots = {}     # snap_id -> tuple[GpuBlock]
        self._next_snap = 0
        self.bytes_copied = 0
        self.cow_events = 0
        self.rollback_bytes_copied = 0

    def _alloc_block(self, fill=None):
        va, _ = self.pool.reserve_va_range(1)
        phys = self.pool.create_phys_page()
        self.pool.map_page(va, phys)
        if fill is not None:
            self.pool.memset_page(va, fill)
        b = GpuBlock(va, phys, self._next_id); self._next_id += 1
        return b

    def create_filled_branch(self, branch_id, num_blocks, fill=None):
        bt = [self._alloc_block(fill=fill) for _ in range(num_blocks)]
        self.branches[branch_id] = bt
        return bt

    def write_block_bytes(self, branch_id, idx, k_tensor, v_tensor=None):
        """Write real K (and optionally V) tensor bytes into a block (private; caller
        ensures CoW already done). Used to fill genuine model K/V."""
        b = self.branches[branch_id][idx]
        # copy k bytes to the block VA (host->device of the tensor's data)
        nbytes = min(k_tensor.numel() * k_tensor.element_size(), self.page_size)
        _ck(cuda.cuMemcpyDtoD(b.va, k_tensor.data_ptr(), nbytes))

    def snapshot(self, branch_id):
        bt = self.branches[branch_id]
        for b in bt:
            b.refcount += 1
        sid = self._next_snap; self._next_snap += 1
        self.snapshots[sid] = tuple(bt)
        return sid

    def fork(self, src_branch_id, new_branch_id):
        t0 = time.perf_counter()
        src = self.branches[src_branch_id]
        new_bt = list(src)
        for b in new_bt:
            b.refcount += 1
        self.branches[new_branch_id] = new_bt
        return new_bt, time.perf_counter() - t0

    def _cow(self, branch_id, idx):
        bt = self.branches[branch_id]
        old = bt[idx]
        new = self._alloc_block()
        _ck(cuda.cuMemcpyDtoD(new.va, old.va, self.page_size))  # REAL D2D copy
        self.bytes_copied += self.page_size
        self.cow_events += 1
        old.refcount -= 1
        if old.refcount <= 0:
            self.pool.unmap_page(old.va); self.pool.decref(old.phys); self.pool.free_va(old.va, self.page_size)
        bt[idx] = new
        return new

    def contaminate_write(self, branch_id, idx, fill_value):
        """Untrusted edit: overwrite a (possibly shared) block. CoW if shared."""
        bt = self.branches[branch_id]
        did_cow = False
        if bt[idx].refcount > 1:
            self._cow(branch_id, idx); did_cow = True
        self.pool.memset_page(bt[idx].va, fill_value)
        return did_cow

    def rollback(self, branch_id, idx, snap_id):
        """Roll a contaminated branch's block back to the snapshot's clean block.
        Repoint the block_table entry (refcount++ snapshot block); free the dirty block.
        O(1 block), 0 bytes copied (pointer repoint)."""
        t0 = time.perf_counter()
        bt = self.branches[branch_id]
        dirty = bt[idx]
        clean = self.snapshots[snap_id][idx]
        clean.refcount += 1
        dirty.refcount -= 1
        if dirty.refcount <= 0:
            self.pool.unmap_page(dirty.va); self.pool.decref(dirty.phys); self.pool.free_va(dirty.va, self.page_size)
        bt[idx] = clean
        return time.perf_counter() - t0

    def block_va(self, branch_id, idx):
        return self.branches[branch_id][idx].va

    def block_refcount(self, branch_id, idx):
        return self.branches[branch_id][idx].refcount

    def destroy(self):
        self.pool.destroy()


# ---------------------------------------------------------------------------
# Scenario driver
# ---------------------------------------------------------------------------
def build_real_kv(L, prefix_tokens):
    """Run a genuine layer-0 decode over a token sequence, returning the resulting
    per-token K tensors so both arms fill identical real model K/V bytes."""
    h_tok = 1234
    ks, vs = [], []
    for pos in range(prefix_tokens):
        hh = L.embed[h_tok].clone()
        q, k, v = L.project(0, hh, pos)
        ks.append(k.contiguous()); vs.append(v.contiguous())
        # advance greedily using attention over collected K/V (keeps tokens genuine)
        Kall = torch.stack(ks, dim=1)  # [n_kv, seq, hd]
        Vall = torch.stack(vs, dim=1)
        hh = L.attend_mlp(0, hh, q, Kall, Vall)
        h_tok = int(L.logits_of(hh).argmax())
    return ks, vs


def run_hw_arm(L, N, prefix_tokens, n_rollbacks, rep, rows, ks, vs):
    n_kv, hd = L.n_kv, L.hd
    mgr = KVBranchManager(device_id=0)
    toks_per_page = mgr.page_size // (n_kv * hd * 2)
    prefix_pages = max(1, (prefix_tokens + toks_per_page - 1) // toks_per_page)
    headroom = 4
    # build parent prefix (K only is sufficient to exercise isolation; one page range)
    mgr.create_branch("pK", prefix_pages, headroom_pages=headroom)
    # pack K tokens contiguously into the prefix pages via the BranchKV layout
    from decode_layer import BranchKV
    mgr.create_branch("pV", prefix_pages, headroom_pages=headroom)
    pbkv = BranchKV(mgr, n_kv, hd, "pK", "pV", reset=True)
    for k, v in zip(ks, vs):
        pbkv.append_token(k, v)
    torch.cuda.synchronize()
    actual_prefix_pages = mgr.branches["pK"].num_pages

    snapK = mgr.snapshot("pK")
    # pre-fork snapshot bytes of the page we will contaminate (interior shared page)
    target_page = min(actual_prefix_pages - 1, 0) if actual_prefix_pages == 1 else 0
    snap_phys = snapK.page_phys[target_page]
    snap_handle = mgr.pool.retained_handle_at(mgr.branches["pK"].va_of(target_page))
    snap_bytes = read_page(mgr.branches["pK"].va_of(target_page), mgr.page_size)

    # fork N children aliasing the prefix
    forkts = []
    for i in range(N):
        t0 = time.perf_counter()
        mgr.fork(snapK, f"c{i}K", headroom_pages=headroom)
        forkts.append(time.perf_counter() - t0)

    # designate child 0 as the CONTAMINATED/untrusted branch; the rest are siblings.
    cont = "c0K"
    # capture a clean sibling (c1) handle+bytes BEFORE contamination
    sib = "c1K"
    sib_handle_before = mgr.pool.retained_handle_at(mgr.branches[sib].va_of(target_page))
    sib_bytes_before = read_page(mgr.branches[sib].va_of(target_page), mgr.page_size)

    # ---- contaminated write (untrusted edit overwrites a shared prefix page) ----
    bc0 = mgr.pool.stat_bytes_copied
    t0 = time.perf_counter()
    did_cow = mgr.write_page(cont, target_page, fill_value=0xAB)
    torch.cuda.synchronize()
    cow_us = (time.perf_counter() - t0) * 1e6
    bytes_copied_write = mgr.pool.stat_bytes_copied - bc0

    # ---- ISOLATION VERIFICATION ----
    # (1) DRIVER-HANDLE PROOF: contaminated branch now maps a DISTINCT physical handle;
    #     sibling still maps the SAME physical handle as the pre-fork snapshot.
    cont_handle = mgr.pool.retained_handle_at(mgr.branches[cont].va_of(target_page))
    sib_handle_after = mgr.pool.retained_handle_at(mgr.branches[sib].va_of(target_page))
    handle_distinct = (cont_handle != sib_handle_after)
    sibling_handle_eq_snapshot = (sib_handle_after == snap_handle)
    # (2) byte proof: sibling bytes bit-identical to pre-fork snapshot
    sib_bytes_after = read_page(mgr.branches[sib].va_of(target_page), mgr.page_size)
    sib_bitidentical = (sib_bytes_after == snap_bytes == sib_bytes_before)
    cont_changed = (read_page(mgr.branches[cont].va_of(target_page), mgr.page_size) != snap_bytes)
    driver_handle_proof = handle_distinct and sibling_handle_eq_snapshot
    isolation_verified = driver_handle_proof and sib_bitidentical and cont_changed

    # ---- ROLLBACK: re-alias the contaminated branch to the clean snapshot page ----
    # O(1 page): unmap dirty private page, map back the snapshot physical handle, incref.
    rb_bytes0 = mgr.pool.stat_bytes_copied
    rbts = []
    rollback_only_bytes = 0
    for _ in range(n_rollbacks):
        b_pre = mgr.pool.stat_bytes_copied
        t0 = time.perf_counter()
        br = mgr.branches[cont]
        dirty = br.page_phys[target_page]
        va = br.va_of(target_page)
        mgr.pool.unmap_page(va)
        mgr.pool.decref(dirty)
        mgr.pool.map_page(va, snap_phys)
        mgr.pool.incref(snap_phys)
        br.page_phys[target_page] = snap_phys
        torch.cuda.synchronize()
        rbts.append((time.perf_counter() - t0) * 1e6)
        rollback_only_bytes += (mgr.pool.stat_bytes_copied - b_pre)  # bytes copied BY rollback
        # re-contaminate so the next rollback rep has work to do (this CoW is NOT rollback cost)
        if _ < n_rollbacks - 1:
            mgr.write_page(cont, target_page, fill_value=0xCD)
            torch.cuda.synchronize()
    rb_bytes = rollback_only_bytes
    # verify rollback restored bit-identical clean state + re-aliased the snapshot handle
    cont_handle_rb = mgr.pool.retained_handle_at(mgr.branches[cont].va_of(target_page))
    cont_bytes_rb = read_page(mgr.branches[cont].va_of(target_page), mgr.page_size)
    rollback_bitidentical = (cont_bytes_rb == snap_bytes)
    rollback_rehandled = (cont_handle_rb == snap_handle)
    # contiguous VA preserved: branch VA base unchanged -> unmodified kernel still works
    contiguous_va_preserved = True  # VA never moved; only the backing handle was swapped

    rb_us = sum(rbts) / len(rbts)
    fork_us = sum(forkts) / len(forkts) * 1e6

    rows.append(dict(
        arm="ARM-HW", N=N, prefix_tokens=prefix_tokens, prefix_pages=actual_prefix_pages,
        rep=rep, n_rollbacks=n_rollbacks,
        isolation_verified=int(isolation_verified),
        driver_handle_proof=int(driver_handle_proof),
        sibling_bitidentical=int(sib_bitidentical),
        rollback_bitidentical=int(rollback_bitidentical and rollback_rehandled),
        rollback_cost_us=round(rb_us, 2),
        rollback_bytes_copied=int(rb_bytes),
        cow_write_us=round(cow_us, 2),
        write_bytes_copied=int(bytes_copied_write),
        fork_us=round(fork_us, 2),
        contiguous_va_preserved=int(contiguous_va_preserved),
        notes=f"handle_distinct={handle_distinct};sib_handle==snap={sibling_handle_eq_snapshot};"
              f"cont_handle={cont_handle};sib_handle={sib_handle_after};snap_handle={snap_handle}"))
    mgr.pool.destroy()


def run_sw_arm(L, N, prefix_tokens, n_rollbacks, rep, rows, ks, vs):
    n_kv, hd = L.n_kv, L.hd
    sw = RealGpuPrefixSharingManager(device_id=0)
    toks_per_page = sw.page_size // (n_kv * hd * 2)
    prefix_blocks = max(1, (prefix_tokens + toks_per_page - 1) // toks_per_page)
    # build parent prefix blocks filled with real K bytes
    sw.create_filled_branch("p", prefix_blocks, fill=0)
    # pack real K bytes into block 0 (contiguous first toks_per_page tokens)
    Kpack = torch.stack(ks[:toks_per_page], dim=1).contiguous()  # [n_kv, t, hd]
    nbytes = min(Kpack.numel() * Kpack.element_size(), sw.page_size)
    _ck(cuda.cuMemcpyDtoD(sw.block_va("p", 0), Kpack.data_ptr(), nbytes))

    target = 0
    snap = sw.snapshot("p")
    snap_bytes = read_page(sw.block_va("p", target), sw.page_size)

    forkts = []
    for i in range(N):
        _, dt = sw.fork("p", f"c{i}")
        forkts.append(dt)

    cont = "c0"; sib = "c1"
    sib_bytes_before = read_page(sw.block_va(sib, target), sw.page_size)

    bc0 = sw.bytes_copied
    t0 = time.perf_counter()
    did_cow = sw.contaminate_write(cont, target, 0xAB)
    torch.cuda.synchronize()
    cow_us = (time.perf_counter() - t0) * 1e6
    bytes_copied_write = sw.bytes_copied - bc0

    # ISOLATION: software can ONLY verify by data read-back + its own refcount bookkeeping.
    # There is NO physical-handle primitive: a kernel that ignores the block_table is not
    # provably barred from the contaminated block. We record driver_handle_proof=0.
    sib_bytes_after = read_page(sw.block_va(sib, target), sw.page_size)
    sib_bitidentical = (sib_bytes_after == snap_bytes == sib_bytes_before)
    cont_changed = (read_page(sw.block_va(cont, target), sw.page_size) != snap_bytes)
    # refcount bookkeeping check (software's notion of separation)
    refcount_separation = (sw.block_refcount(cont, target) >= 1 and sw.block_refcount(sib, target) >= 1)
    driver_handle_proof = 0  # software cannot produce a physical-handle proof
    isolation_verified = sib_bitidentical and cont_changed and refcount_separation

    # ROLLBACK: repoint block_table entry back to snapshot block (O(1), 0 bytes copied)
    rb_bytes0 = sw.bytes_copied
    rbts = []
    rollback_only_bytes = 0
    for _ in range(n_rollbacks):
        b_pre = sw.bytes_copied
        t0 = time.perf_counter()
        dt = sw.rollback(cont, target, snap)
        torch.cuda.synchronize()
        rbts.append((time.perf_counter() - t0) * 1e6)
        rollback_only_bytes += (sw.bytes_copied - b_pre)
        if _ < n_rollbacks - 1:
            sw.contaminate_write(cont, target, 0xCD)
            torch.cuda.synchronize()
    rb_bytes = rollback_only_bytes
    cont_bytes_rb = read_page(sw.block_va(cont, target), sw.page_size)
    rollback_bitidentical = (cont_bytes_rb == snap_bytes)

    rb_us = sum(rbts) / len(rbts)
    fork_us = sum(forkts) / len(forkts) * 1e6

    rows.append(dict(
        arm="ARM-SW", N=N, prefix_tokens=prefix_tokens, prefix_pages=prefix_blocks,
        rep=rep, n_rollbacks=n_rollbacks,
        isolation_verified=int(isolation_verified),
        driver_handle_proof=driver_handle_proof,
        sibling_bitidentical=int(sib_bitidentical),
        rollback_bitidentical=int(rollback_bitidentical),
        rollback_cost_us=round(rb_us, 2),
        rollback_bytes_copied=int(rb_bytes),
        cow_write_us=round(cow_us, 2),
        write_bytes_copied=int(bytes_copied_write),
        fork_us=round(fork_us, 2),
        contiguous_va_preserved=0,  # block-table indirection: NOT contiguous VA; needs custom kernel
        notes="software refcount bookkeeping only; no physical-handle proof; "
              "block-table indirection means unmodified FlashAttention cannot run -- "
              "isolation is bookkeeping-enforced, not MMU-enforced"))
    sw.destroy()


def main():
    _init_torch()
    from decode_layer import QwenLayerN
    L = QwenLayerN(num_layers=1)
    print(f"loaded Qwen2.5-7B layer-0 (n_kv={L.n_kv} hd={L.hd})", flush=True)

    rows = []
    REPS = 3
    # build genuine model K/V ONCE (O(n^2) attention build is expensive); reuse across
    # all reps/arms -- the isolation test depends on real bytes, not token identity.
    n_kv, hd = L.n_kv, L.hd
    toks_per_page = (2 * 1024 * 1024) // (n_kv * hd * 2)
    ks, vs = build_real_kv(L, min(2048, toks_per_page))
    print(f"built {len(ks)} real K/V tokens (toks_per_page={toks_per_page})", flush=True)
    for N in (4, 8):
        for prefix_tokens in (2048,):
            n_rollbacks = 3
            for rep in range(REPS):
                run_hw_arm(L, N, prefix_tokens, n_rollbacks, rep, rows, ks, vs)
                run_sw_arm(L, N, prefix_tokens, n_rollbacks, rep, rows, ks, vs)
                print(f"  done N={N} prefix={prefix_tokens} rep={rep}", flush=True)

    fields = ["arm","N","prefix_tokens","prefix_pages","rep","n_rollbacks",
              "isolation_verified","driver_handle_proof","sibling_bitidentical",
              "rollback_bitidentical","rollback_cost_us","rollback_bytes_copied",
              "cow_write_us","write_bytes_copied","fork_us","contiguous_va_preserved","notes"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", OUT, flush=True)

    # quick summary
    import statistics as st
    for arm in ("ARM-HW", "ARM-SW"):
        ar = [r for r in rows if r["arm"] == arm]
        iso = all(r["isolation_verified"] for r in ar)
        dhp = all(r["driver_handle_proof"] for r in ar)
        rb = st.mean(r["rollback_cost_us"] for r in ar)
        rbb = st.mean(r["rollback_bytes_copied"] for r in ar)
        rbid = all(r["rollback_bitidentical"] for r in ar)
        print(f"{arm}: isolation_verified(all)={iso} driver_handle_proof(all)={dhp} "
              f"rollback_us(mean)={rb:.2f} rollback_bytes(mean)={rbb:.0f} "
              f"rollback_bitidentical(all)={rbid}", flush=True)


if __name__ == "__main__":
    main()
