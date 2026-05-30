"""
Metric 5c (P0-4 R2): CoW-ON-WRITE decode stress — exercise the mechanism's HOT PATH.

Metrics 5/5b are append-dominated: branches grow private tail pages and (5b unaligned)
trigger CoW only on the partially-filled boundary page. They do NOT exercise the headline
scenario the mechanism is built for: a branch OVERWRITING a SHARED PREFIX page mid-decode
— a speculative edit / tree-of-thought ROLLBACK where a child re-writes part of the
context it forked from. That write-after-share is exactly where _cow() must fire correctly.

This benchmark, using ONE real Qwen2.5-7B layer over CoW-backed KV (src/decode_layer.py):
  1. Build a multi-page shared PREFIX, snapshot it.
  2. Fork two children A and B that ALIAS the prefix (zero copy; refcount==2 per page).
  3. Child A deliberately OVERWRITES a SHARED prefix page mid-decode (rewrites the K/V of
     an earlier context position — a tree-of-thought rollback / speculative context edit).
  4. ASSERT: _cow fired (cow_events += exactly 1 per overwritten page); refcount of the
     parent's copy dropped 2->1; ONLY ONE page was copied (2 MiB), not the whole prefix.
  5. ASSERT: child B's prefix page is UNCHANGED (parent context not corrupted); child A's
     overwritten page now holds different bytes (the edit took effect, privately).
  6. Report bytes copied per branch and the per-page diverged-handle proof.

Output: data/metric5c_cow_write.csv
"""
import sys, os, csv, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
from kv_branch_manager import KVBranchManager
from explog import log

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "metric5c_cow_write.csv")
MIB = 1024 * 1024


def _init_torch():
    torch.cuda.init(); torch.cuda.set_device(0)
    _ = torch.zeros(8, device="cuda"); torch.cuda.synchronize()


def main():
    _init_torch()
    from decode_layer import QwenLayerN, BranchKV
    L = QwenLayerN(num_layers=1)
    n_kv, hd = L.n_kv, L.hd
    print(f"loaded Qwen2.5-7B layer-0 (n_kv={n_kv} hd={hd})")

    mgr = KVBranchManager(device_id=0)
    toks_per_page = mgr.page_size // (n_kv * hd * 2)
    # Build a 3-page prefix so there are clearly-shared interior pages to overwrite.
    prefix_tokens = toks_per_page * 3
    headroom = 8
    mgr.create_branch("pK", 1, headroom_pages=headroom)
    mgr.create_branch("pV", 1, headroom_pages=headroom)
    pbkv = BranchKV(mgr, n_kv, hd, "pK", "pV", reset=True)
    # fill prefix with a real per-position forward so K/V are genuine
    h_tok = 1234
    for pos in range(prefix_tokens):
        h, q, k, v = None, None, None, None
        hh = L.embed[h_tok].clone()
        q, k, v = L.project(0, hh, pos)
        pbkv.append_token(k, v)
        Kall = pbkv.k_view(); Vall = pbkv.v_view()
        hh = L.attend_mlp(0, hh, q, Kall, Vall)
        h_tok = int(L.logits_of(hh).argmax())
    prefix_pages = mgr.branches["pK"].num_pages
    print(f"prefix: {prefix_tokens} tokens = {prefix_pages} pages (K) + {prefix_pages} (V)")
    torch.cuda.synchronize()

    snapK = mgr.snapshot("pK"); snapV = mgr.snapshot("pV")
    # fork two children (alias the prefix, zero copy)
    mgr.fork(snapK, "aK", headroom_pages=headroom); mgr.fork(snapV, "aV", headroom_pages=headroom)
    mgr.fork(snapK, "bK", headroom_pages=headroom); mgr.fork(snapV, "bV", headroom_pages=headroom)

    # pick a SHARED interior prefix page to overwrite (page 1 of 3)
    target_page = 1
    aK = mgr.branches["aK"]; bK = mgr.branches["bK"]; pK = mgr.branches["pK"]
    shared_pg = aK.page_phys[target_page]
    rc_before = shared_pg.refcount
    # prove A and B alias the parent on this page BEFORE the write
    alias_ab_before = mgr.shared_handle("aK", "bK", target_page)
    alias_ap_before = mgr.shared_handle("aK", "pK", target_page)
    bytes_copied_before = mgr.pool.stat_bytes_copied
    cow_before = mgr.pool.stat_cow_events

    # capture B's bytes on the target page before A's write (to prove non-corruption)
    b_before = bytes(torch.empty(64, dtype=torch.uint8))
    import ctypes
    from cuda import cuda
    from vmm_pool import _ck
    def read64(va):
        host = (ctypes.c_uint8 * 64)()
        _ck(cuda.cuMemcpyDtoH(host, va, 64)); 
        return bytes(host)
    b_target_before = read64(bK.va_of(target_page))
    a_target_before = read64(aK.va_of(target_page))

    # --- THE HOT PATH: child A overwrites a SHARED prefix page mid-decode ---
    t0 = time.perf_counter()
    did_cow = mgr.write_page("aK", target_page, fill_value=200)  # rewrite K of an earlier ctx position
    torch.cuda.synchronize()
    cow_latency_us = (time.perf_counter() - t0) * 1e6

    bytes_copied = mgr.pool.stat_bytes_copied - bytes_copied_before
    cow_events = mgr.pool.stat_cow_events - cow_before
    rc_after = shared_pg.refcount
    alias_ab_after = mgr.shared_handle("aK", "bK", target_page)
    alias_ap_after = mgr.shared_handle("aK", "pK", target_page)
    a_target_after = read64(aK.va_of(target_page))
    b_target_after = read64(bK.va_of(target_page))

    # only ONE page (2 MiB) copied, not the whole prefix
    pages_copied = bytes_copied // mgr.page_size
    # other prefix pages must still be shared (CoW is per-page)
    page0_still_shared = mgr.shared_handle("aK", "pK", 0)
    page2_still_shared = mgr.shared_handle("aK", "pK", 2)

    print("\n--- CoW-on-write assertions ---")
    print(f"did_cow                 : {did_cow}            (expect True)")
    print(f"cow_events fired        : {cow_events}            (expect 1)")
    print(f"bytes copied            : {bytes_copied/MIB:.2f} MiB = {pages_copied} page(s) (expect 1 page, 2 MiB)")
    print(f"shared pg refcount      : {rc_before} -> {rc_after}   (expect drop by 1: A leaves the alias set)")
    print(f"A aliased parent before : {alias_ap_before}  after: {alias_ap_after}  (expect True -> False)")
    print(f"A aliased B before      : {alias_ab_before}  after: {alias_ab_after}  (expect True -> False)")
    print(f"B target page bytes     : {'UNCHANGED' if b_target_after==b_target_before else 'CHANGED'} (expect UNCHANGED: parent not corrupted)")
    print(f"A target page bytes     : {'CHANGED' if a_target_after!=a_target_before else 'UNCHANGED'} (expect CHANGED: edit took effect)")
    print(f"prefix page0 still shared A==parent: {page0_still_shared} (expect True: CoW is per-page)")
    print(f"prefix page2 still shared A==parent: {page2_still_shared} (expect True)")
    print(f"CoW latency (1 page)    : {cow_latency_us:.1f} us")

    assert did_cow, "write to shared page did not trigger CoW"
    assert cow_events == 1, f"expected exactly 1 CoW event, got {cow_events}"
    assert pages_copied == 1, f"expected exactly 1 page copied, got {pages_copied}"
    assert rc_after == rc_before - 1, f"refcount should drop by 1 (got {rc_before}->{rc_after})"
    assert alias_ap_after is False, "A should no longer alias parent after CoW"
    assert b_target_after == b_target_before, "parent/sibling B page was CORRUPTED by A's write"
    assert a_target_after != a_target_before, "A's overwrite did not take effect"
    assert page0_still_shared and page2_still_shared, "untouched prefix pages should stay shared"
    print("\nALL Metric 5c CoW-ON-WRITE ASSERTIONS PASSED")

    rows = [
        ("did_cow", int(did_cow)),
        ("cow_events", cow_events),
        ("bytes_copied_mib", bytes_copied / MIB),
        ("pages_copied", pages_copied),
        ("prefix_pages", prefix_pages),
        ("refcount_before", rc_before),
        ("refcount_after", rc_after),
        ("A_aliased_parent_before", int(alias_ap_before)),
        ("A_aliased_parent_after", int(alias_ap_after)),
        ("B_page_unchanged", int(b_target_after == b_target_before)),
        ("A_page_changed", int(a_target_after != a_target_before)),
        ("page0_still_shared", int(page0_still_shared)),
        ("page2_still_shared", int(page2_still_shared)),
        ("cow_latency_us", cow_latency_us),
    ]
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["quantity", "value"]); w.writerows(rows)
    print("wrote", OUT)
    log("metric5c_cow_write",
        dict(prefix_pages=prefix_pages, target_page=target_page, n_children=2, model="Qwen2.5-7B-layer0"),
        dict(cow_events=cow_events, bytes_copied=bytes_copied, pages_copied=pages_copied,
             refcount_before=rc_before, refcount_after=rc_after, cow_latency_us=cow_latency_us,
             parent_uncorrupted=int(b_target_after == b_target_before)))

    print(f"\nHEADLINE (5c): a tree-of-thought ROLLBACK that overwrites a shared prefix page "
          f"triggers a per-page CoW copying exactly {pages_copied} page ({bytes_copied/MIB:.0f} MiB), "
          f"NOT the {prefix_pages}-page prefix; sibling/parent context is provably uncorrupted; "
          f"untouched prefix pages stay aliased. CoW latency {cow_latency_us:.0f} us.")
    mgr.pool.destroy()


if __name__ == "__main__":
    main()
