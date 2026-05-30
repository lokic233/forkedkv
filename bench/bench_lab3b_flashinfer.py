"""
Lab 3b — PRODUCTION Paged-Attention Baseline: SDPA (contiguous) vs FlashInfer.

Why this exists
---------------
Lab 3 compared SDPA-over-contiguous-KV (the ForkedKV / kernel-transparent VA path)
against a *minimal* hand-written Triton paged kernel. A reviewer correctly noted that
a minimal Triton paged kernel is a weak baseline: it is NOT what production serving
stacks (vLLM, SGLang) actually run. To make the comparison fair we re-run the SAME
decode workload against FlashInfer's BatchDecodeWithPagedKVCacheWrapper — the
production-grade paged-attention kernel used by SGLang and others (split-K,
tensor-core GQA, tuned block loads).

Comparison
----------
(A) SDPA over contiguous KV         -> "SDPA (contiguous, ForkedKV path)"
    Standard F.scaled_dot_product_attention on [B, H_qo, S, D] contiguous tensors.
    Expanded KV heads to H_qo to feed dense SDPA (GQA via repeat_interleave),
    which is what a contiguous-VA kernel sees.

(B) FlashInfer paged decode         -> "production paged baseline (FlashInfer)"
    flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper over a paged KV pool with a
    block table. page_size=16 tokens (FlashInfer-typical). use_tensor_cores=True for
    the GQA group size (28/4=7).

Config (Qwen2.5-7B single-layer decode, GQA):
    num_qo_heads = 28, num_kv_heads = 4, head_dim = 128, page_size = 16, fp16
    batch_sizes = [32, 64], seqlens = [512, 2048, 8192]
    Q = single decode token per sequence.

Metrics: per-call latency (ms, CUDA-event timed) and decode throughput (tokens/s = B / latency).
Output : data/lab3b_flashinfer.csv  columns: batch,seqlen,method,rep,ms,tokens_per_s
Reps   : 100 timed (median reported), 30 warmup.
"""
import os, sys, csv, statistics, time

import torch
import torch.nn.functional as F
import flashinfer

# ---- GQA config (matches task spec; Qwen2.5-7B) ----
NUM_QO_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM     = 128
PAGE_SIZE    = 16
DTYPE        = torch.float16

BATCHES = [32, 64]
SEQLENS = [512, 2048, 8192]

WARMUP = 30
REPS   = 100

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "lab3b_flashinfer.csv")

DEV = "cuda"
SM_SCALE = 1.0 / (HEAD_DIM ** 0.5)


def bench_sdpa(B, S):
    """Contiguous-VA SDPA, GQA expanded to dense MHA (KV repeated to 28 heads).
    This is what a naive standard kernel sees if it does not exploit GQA."""
    Q  = torch.randn(B, NUM_QO_HEADS, 1, HEAD_DIM, dtype=DTYPE, device=DEV)
    Kk = torch.randn(B, NUM_KV_HEADS, S, HEAD_DIM, dtype=DTYPE, device=DEV)
    Vk = torch.randn(B, NUM_KV_HEADS, S, HEAD_DIM, dtype=DTYPE, device=DEV)
    rep = NUM_QO_HEADS // NUM_KV_HEADS
    K = Kk.repeat_interleave(rep, dim=1)
    V = Vk.repeat_interleave(rep, dim=1)

    for _ in range(WARMUP):
        F.scaled_dot_product_attention(Q, K, V, is_causal=False, scale=SM_SCALE)
    torch.cuda.synchronize()
    times = []
    for _ in range(REPS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        F.scaled_dot_product_attention(Q, K, V, is_causal=False, scale=SM_SCALE)
        e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return times


def bench_sdpa_gqa(B, S):
    """Contiguous-VA SDPA using NATIVE GQA (enable_gqa=True) — fairest contiguous
    baseline. KV stays at 4 heads; the kernel broadcasts internally (no KV blow-up).
    This is the strongest 'ForkedKV / kernel-transparent VA' representative."""
    Q  = torch.randn(B, NUM_QO_HEADS, 1, HEAD_DIM, dtype=DTYPE, device=DEV)
    K  = torch.randn(B, NUM_KV_HEADS, S, HEAD_DIM, dtype=DTYPE, device=DEV)
    V  = torch.randn(B, NUM_KV_HEADS, S, HEAD_DIM, dtype=DTYPE, device=DEV)

    for _ in range(WARMUP):
        F.scaled_dot_product_attention(Q, K, V, is_causal=False, scale=SM_SCALE, enable_gqa=True)
    torch.cuda.synchronize()
    times = []
    for _ in range(REPS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        F.scaled_dot_product_attention(Q, K, V, is_causal=False, scale=SM_SCALE, enable_gqa=True)
        e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return times


def build_flashinfer(B, S, wrapper):
    """Build paged KV pool + plan. NHD layout: kv each [max_pages, page_size, H_kv, D]."""
    pages_per_seq = (S + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages   = B * pages_per_seq
    last_page_len = S - (pages_per_seq - 1) * PAGE_SIZE  # tokens used in final page

    k_cache = torch.randn(total_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)
    v_cache = torch.randn(total_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)

    # indptr: [B+1] cumulative page counts; indices: flat page ids (sequential = fair)
    indptr  = torch.arange(0, (B + 1) * pages_per_seq, pages_per_seq, dtype=torch.int32, device=DEV)
    indices = torch.arange(total_pages, dtype=torch.int32, device=DEV)
    last    = torch.full((B,), last_page_len, dtype=torch.int32, device=DEV)

    wrapper.plan(
        indptr, indices, last,
        NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE,
        q_data_type=DTYPE, kv_data_type=DTYPE,
        sm_scale=SM_SCALE,
    )
    q = torch.randn(B, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)
    return q, (k_cache, v_cache)


def bench_flashinfer(B, S, workspace):
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout="NHD", use_tensor_cores=True,
    )
    q, kv = build_flashinfer(B, S, wrapper)

    for _ in range(WARMUP):
        wrapper.run(q, kv)
    torch.cuda.synchronize()
    times = []
    for _ in range(REPS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        wrapper.run(q, kv)
        e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return times


def correctness_check(workspace, B=2, S=128):
    """FlashInfer paged decode vs dense SDPA on the SAME data, within fp16 tol."""
    pages_per_seq = (S + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages   = B * pages_per_seq
    last_page_len = S - (pages_per_seq - 1) * PAGE_SIZE

    # NHD pool
    k_cache = torch.randn(total_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)
    v_cache = torch.randn(total_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)
    indptr  = torch.arange(0, (B + 1) * pages_per_seq, pages_per_seq, dtype=torch.int32, device=DEV)
    indices = torch.arange(total_pages, dtype=torch.int32, device=DEV)
    last    = torch.full((B,), last_page_len, dtype=torch.int32, device=DEV)
    q = torch.randn(B, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=DEV)

    w = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD", use_tensor_cores=True)
    w.plan(indptr, indices, last, NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE,
           q_data_type=DTYPE, kv_data_type=DTYPE, sm_scale=SM_SCALE)
    out_fi = w.run(q, (k_cache, v_cache))  # [B, H_qo, D]

    # Reconstruct contiguous [B, H_kv, S, D] from the pool for reference SDPA
    # pool[seq] = pages indices[indptr[b]:indptr[b+1]] -> [pages_per_seq, PAGE, H_kv, D]
    K_ref = k_cache.view(B, pages_per_seq, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM)\
                   .permute(0, 3, 1, 2, 4).reshape(B, NUM_KV_HEADS, pages_per_seq * PAGE_SIZE, HEAD_DIM)[:, :, :S, :]
    V_ref = v_cache.view(B, pages_per_seq, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM)\
                   .permute(0, 3, 1, 2, 4).reshape(B, NUM_KV_HEADS, pages_per_seq * PAGE_SIZE, HEAD_DIM)[:, :, :S, :]
    rep = NUM_QO_HEADS // NUM_KV_HEADS
    K_ref = K_ref.repeat_interleave(rep, dim=1).contiguous()
    V_ref = V_ref.repeat_interleave(rep, dim=1).contiguous()
    Q_ref = q.unsqueeze(2)  # [B, H_qo, 1, D]
    out_ref = F.scaled_dot_product_attention(Q_ref, K_ref, V_ref, scale=SM_SCALE).squeeze(2)

    diff = (out_fi.float() - out_ref.float()).abs().max().item()
    rel  = diff / (out_ref.float().abs().max().item() + 1e-6)
    print(f"  flashinfer-vs-sdpa numerical check: max_abs={diff:.4f} rel={rel:.4f}", flush=True)
    if rel > 0.02:
        print("  WARNING: flashinfer/SDPA mismatch (rel > 2%) — check layout!", flush=True)
    return rel


def main():
    print(f"H100 GQA  qo_heads={NUM_QO_HEADS} kv_heads={NUM_KV_HEADS} D={HEAD_DIM} "
          f"page={PAGE_SIZE} dtype={DTYPE}  WARMUP={WARMUP} REPS={REPS}", flush=True)
    print(f"torch {torch.__version__}  flashinfer {flashinfer.__version__}", flush=True)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEV)

    print("verifying flashinfer correctness vs SDPA...", flush=True)
    correctness_check(workspace)

    rows = []
    print(f"\n{'B':>3} {'S':>5} {'sdpa_ms':>9} {'sdpaGQA':>9} {'fi_ms':>9} {'fi/sdpaGQA':>10} "
          f"{'sdpaGQA_tok/s':>13} {'fi_tok/s':>11}", flush=True)
    for B in BATCHES:
        for S in SEQLENS:
            tc  = bench_sdpa(B, S)
            tg  = bench_sdpa_gqa(B, S)
            tf  = bench_flashinfer(B, S, workspace)
            mc, mg, mf = statistics.median(tc), statistics.median(tg), statistics.median(tf)
            for r, (a, g, b) in enumerate(zip(tc, tg, tf)):
                rows.append((B, S, "sdpa_contig_dense", r, a, B * 1000.0 / a))
                rows.append((B, S, "sdpa_contig_gqa",   r, g, B * 1000.0 / g))
                rows.append((B, S, "flashinfer_paged",  r, b, B * 1000.0 / b))
            ratio = mf / mg
            print(f"{B:>3} {S:>5} {mc:>9.4f} {mg:>9.4f} {mf:>9.4f} {ratio:>9.2f}x "
                  f"{B*1000.0/mg:>13.1f} {B*1000.0/mf:>11.1f}", flush=True)

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch", "seqlen", "method", "rep", "ms", "tokens_per_s"])
        w.writerows(rows)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
