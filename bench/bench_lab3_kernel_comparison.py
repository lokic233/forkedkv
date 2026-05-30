"""
Lab 3 — Kernel-Throughput Comparison: Contiguous-VA SDPA  vs  Paged-Attention.

Question: when KV is contiguously addressable (the ForkedKV case — VMM gives one VA
range per branch), how much faster is the standard attention kernel than a paged
kernel that has to indirect through a block table (the vLLM-APC software baseline)?

This isolates the *one* claim of the paper: "kernel-transparent VA matters for
throughput because standard kernels are faster than block-table kernels at equal
batch size."

Methodology
-----------
Workload: prefill done; we measure DECODE attention (Q is one token, K/V is the
full sequence) at varying (batch, seqlen). This is the regime where attention
kernel cost dominates per-token decode latency in agent serving.

Two implementations under test:

(A) SDPA over contiguous KV    [represents ForkedKV / standard kernels]
    K, V shape:  [B, H, S, D]  -- one tensor, contiguous in memory
    Q shape:     [B, H, 1, D]
    Run: F.scaled_dot_product_attention -> cuDNN flash-attention.

(B) Triton paged attention     [represents vLLM-APC / block-table kernels]
    K, V shape:  [num_blocks, H, BLOCK, D]  -- block pool
    block_table: [B, max_blocks_per_seq] int32 mapping logical block -> physical
    Q shape:     [B, H, 1, D]
    Run: a Triton kernel that loads K/V blocks via block_table indirection,
    computes attention, online softmax. Mirrors vLLM's paged_attention_v1.
    NOTE: this is a *minimum-viable* paged kernel — it has the right asymptotic
    work (block-by-block load + online softmax) but does not implement vLLM's full
    optimizations (split-K, multi-query). We label it "paged_minimal" in the
    output and discuss in WRITEUP.

Output: data/lab3_kernel_comparison.csv columns:
    batch, seqlen, method, rep, ms, tokens_per_s
"""
import os, sys, csv, statistics, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# Match Qwen2.5-7B-ish single-layer dimensions
HEADS = 32
HEAD_DIM = 128
DTYPE = torch.float16

BLOCK_TOKENS = 16  # vLLM default block size

# Decode workload: Q is one token; KV grows with seqlen
BATCHES = [1, 4, 16, 32, 64]
SEQLENS = [512, 2048, 8192]

WARMUP = 30
REPS   = 100

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "lab3_kernel_comparison.csv")


# ----------------- Triton paged-attention kernel (minimum viable) ----------------- #
@triton.jit
def _paged_attn_kernel(
    Q_ptr, Kc_ptr, Vc_ptr, Out_ptr,
    BT_ptr, ctx_lens_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kn, stride_kh, stride_kt, stride_kd,
    stride_vn, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_od,
    stride_btb, stride_btn,
    BLOCK: tl.constexpr,        # tokens per block
    HEAD_DIM: tl.constexpr,
    MAX_NUM_BLOCKS: tl.constexpr,
):
    """One program per (batch, head).  Online softmax over up to MAX_NUM_BLOCKS blocks."""
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Load Q for this (b, h)
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(Q_ptr + b*stride_qb + h*stride_qh + offs_d*stride_qd).to(tl.float32)

    ctx = tl.load(ctx_lens_ptr + b)

    m_i = -float('inf')
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    offs_t = tl.arange(0, BLOCK)
    for blk_idx in range(0, MAX_NUM_BLOCKS):
        blk_start = blk_idx * BLOCK
        # mask out blocks past ctx
        in_range = blk_start < ctx
        if in_range:
            phys = tl.load(BT_ptr + b*stride_btb + blk_idx*stride_btn)
            # token mask within this block
            token_pos = blk_start + offs_t
            tmask = token_pos < ctx
            # K shape: [num_blocks, H, BLOCK, D]
            k_off = phys*stride_kn + h*stride_kh + offs_t[:, None]*stride_kt + offs_d[None, :]*stride_kd
            v_off = phys*stride_vn + h*stride_vh + offs_t[:, None]*stride_vt + offs_d[None, :]*stride_vd
            k = tl.load(Kc_ptr + k_off, mask=tmask[:, None], other=0.0).to(tl.float32)
            v = tl.load(Vc_ptr + v_off, mask=tmask[:, None], other=0.0).to(tl.float32)
            # qk: [BLOCK]
            qk = tl.sum(q[None, :] * k, axis=1) * sm_scale
            qk = tl.where(tmask, qk, -float('inf'))
            m_new = tl.maximum(m_i, tl.max(qk, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new)
            l_i = l_i*alpha + tl.sum(p, axis=0)
            acc = acc*alpha + tl.sum(p[:, None]*v, axis=0)
            m_i = m_new

    out = acc / l_i
    tl.store(Out_ptr + b*stride_ob + h*stride_oh + offs_d*stride_od, out.to(Out_ptr.dtype.element_ty))


def paged_attn(Q, Kc, Vc, block_table, ctx_lens):
    """
    Q: [B, H, 1, D]
    Kc, Vc: [num_blocks, H, BLOCK, D]
    block_table: [B, max_num_blocks] int32
    ctx_lens: [B] int32
    """
    B, H, _, D = Q.shape
    max_num_blocks = block_table.shape[1]
    Out = torch.empty(B, H, D, dtype=Q.dtype, device=Q.device)
    sm_scale = 1.0 / (D ** 0.5)
    Q_ = Q.squeeze(2).contiguous()  # [B, H, D]
    grid = (B, H)
    _paged_attn_kernel[grid](
        Q_, Kc, Vc, Out, block_table, ctx_lens, sm_scale,
        Q_.stride(0), Q_.stride(1), Q_.stride(2),
        Kc.stride(0), Kc.stride(1), Kc.stride(2), Kc.stride(3),
        Vc.stride(0), Vc.stride(1), Vc.stride(2), Vc.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2),
        block_table.stride(0), block_table.stride(1),
        BLOCK=BLOCK_TOKENS,
        HEAD_DIM=D,
        MAX_NUM_BLOCKS=max_num_blocks,
        num_warps=4,
    )
    return Out.unsqueeze(2)


# ---------------------- Benchmark harness ---------------------- #
def bench_sdpa(B, S):
    Q = torch.randn(B, HEADS, 1, HEAD_DIM, dtype=DTYPE, device='cuda')
    K = torch.randn(B, HEADS, S, HEAD_DIM, dtype=DTYPE, device='cuda')
    V = torch.randn(B, HEADS, S, HEAD_DIM, dtype=DTYPE, device='cuda')

    for _ in range(WARMUP):
        F.scaled_dot_product_attention(Q, K, V, is_causal=False)
    torch.cuda.synchronize()
    times = []
    for _ in range(REPS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        F.scaled_dot_product_attention(Q, K, V, is_causal=False)
        e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return times


def bench_paged(B, S):
    # Build a block pool that holds B sequences of S tokens, BLOCK_TOKENS per block
    blocks_per_seq = (S + BLOCK_TOKENS - 1) // BLOCK_TOKENS
    total_blocks = B * blocks_per_seq

    Kc = torch.randn(total_blocks, HEADS, BLOCK_TOKENS, HEAD_DIM, dtype=DTYPE, device='cuda')
    Vc = torch.randn(total_blocks, HEADS, BLOCK_TOKENS, HEAD_DIM, dtype=DTYPE, device='cuda')
    # block_table: [B, blocks_per_seq] -> sequential physical blocks for fairness
    bt = torch.arange(total_blocks, dtype=torch.int32, device='cuda').view(B, blocks_per_seq)
    ctx_lens = torch.full((B,), S, dtype=torch.int32, device='cuda')
    Q = torch.randn(B, HEADS, 1, HEAD_DIM, dtype=DTYPE, device='cuda')

    # warm
    for _ in range(WARMUP):
        paged_attn(Q, Kc, Vc, bt, ctx_lens)
    torch.cuda.synchronize()
    times = []
    for _ in range(REPS):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        paged_attn(Q, Kc, Vc, bt, ctx_lens)
        e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return times


def numerical_check(B=2, S=128):
    """Verify our paged kernel matches SDPA to within fp16 tolerance."""
    blocks_per_seq = (S + BLOCK_TOKENS - 1) // BLOCK_TOKENS
    total_blocks = B * blocks_per_seq
    Kc = torch.randn(total_blocks, HEADS, BLOCK_TOKENS, HEAD_DIM, dtype=DTYPE, device='cuda')
    Vc = torch.randn(total_blocks, HEADS, BLOCK_TOKENS, HEAD_DIM, dtype=DTYPE, device='cuda')
    bt = torch.arange(total_blocks, dtype=torch.int32, device='cuda').view(B, blocks_per_seq)
    ctx_lens = torch.full((B,), S, dtype=torch.int32, device='cuda')
    Q = torch.randn(B, HEADS, 1, HEAD_DIM, dtype=DTYPE, device='cuda')

    # build the equivalent contiguous view by concatenating each batch's blocks
    K_contig = Kc.view(B, blocks_per_seq, HEADS, BLOCK_TOKENS, HEAD_DIM)\
                 .permute(0,2,1,3,4).reshape(B, HEADS, blocks_per_seq*BLOCK_TOKENS, HEAD_DIM)[:, :, :S, :].contiguous()
    V_contig = Vc.view(B, blocks_per_seq, HEADS, BLOCK_TOKENS, HEAD_DIM)\
                 .permute(0,2,1,3,4).reshape(B, HEADS, blocks_per_seq*BLOCK_TOKENS, HEAD_DIM)[:, :, :S, :].contiguous()

    out_paged = paged_attn(Q, Kc, Vc, bt, ctx_lens).squeeze(2)
    out_ref = F.scaled_dot_product_attention(Q, K_contig, V_contig).squeeze(2)
    diff = (out_paged - out_ref).abs().max().item()
    rel = diff / (out_ref.abs().max().item() + 1e-6)
    print(f"  paged-vs-sdpa numerical check: max_abs={diff:.4f} rel={rel:.4f}")
    if rel > 0.02:
        print("  WARNING: paged kernel may have a bug (rel > 2%)")
    return rel


def main():
    print(f"H100  HEADS={HEADS} HEAD_DIM={HEAD_DIM} BLOCK={BLOCK_TOKENS}  WARMUP={WARMUP} REPS={REPS}")
    print("verifying paged kernel correctness...")
    rel = numerical_check()
    rows = []
    print(f"\n{'B':>3} {'S':>5} {'sdpa_ms':>9} {'paged_ms':>10} {'speedup':>9} {'sdpa_tok/s':>11} {'paged_tok/s':>12}")
    for B in BATCHES:
        for S in SEQLENS:
            tc = bench_sdpa(B, S)
            tp = bench_paged(B, S)
            mc, mp = statistics.median(tc), statistics.median(tp)
            tps_c = B*1000.0/mc; tps_p = B*1000.0/mp
            for r,(a,b) in enumerate(zip(tc, tp)):
                rows.append((B, S, "sdpa_contig", r, a, B*1000.0/a))
                rows.append((B, S, "paged_minimal", r, b, B*1000.0/b))
            print(f"{B:>3} {S:>5} {mc:>9.4f} {mp:>10.4f} {mp/mc:>8.2f}x {tps_c:>11.1f} {tps_p:>12.1f}")
    with open(OUT, "w", newline="") as f:
        w=csv.writer(f); w.writerow(["batch","seqlen","method","rep","ms","tokens_per_s"]); w.writerows(rows)
    print(f"\nwrote {OUT}")

if __name__ == "__main__":
    main()
