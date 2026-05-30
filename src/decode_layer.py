"""
decode_layer.py  (P0-A) — A real single-transformer-layer autoregressive decode loop
whose KV cache is physically backed by the CoW VMM pages of KVBranchManager.

We load ONE real transformer layer (layer 0) of Qwen2.5-7B-Instruct (open weights:
28 q-heads, 4 kv-heads, head_dim 128, RoPE theta 1e6, RMSNorm) plus the embedding table
and final norm + lm_head, and run a genuine autoregressive decode:

  for each step:
    h = embed(token)                         # real embedding
    h = rmsnorm(h, input_layernorm_w)        # real RMSNorm
    q,k,v = proj(h)                          # real q/k/v projections (+ bias)
    q,k = rope(q,k, pos)                     # real rotary embedding
    APPEND k,v to the branch's CoW-backed KV pages (vmm)     <-- the contribution
    o = sdpa(q, K_all, V_all)                # attention over ALL KV in the branch
    h = h + o_proj(o)                        # residual
    logits = lm_head(norm(h))                # real vocab logits
    next_token = argmax(logits)              # greedy

This is NOT the full 28-layer model (R1-D2: one layer is sufficient to prove the CoW
pages support REAL attention compute with REAL weights). Token sequences are therefore
not natural language; we report tokens/sec, peak HBM, and bytes copied — the systems
quantities — NOT generation quality. See LIMITATIONS.md.

KV physical layout (per branch, layer 0 only):
  K tensor: [n_kv_heads, seq, head_dim] fp16  -> packed contiguously in VMM pages
  V tensor: [n_kv_heads, seq, head_dim] fp16  -> separate VMM page range
Each VMM page (2 MiB) holds 2MiB / (n_kv_heads*head_dim*2) tokens for K (and V).
"""
import sys, os, json, glob, time
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn.functional as F
from cuda import cuda
from vmm_pool import VMMPool, _ck
from kv_branch_manager import KVBranchManager


def _snap_dir():
    return glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*"))[0]


def tensor_from_va(va, numel, dtype, shape):
    """Zero-copy torch view over a VMM virtual address (same trick as Metric 3)."""
    elem = torch.tensor([], dtype=dtype).element_size()
    nbytes = numel * elem
    class _Arr:
        def __init__(self, ptr, nbytes):
            self.__cuda_array_interface__ = dict(
                shape=(nbytes,), typestr='|u1', data=(ptr, False), version=3, strides=None)
    u8 = torch.as_tensor(_Arr(int(va), nbytes), device='cuda')
    return u8.view(dtype).view(*shape)


class QwenLayer0:
    """Real Qwen2.5-7B layer-0 weights + embedding + lm_head, fp16 on cuda."""
    def __init__(self, device="cuda", dtype=torch.float16):
        from safetensors import safe_open
        snap = _snap_dir()
        self.cfg = json.load(open(os.path.join(snap, "config.json")))
        self.n_q = self.cfg["num_attention_heads"]        # 28
        self.n_kv = self.cfg["num_key_value_heads"]        # 4
        self.hidden = self.cfg["hidden_size"]              # 3584
        self.hd = self.hidden // self.n_q                  # 128
        self.theta = self.cfg.get("rope_theta", 1e6)
        self.eps = self.cfg.get("rms_norm_eps", 1e-6)
        self.dtype = dtype; self.device = device
        shard = os.path.join(snap, "model-00001-of-00004.safetensors")
        norm_shard = os.path.join(snap, "model-00004-of-00004.safetensors")
        w = {}
        with safe_open(shard, framework="pt", device=device) as f:
            for k in f.keys():
                if "layers.0." in k or k == "model.embed_tokens.weight":
                    w[k] = f.get_tensor(k).to(dtype)
        with safe_open(norm_shard, framework="pt", device=device) as f:
            w["model.norm.weight"] = f.get_tensor("model.norm.weight").to(dtype)
        self.embed = w["model.embed_tokens.weight"]                 # tied lm_head
        p = "model.layers.0."
        self.ln1 = w[p+"input_layernorm.weight"]
        self.wq = w[p+"self_attn.q_proj.weight"]; self.bq = w[p+"self_attn.q_proj.bias"]
        self.wk = w[p+"self_attn.k_proj.weight"]; self.bk = w[p+"self_attn.k_proj.bias"]
        self.wv = w[p+"self_attn.v_proj.weight"]; self.bv = w[p+"self_attn.v_proj.bias"]
        self.wo = w[p+"self_attn.o_proj.weight"]
        self.norm_f = w["model.norm.weight"]
        # precompute RoPE inv_freq
        self.inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, device=device).float() / self.hd))

    def _rmsnorm(self, x, w):
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * w.float()).to(self.dtype)

    def _rope(self, x, pos):
        # x: [heads, hd]; pos: scalar position
        ang = pos * self.inv_freq                      # [hd/2]
        cos = torch.cos(ang); sin = torch.sin(ang)
        cos = torch.cat([cos, cos]); sin = torch.cat([sin, sin])  # [hd]
        x1 = x[..., : self.hd // 2]; x2 = x[..., self.hd // 2:]
        rot = torch.cat([-x2, x1], dim=-1)
        return (x.float() * cos + rot.float() * sin).to(self.dtype)

    @torch.no_grad()
    def project(self, token_id, pos):
        """One token forward up to (and including) q/k/v projection + RoPE.
        Returns (h_resid, q[n_q,hd], k[n_kv,hd], v[n_kv,hd])."""
        h = self.embed[token_id]                       # [hidden]
        x = self._rmsnorm(h, self.ln1)
        q = (x @ self.wq.T + self.bq).view(self.n_q, self.hd)
        k = (x @ self.wk.T + self.bk).view(self.n_kv, self.hd)
        v = (x @ self.wv.T + self.bv).view(self.n_kv, self.hd)
        q = self._rope(q, pos); k = self._rope(k, pos)
        return h, q, k, v

    @torch.no_grad()
    def attend_and_logits(self, h, q, K_all, V_all):
        """q:[n_q,hd]; K_all/V_all:[n_kv, seq, hd]. GQA: repeat kv heads. Returns logits."""
        rep = self.n_q // self.n_kv
        K = K_all.repeat_interleave(rep, dim=0)        # [n_q, seq, hd]
        V = V_all.repeat_interleave(rep, dim=0)
        Q = q.unsqueeze(1)                             # [n_q, 1, hd]
        o = F.scaled_dot_product_attention(Q, K, V)    # [n_q, 1, hd]
        o = o.reshape(self.n_q * self.hd)
        h = h + (o @ self.wo.T)
        logits = self._rmsnorm(h, self.norm_f) @ self.embed.T   # tied lm_head
        return logits


class BranchKV:
    """K/V for ONE branch's layer-0, physically backed by CoW VMM pages.
    K and V each live in their own page range inside the shared KVBranchManager pool.
    append_token() writes one token's K and V into the current tail page, growing via
    append_page() when a page fills. tokens_per_page = page_size / (n_kv*hd*2)."""
    def __init__(self, mgr: KVBranchManager, n_kv, hd, kid, vid, reset=True):
        self.mgr = mgr; self.n_kv = n_kv; self.hd = hd
        self.bytes_per_tok = n_kv * hd * 2  # fp16
        self.toks_per_page = mgr.page_size // self.bytes_per_tok
        self.kid = kid; self.vid = vid     # branch ids inside mgr for K and V ranges
        self.seq = 0
        if reset:
            # fresh branches: start with 0 active pages; first append maps slot 0.
            self.mgr.branches[kid].num_pages = 0
            self.mgr.branches[vid].num_pages = 0

    def _kv_views(self, branch_id):
        """Return a [n_kv, capacity_tokens, hd] fp16 view over all mapped pages of a
        branch. Pages are contiguous in VA (cuMemAddressReserve), so one view spans them."""
        br = self.mgr.branches[branch_id]
        ntok = br.num_pages * self.toks_per_page
        return tensor_from_va(br.va_base, self.n_kv * ntok * self.hd, torch.float16,
                              (self.n_kv, ntok, self.hd))

    def k_view(self): return self._kv_views(self.kid)[:, :self.seq, :]
    def v_view(self): return self._kv_views(self.vid)[:, :self.seq, :]

    def _ensure_private(self, branch_id, page_index):
        """If the target tail page is SHARED (refcount>1, e.g. a forked prefix page),
        trigger a real CoW (private copy + remap) BEFORE writing — preserves correctness
        so a child's decode never corrupts the shared parent prefix. No-op if private."""
        br = self.mgr.branches[branch_id]
        pg = br.page_phys[page_index]
        if pg is not None and pg.refcount > 1:
            self.mgr._cow(br, page_index, pg)   # real GPU MMU CoW remap + D2D copy

    def append_token(self, k, v):
        """Write one token's k,v (each [n_kv, hd]) at position self.seq, growing pages.
        CoW-correct: if the token lands in a SHARED prefix page (forked child), the page
        is privatized first."""
        page_index = self.seq // self.toks_per_page
        need_page = page_index + 1
        # grow tail with fresh PRIVATE pages if we ran off the end
        while self.mgr.branches[self.kid].num_pages < need_page:
            self.mgr.append_page(self.kid)
            self.mgr.append_page(self.vid)
        # if writing into an existing (possibly shared/forked) page, CoW it private first
        self._ensure_private(self.kid, page_index)
        self._ensure_private(self.vid, page_index)
        Kv = self._kv_views(self.kid); Vv = self._kv_views(self.vid)
        Kv[:, self.seq, :] = k
        Vv[:, self.seq, :] = v
        self.seq += 1


# =====================================================================================
# P0-1 (R2): MULTI-LAYER decode. N=4 FULL transformer blocks (attention + SwiGLU MLP),
# each with its OWN per-branch CoW-backed K/V pages. This proves the CoW mechanism
# composes across a real layer stack and produces NON-DEGENERATE tokens (real residual
# + MLP language-modeling flow), answering reviewer C1 ("single layer is a toy").
# =====================================================================================
class QwenLayerN:
    """First N full transformer blocks of Qwen2.5-7B-Instruct + embed + final norm + tied
    lm_head, fp16 on cuda. Each block: input_layernorm -> GQA attention (per-layer KV) ->
    residual -> post_attention_layernorm -> SwiGLU MLP -> residual."""
    def __init__(self, num_layers=4, device="cuda", dtype=torch.float16):
        from safetensors import safe_open
        snap = _snap_dir()
        self.cfg = json.load(open(os.path.join(snap, "config.json")))
        self.N = num_layers
        self.n_q = self.cfg["num_attention_heads"]
        self.n_kv = self.cfg["num_key_value_heads"]
        self.hidden = self.cfg["hidden_size"]
        self.hd = self.hidden // self.n_q
        self.inter = self.cfg["intermediate_size"]
        self.theta = self.cfg.get("rope_theta", 1e6)
        self.eps = self.cfg.get("rms_norm_eps", 1e-6)
        self.dtype = dtype; self.device = device
        shard = os.path.join(snap, "model-00001-of-00004.safetensors")
        norm_shard = os.path.join(snap, "model-00004-of-00004.safetensors")
        want_layers = set(range(num_layers))
        w = {}
        with safe_open(shard, framework="pt", device=device) as f:
            for k in f.keys():
                if k == "model.embed_tokens.weight":
                    w[k] = f.get_tensor(k).to(dtype)
                elif k.startswith("model.layers."):
                    li = int(k.split(".")[2])
                    if li in want_layers:
                        w[k] = f.get_tensor(k).to(dtype)
        with safe_open(norm_shard, framework="pt", device=device) as f:
            w["model.norm.weight"] = f.get_tensor("model.norm.weight").to(dtype)
        self.embed = w["model.embed_tokens.weight"]
        self.norm_f = w["model.norm.weight"]
        self.L = []   # per-layer weight dict
        for li in range(num_layers):
            p = f"model.layers.{li}."
            self.L.append(dict(
                ln1=w[p+"input_layernorm.weight"],
                ln2=w[p+"post_attention_layernorm.weight"],
                wq=w[p+"self_attn.q_proj.weight"], bq=w[p+"self_attn.q_proj.bias"],
                wk=w[p+"self_attn.k_proj.weight"], bk=w[p+"self_attn.k_proj.bias"],
                wv=w[p+"self_attn.v_proj.weight"], bv=w[p+"self_attn.v_proj.bias"],
                wo=w[p+"self_attn.o_proj.weight"],
                gate=w[p+"mlp.gate_proj.weight"], up=w[p+"mlp.up_proj.weight"],
                down=w[p+"mlp.down_proj.weight"]))
        self.inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, device=device).float() / self.hd))

    def _rmsnorm(self, x, weight):
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * weight.float()).to(self.dtype)

    def _rope(self, x, pos):
        ang = pos * self.inv_freq
        cos = torch.cos(ang); sin = torch.sin(ang)
        cos = torch.cat([cos, cos]); sin = torch.cat([sin, sin])
        x1 = x[..., : self.hd // 2]; x2 = x[..., self.hd // 2:]
        rot = torch.cat([-x2, x1], dim=-1)
        return (x.float() * cos + rot.float() * sin).to(self.dtype)

    @torch.no_grad()
    def project(self, li, h, pos):
        """Layer li: input_layernorm + q/k/v projection + RoPE. Returns (q,k,v) for this
        token. h is the residual-stream hidden state ENTERING layer li."""
        Lw = self.L[li]
        x = self._rmsnorm(h, Lw["ln1"])
        q = (x @ Lw["wq"].T + Lw["bq"]).view(self.n_q, self.hd)
        k = (x @ Lw["wk"].T + Lw["bk"]).view(self.n_kv, self.hd)
        v = (x @ Lw["wv"].T + Lw["bv"]).view(self.n_kv, self.hd)
        q = self._rope(q, pos); k = self._rope(k, pos)
        return q, k, v

    @torch.no_grad()
    def attend_mlp(self, li, h, q, K_all, V_all):
        """Layer li: GQA attention over this layer's KV + residual + SwiGLU MLP + residual.
        Returns the hidden state EXITING layer li (input to layer li+1)."""
        Lw = self.L[li]
        rep = self.n_q // self.n_kv
        K = K_all.repeat_interleave(rep, dim=0)
        V = V_all.repeat_interleave(rep, dim=0)
        Q = q.unsqueeze(1)
        o = F.scaled_dot_product_attention(Q, K, V).reshape(self.n_q * self.hd)
        h = h + (o @ Lw["wo"].T)                          # attention residual
        # SwiGLU MLP
        xn = self._rmsnorm(h, Lw["ln2"])
        gate = F.silu((xn @ Lw["gate"].T).float()).to(self.dtype)
        up = xn @ Lw["up"].T
        mlp = (gate * up) @ Lw["down"].T
        h = h + mlp                                        # MLP residual
        return h

    @torch.no_grad()
    def logits_of(self, h):
        return self._rmsnorm(h, self.norm_f) @ self.embed.T


class MultiLayerBranchKV:
    """Per-branch, per-LAYER K/V backed by CoW VMM pages. Holds one BranchKV per layer.
    Each layer gets its own pair of VMM branches (K range + V range) inside the shared
    KVBranchManager, so a forked child aliases ALL layers' prefix pages with zero copy."""
    def __init__(self, mgr, n_layers, n_kv, hd, prefix_id, reset=True):
        self.mgr = mgr; self.n_layers = n_layers
        self.layers = []
        for li in range(n_layers):
            kid, vid = f"{prefix_id}_L{li}K", f"{prefix_id}_L{li}V"
            self.layers.append(BranchKV(mgr, n_kv, hd, kid, vid, reset=reset))

    @property
    def seq(self):
        return self.layers[0].seq

    def set_seq(self, s):
        for bkv in self.layers:
            bkv.seq = s
