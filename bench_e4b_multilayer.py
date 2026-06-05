"""
E4b — Does an INTERIOR fixed-width edit corrupt suffix K/V at layers >0? (the claude48/codex R6 RED)
Claim under test: "E4 proved equality only at layer 0; at L>0 the suffix attends to the edited
slot, so suffix hidden states (hence suffix K/V) change -> pointer-stable suffix reuse is NOT
exact for an INTERIOR edit. It is exact only for a TERMINAL edit (slot at end, no suffix)."

Method: REAL Qwen2.5-7B first N layers (QwenLayerN). Build a sequence of S tokens with a
fixed-width W-token slot. Full multi-layer prefill caches per-layer K/V for the ORIGINAL.
Then change the slot's token values (SAME width -> RoPE positions unchanged) and full-prefill
the EDITED sequence. Compare per-layer suffix K/V (tokens after the slot) between original and
edited. We do this for two slot positions:
   OFF = S//2   (INTERIOR edit: there IS a suffix after the slot)
   OFF = S - W  (TERMINAL edit: NO suffix after the slot)
If interior suffix K/V diverge at L>=1 but terminal does not, the RED is CONFIRMED and the repair
must be scoped to terminal/append edits.

Reports, per layer, max|dK|,|dV| over SUFFIX tokens only, for interior vs terminal.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import torch
from decode_layer import QwenLayerN

DEV="cuda"
N = int(os.environ.get("E4B_LAYERS","6"))
S = int(os.environ.get("E4B_S","2048"))
W = int(os.environ.get("E4B_W","64"))
torch.manual_seed(0)

m = QwenLayerN(num_layers=N, device=DEV)
V = m.embed.shape[0]

@torch.no_grad()
def prefill(ids):
    """Full multi-layer prefill. Returns per-layer K,V tensors [n_kv, S, hd] (post-RoPE)."""
    T=len(ids)
    ids_t=torch.tensor(ids,device=DEV)
    h=m.embed[ids_t]  # [T, hidden]
    pos=torch.arange(T,device=DEV).float().view(T,1)
    Ks=[]; Vs=[]
    for li in range(m.N):
        Lw=m.L[li]
        x=m._rmsnorm(h, Lw["ln1"])
        q=(x@Lw["wq"].T+Lw["bq"]).view(T,m.n_q,m.hd)
        k=(x@Lw["wk"].T+Lw["bk"]).view(T,m.n_kv,m.hd)
        v=(x@Lw["wv"].T+Lw["bv"]).view(T,m.n_kv,m.hd)
        # rope per-position
        ang=pos*m.inv_freq  # [T, hd/2]
        cos=torch.cat([ang.cos(),ang.cos()],dim=-1)[:,None,:]
        sin=torch.cat([ang.sin(),ang.sin()],dim=-1)[:,None,:]
        def rope(t):
            t1=t[...,:m.hd//2]; t2=t[...,m.hd//2:]
            return (t.float()*cos+torch.cat([-t2,t1],dim=-1).float()*sin).to(t.dtype)
        q=rope(q); k=rope(k)
        # causal attention over full seq, GQA
        rep=m.n_q//m.n_kv
        Kf=k.permute(1,0,2).repeat_interleave(rep,dim=0)  # [n_q,T,hd]
        Vf=v.permute(1,0,2).repeat_interleave(rep,dim=0)
        Qf=q.permute(1,0,2)  # [n_q,T,hd]
        o=torch.nn.functional.scaled_dot_product_attention(Qf,Kf,Vf,is_causal=True)
        o=o.permute(1,0,2).reshape(T,m.n_q*m.hd)
        h=h+(o@Lw["wo"].T)
        xn=m._rmsnorm(h,Lw["ln2"])
        gate=torch.nn.functional.silu((xn@Lw["gate"].T).float()).to(m.dtype)
        up=xn@Lw["up"].T
        h=h+((gate*up)@Lw["down"].T)
        Ks.append(k.permute(1,0,2).contiguous())  # [n_kv,T,hd]
        Vs.append(v.permute(1,0,2).contiguous())
    return Ks,Vs

def run(off, label):
    base=torch.randint(0,V,(S,)).tolist()
    # carve a fixed-width slot
    for i in range(W): base[off+i]= (1000+i)%V
    edited=list(base)
    for i in range(W): edited[off+i]= (2000+i)%V   # new same-width values
    K0,V0=prefill(base); K1,V1=prefill(edited)
    print(f"\n=== {label}: OFF={off} (slot {off}..{off+W}), suffix=[{off+W}..{S}] ===")
    print("| layer | suffix max|dK| | suffix max|dV| | slot max|dK| |")
    print("|---|---|---|---|")
    suf=slice(off+W,S)
    for li in range(m.N):
        dK_suf=(K0[li][:,suf,:]-K1[li][:,suf,:]).abs().max().item() if off+W<S else 0.0
        dV_suf=(V0[li][:,suf,:]-V1[li][:,suf,:]).abs().max().item() if off+W<S else 0.0
        dK_slot=(K0[li][:,off:off+W,:]-K1[li][:,off:off+W,:]).abs().max().item()
        print(f"| {li} | {dK_suf:.4e} | {dV_suf:.4e} | {dK_slot:.4e} |")

print(f"E4b multi-layer suffix-divergence test: N={N} layers, S={S}, W={W}, Qwen2.5-7B")
run(S//2, "INTERIOR EDIT")
run(S-W,  "TERMINAL EDIT")
