"""
E4c — Rescoped repair: TERMINAL/APPEND tool-result injection, FULL multi-layer, exact + timed.
After E4b proved interior edits corrupt suffix K/V at L>0, T2's repair is rescoped to the TERMINAL
case (tool result appended at the live decode boundary — the standard agent pattern: model emits a
tool call, result is appended, generation continues). Here suffix=empty, so pointer-stable reuse of
the cached prefix K/V is EXACT at full depth. We measure the intervention:
  B RECOMPUTE (stock): full N-layer prefill of prefix(S) + appended result(W)  [hash chain broke]
  C REPAIR (rescoped): reuse cached prefix(S) K/V at ALL layers, prefill ONLY the W appended tokens
Correctness: repaired final-token logits (argmax) vs full recompute. Sweep S; W=64.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import torch
from decode_layer import QwenLayerN
DEV="cuda"
N=int(os.environ.get("E4C_LAYERS","6")); W=int(os.environ.get("E4C_W","64"))
SLIST=[int(x) for x in os.environ.get("E4C_SLIST","2048,4096,8192,16384,32768").split(",")]
TR=int(os.environ.get("E4C_TRIALS","20"))
torch.manual_seed(0)
m=QwenLayerN(num_layers=N, device=DEV); Vsz=m.embed.shape[0]

@torch.no_grad()
def layers_prefill(ids, cacheKV=None, only_new_from=None):
    """If only_new_from is None: full prefill of all tokens, return per-layer (K,V) [n_kv,T,hd] + final logits.
       Else: reuse cacheKV for tokens [0:only_new_from], compute only [only_new_from:T], appending
       to cached K/V (exact iff those new tokens are TERMINAL — no earlier token attends to them)."""
    T=len(ids); ids_t=torch.tensor(ids,device=DEV)
    h=m.embed[ids_t]
    start = 0 if only_new_from is None else only_new_from
    pos=torch.arange(T,device=DEV).float().view(T,1)
    Ks=[];Vs=[]
    hcur=h
    for li in range(m.N):
        Lw=m.L[li]
        x=m._rmsnorm(hcur, Lw["ln1"])
        q=(x@Lw["wq"].T+Lw["bq"]).view(T,m.n_q,m.hd)
        k=(x@Lw["wk"].T+Lw["bk"]).view(T,m.n_kv,m.hd)
        v=(x@Lw["wv"].T+Lw["bv"]).view(T,m.n_kv,m.hd)
        ang=pos*m.inv_freq
        cos=torch.cat([ang.cos(),ang.cos()],dim=-1)[:,None,:]; sin=torch.cat([ang.sin(),ang.sin()],dim=-1)[:,None,:]
        def rope(t):
            t1=t[...,:m.hd//2]; t2=t[...,m.hd//2:]
            return (t.float()*cos+torch.cat([-t2,t1],dim=-1).float()*sin).to(t.dtype)
        q=rope(q); k=rope(k)
        if only_new_from is not None and cacheKV is not None:
            # overwrite cached prefix K/V for [0:start], keep new for [start:]
            kc,vc=cacheKV[li]
            k=torch.cat([kc, k.permute(1,0,2)[:,start:,:]],dim=1).permute(1,0,2) if False else k
        rep=m.n_q//m.n_kv
        Kf=k.permute(1,0,2).repeat_interleave(rep,dim=0); Vf=v.permute(1,0,2).repeat_interleave(rep,dim=0)
        Qf=q.permute(1,0,2)
        o=torch.nn.functional.scaled_dot_product_attention(Qf,Kf,Vf,is_causal=True).permute(1,0,2).reshape(T,m.n_q*m.hd)
        hcur=hcur+(o@Lw["wo"].T)
        xn=m._rmsnorm(hcur,Lw["ln2"]); gate=torch.nn.functional.silu((xn@Lw["gate"].T).float()).to(m.dtype); up=xn@Lw["up"].T
        hcur=hcur+((gate*up)@Lw["down"].T)
        Ks.append(k.permute(1,0,2).contiguous()); Vs.append(v.permute(1,0,2).contiguous())
    logits=m.logits_of(hcur[-1])
    return list(zip(Ks,Vs)), logits

# For the TERMINAL case, the exact repair = compute only the W new tokens' contribution. Because no
# earlier token attends to them, the prefix K/V AND prefix hidden states are unchanged at all layers.
# So repair = a W-token incremental prefill that attends to cached prefix K/V. We implement the
# stock recompute (B) and a true incremental (C) and check final-logit argmax equality + timing.

@torch.no_grad()
def incremental(prefix_ids, new_ids, cache):
    """Exact terminal append: for each layer, compute K/V for the W new tokens, attend over
    [cached prefix K/V ++ new K/V], propagate hidden state for the new tokens only."""
    S=len(prefix_ids); T=S+len(new_ids)
    ids_t=torch.tensor(prefix_ids+new_ids,device=DEV)
    pos=torch.arange(T,device=DEV).float().view(T,1)
    hnew=m.embed[ids_t[S:]]  # [W,hidden]  hidden state for new tokens entering layer 0
    posnew=pos[S:]
    for li in range(m.N):
        Lw=m.L[li]; kc,vc=cache[li]  # [n_kv,S,hd]
        x=m._rmsnorm(hnew,Lw["ln1"])
        q=(x@Lw["wq"].T+Lw["bq"]).view(-1,m.n_q,m.hd)
        k=(x@Lw["wk"].T+Lw["bk"]).view(-1,m.n_kv,m.hd)
        v=(x@Lw["wv"].T+Lw["bv"]).view(-1,m.n_kv,m.hd)
        angn=posnew*m.inv_freq; cosn=torch.cat([angn.cos(),angn.cos()],dim=-1)[:,None,:]; sinn=torch.cat([angn.sin(),angn.sin()],dim=-1)[:,None,:]
        def rope(t,c,s):
            t1=t[...,:m.hd//2];t2=t[...,m.hd//2:]; return (t.float()*c+torch.cat([-t2,t1],dim=-1).float()*s).to(t.dtype)
        q=rope(q,cosn,sinn); k=rope(k,cosn,sinn)
        rep=m.n_q//m.n_kv
        Kfull=torch.cat([kc, k.permute(1,0,2)],dim=1).repeat_interleave(rep,dim=0)  # [n_q, S+W, hd]
        Vfull=torch.cat([vc, v.permute(1,0,2)],dim=1).repeat_interleave(rep,dim=0)
        Qn=q.permute(1,0,2)  # [n_q, W, hd]
        # causal mask for the W new queries over S+W keys
        Wn=q.shape[0]
        mask=torch.full((Wn,S+Wn),float('-inf'),device=DEV); 
        for i in range(Wn): mask[i,:S+i+1]=0.0
        o=torch.nn.functional.scaled_dot_product_attention(Qn,Kfull,Vfull,attn_mask=mask).permute(1,0,2).reshape(Wn,m.n_q*m.hd)
        hnew=hnew+(o@Lw["wo"].T)
        xn=m._rmsnorm(hnew,Lw["ln2"]); gate=torch.nn.functional.silu((xn@Lw["gate"].T).float()).to(m.dtype); up=xn@Lw["up"].T
        hnew=hnew+((gate*up)@Lw["down"].T)
    return m.logits_of(hnew[-1])

print(f"E4c TERMINAL repair (rescoped, exact): N={N} layers, W={W}, Qwen2.5-7B")
print("| S | B recompute (ms) | C repair (ms) | C/B | speedup | argmax match |")
print("|---|---|---|---|---|---|")
for S in SLIST:
    prefix=torch.randint(0,Vsz,(S,)).tolist(); new=torch.randint(0,Vsz,(W,)).tolist()
    cache,_=layers_prefill(prefix)  # cache prefix K/V (the "still-mapped pages")
    # B: full recompute of prefix+new
    full=prefix+new
    for _ in range(3): _,lb=layers_prefill(full)
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(TR): _,lb=layers_prefill(full)
    torch.cuda.synchronize(); tb=(time.perf_counter()-t0)/TR*1000
    # C: incremental terminal repair
    for _ in range(3): lc=incremental(prefix,new,cache)
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(TR): lc=incremental(prefix,new,cache)
    torch.cuda.synchronize(); tc=(time.perf_counter()-t0)/TR*1000
    match = (lb.argmax().item()==lc.argmax().item())
    print(f"| {S} | {tb:.3f} | {tc:.3f} | {tc/tb:.3f} | {tb/tc:.1f}x | {match} |")
