import sys, os; sys.path.insert(0,"src")
import torch, random
from decode_layer import QwenLayerN
m=QwenLayerN(num_layers=28,device="cuda"); V=m.embed.shape[0]
torch.manual_seed(0)

@torch.no_grad()
def prefill(ids, split=None):
    """Single-pass (split=None) vs two-chunk prefill [0:split]+[split:]. Returns last-token logits.
    Two-chunk emulates real incremental/chunked prefill: chunk1 KV cached, chunk2 attends to it.
    Mathematically identical; fp16 reduction order differs -> tests if outputs diverge."""
    T=len(ids); ids_t=torch.tensor(ids,device="cuda")
    pos=torch.arange(T,device="cuda").float().view(T,1)
    h=m.embed[ids_t]
    def rope_full(t,p):
        ang=p*m.inv_freq; cos=torch.cat([ang.cos(),ang.cos()],dim=-1)[:,None,:]; sin=torch.cat([ang.sin(),ang.sin()],dim=-1)[:,None,:]
        t1=t[...,:m.hd//2];t2=t[...,m.hd//2:]; return (t.float()*cos+torch.cat([-t2,t1],dim=-1).float()*sin).to(t.dtype)
    rep=m.n_q//m.n_kv
    for li in range(m.N):
        Lw=m.L[li]; x=m._rmsnorm(h,Lw["ln1"])
        q=(x@Lw["wq"].T+Lw["bq"]).view(T,m.n_q,m.hd); k=(x@Lw["wk"].T+Lw["bk"]).view(T,m.n_kv,m.hd); v=(x@Lw["wv"].T+Lw["bv"]).view(T,m.n_kv,m.hd)
        q=rope_full(q,pos); k=rope_full(k,pos)
        Kf=k.permute(1,0,2).repeat_interleave(rep,dim=0); Vf=v.permute(1,0,2).repeat_interleave(rep,dim=0); Qf=q.permute(1,0,2)
        if split is None:
            o=torch.nn.functional.scaled_dot_product_attention(Qf,Kf,Vf,is_causal=True)
        else:
            # chunk1 queries attend causally within [0:split]; chunk2 queries attend to [0:T] causally.
            o1=torch.nn.functional.scaled_dot_product_attention(Qf[:,:split,:],Kf[:,:split,:],Vf[:,:split,:],is_causal=True)
            mask=torch.full((T-split,T),float('-inf'),device="cuda")
            for i in range(T-split): mask[i,:split+i+1]=0.0
            o2=torch.nn.functional.scaled_dot_product_attention(Qf[:,split:,:],Kf,Vf,attn_mask=mask)
            o=torch.cat([o1,o2],dim=1)
        o=o.permute(1,0,2).reshape(T,m.n_q*m.hd)
        h=h+(o@Lw["wo"].T); xn=m._rmsnorm(h,Lw["ln2"]); g=torch.nn.functional.silu((xn@Lw["gate"].T).float()).to(m.dtype); u=xn@Lw["up"].T; h=h+((g*u)@Lw["down"].T)
    return m.logits_of(h[-1])

flips=0; total=0; gaps=[]
for _ in range(60):
    n=random.randint(300,1200); R=[random.randint(0,V-1) for _ in range(n)]
    split=random.randint(50,n-50)
    a=prefill(R, split=None); b=prefill(R, split=split)
    ta,tb=a.argmax().item(), b.argmax().item()
    total+=1
    if ta!=tb: flips+=1
    # top-1 logit gap to nearest rival (margin) — small margins flip easily
    s=a.sort(descending=True).values; gaps.append((s[0]-s[1]).item())
import statistics
print(f"chunked-vs-single prefill: next-token argmax FLIPS = {flips}/{total}")
print(f"median top1-top2 logit margin: {statistics.median(gaps):.3f}")
