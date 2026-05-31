import sys, os; sys.path.insert(0,"src")
import torch, random
from decode_layer import QwenLayerN
m=QwenLayerN(num_layers=28,device="cuda"); V=m.embed.shape[0]
torch.manual_seed(0)
@torch.no_grad()
def kv(ids):
    T=len(ids);ids_t=torch.tensor(ids,device="cuda");pos=torch.arange(T,device="cuda").float().view(T,1);h=m.embed[ids_t]
    def rp(t,p):
        ang=p*m.inv_freq;cos=torch.cat([ang.cos(),ang.cos()],dim=-1)[:,None,:];sin=torch.cat([ang.sin(),ang.sin()],dim=-1)[:,None,:]
        t1=t[...,:m.hd//2];t2=t[...,m.hd//2:];return (t.float()*cos+torch.cat([-t2,t1],dim=-1).float()*sin).to(t.dtype)
    rep=m.n_q//m.n_kv;Ks=[];Vs=[]
    for li in range(m.N):
        Lw=m.L[li];x=m._rmsnorm(h,Lw["ln1"])
        q=(x@Lw["wq"].T+Lw["bq"]).view(T,m.n_q,m.hd);k=(x@Lw["wk"].T+Lw["bk"]).view(T,m.n_kv,m.hd);vv=(x@Lw["wv"].T+Lw["bv"]).view(T,m.n_kv,m.hd)
        q=rp(q,pos);k=rp(k,pos);Kf=k.permute(1,0,2).repeat_interleave(rep,dim=0);Vf=vv.permute(1,0,2).repeat_interleave(rep,dim=0);Qf=q.permute(1,0,2)
        o=torch.nn.functional.scaled_dot_product_attention(Qf,Kf,Vf,is_causal=True).permute(1,0,2).reshape(T,m.n_q*m.hd)
        h=h+(o@Lw["wo"].T);xn=m._rmsnorm(h,Lw["ln2"]);g=torch.nn.functional.silu((xn@Lw["gate"].T).float()).to(m.dtype);u=xn@Lw["up"].T;h=h+((g*u)@Lw["down"].T)
        Ks.append(k.permute(1,0,2).contiguous());Vs.append(vv.permute(1,0,2).contiguous())
    return Ks,Vs
# A repeated CHUNK of identical tokens appears at position p1 (early) and p2 (later) in the SAME
# context. RadixAttention can't reuse p1's KV for p2 (different absolute position). Question:
# how DIFFERENT is the KV of the same tokens at p1 vs p2? If small at some layers, a
# position-shifted reuse (de-RoPE/re-RoPE) could recover sharing RadixAttention misses.
W=48  # repeated chunk width
chunk=[random.randint(0,V-1) for _ in range(W)]
filler1=[random.randint(0,V-1) for _ in range(100)]
filler2=[random.randint(0,V-1) for _ in range(400)]
seq = filler1 + chunk + filler2 + chunk   # chunk at pos 100 and pos 100+W+400=548
p1=len(filler1); p2=len(filler1)+W+len(filler2)
Ks,Vs=kv(seq)
print(f"Same {W}-token chunk at pos {p1} and pos {p2} (gap {p2-p1}). KV difference per layer:")
print("| layer | max|dK| (K is RoPE'd) | max|dV| (V is NOT RoPE'd) |")
for li in [0,1,7,14,21,27]:
    dK=(Ks[li][:,p1:p1+W,:]-Ks[li][:,p2:p2+W,:]).abs().max().item()
    dV=(Vs[li][:,p1:p1+W,:]-Vs[li][:,p2:p2+W,:]).abs().max().item()
    print(f"| {li} | {dK:.3f} | {dV:.4f} |")
# Critical test: V is NOT position-rotated. Is V of the same tokens nearly identical across positions?
# If yes at layer 0 but diverging at depth (because V depends on attention over different context),
# that quantifies how much CONTEXT (not just position) drives KV -> bounds any positional-reuse scheme.
