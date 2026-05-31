import sys, os; sys.path.insert(0,"src")
import torch, random
from decode_layer import QwenLayerN
m=QwenLayerN(num_layers=28,device="cuda"); V=m.embed.shape[0]
@torch.no_grad()
def kv(ids):
    T=len(ids);ids_t=torch.tensor(ids,device="cuda");pos=torch.arange(T,device="cuda").float().view(T,1);h=m.embed[ids_t]
    def rp(t,p):
        ang=p*m.inv_freq;cos=torch.cat([ang.cos(),ang.cos()],dim=-1)[:,None,:];sin=torch.cat([ang.sin(),ang.sin()],dim=-1)[:,None,:]
        t1=t[...,:m.hd//2];t2=t[...,m.hd//2:];return (t.float()*cos+torch.cat([-t2,t1],dim=-1).float()*sin).to(t.dtype)
    rep=m.n_q//m.n_kv;Ks=[];Vs=[];Kpre=[]
    for li in range(m.N):
        Lw=m.L[li];x=m._rmsnorm(h,Lw["ln1"])
        q=(x@Lw["wq"].T+Lw["bq"]).view(T,m.n_q,m.hd);kk=(x@Lw["wk"].T+Lw["bk"]).view(T,m.n_kv,m.hd);vv=(x@Lw["wv"].T+Lw["bv"]).view(T,m.n_kv,m.hd)
        Kpre.append(kk.permute(1,0,2).contiguous())
        q=rp(q,pos);k=rp(kk,pos);Kf=k.permute(1,0,2).repeat_interleave(rep,dim=0);Vf=vv.permute(1,0,2).repeat_interleave(rep,dim=0);Qf=q.permute(1,0,2)
        o=torch.nn.functional.scaled_dot_product_attention(Qf,Kf,Vf,is_causal=True).permute(1,0,2).reshape(T,m.n_q*m.hd)
        h=h+(o@Lw["wo"].T);xn=m._rmsnorm(h,Lw["ln2"]);g=torch.nn.functional.silu((xn@Lw["gate"].T).float()).to(m.dtype);u=xn@Lw["up"].T;h=h+((g*u)@Lw["down"].T)
        Ks.append(k.permute(1,0,2).contiguous());Vs.append(vv.permute(1,0,2).contiguous())
    return Kpre,Ks,Vs
# GATE: for a repeated chunk, how many CONTIGUOUS shallow layers L0..Lk have KV reusable within
# tolerance eps (relative)? That k/28 is the max FLOP fraction reuse could skip for that chunk.
# Then ceiling savings = incidence(1.5%) * (k/28). GREEN if >=10%, KILL if <2%.
def rel(a,b): 
    d=(a-b).abs().max().item(); s=b.abs().max().item()+1e-6; return d/s
for eps in [0.01, 0.05, 0.10]:
    ks=[]
    for _ in range(8):
        W=48;chunk=[random.randint(0,V-1) for _ in range(W)]
        f1=[random.randint(0,V-1) for _ in range(random.randint(50,150))];f2=[random.randint(0,V-1) for _ in range(random.randint(300,500))]
        seq=f1+chunk+f2+chunk;p1=len(f1);p2=len(f1)+W+len(f2)
        Kpre,Ks,Vs=kv(seq)
        k_ok=0
        for li in range(m.N):
            # reuse needs: pre-RoPE K reusable (re-RoPE recovers position) AND V reusable
            rk=rel(Kpre[li][:,p1:p1+W,:],Kpre[li][:,p2:p2+W,:])
            rv=rel(Vs[li][:,p1:p1+W,:],Vs[li][:,p2:p2+W,:])
            if rk<eps and rv<eps: k_ok+=1
            else: break  # contiguous from L0
        ks.append(k_ok)
    import statistics
    kmed=statistics.median(ks)
    layer_frac=kmed/m.N
    ceiling=0.015*layer_frac*100  # incidence 1.5% * layer fraction
    print(f"eps={eps:.2f}: reusable contiguous shallow layers (median) = {kmed}/{m.N} ({layer_frac*100:.0f}% of layers) -> max prefill-FLOP saving ceiling at 1.5% incidence = {ceiling:.3f}%")
