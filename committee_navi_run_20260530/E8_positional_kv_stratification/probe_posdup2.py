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
        Kpre.append(kk.permute(1,0,2).contiguous())  # K BEFORE rope
        q=rp(q,pos);k=rp(kk,pos);Kf=k.permute(1,0,2).repeat_interleave(rep,dim=0);Vf=vv.permute(1,0,2).repeat_interleave(rep,dim=0);Qf=q.permute(1,0,2)
        o=torch.nn.functional.scaled_dot_product_attention(Qf,Kf,Vf,is_causal=True).permute(1,0,2).reshape(T,m.n_q*m.hd)
        h=h+(o@Lw["wo"].T);xn=m._rmsnorm(h,Lw["ln2"]);g=torch.nn.functional.silu((xn@Lw["gate"].T).float()).to(m.dtype);u=xn@Lw["up"].T;h=h+((g*u)@Lw["down"].T)
        Ks.append(k.permute(1,0,2).contiguous());Vs.append(vv.permute(1,0,2).contiguous())
    return Kpre,Ks,Vs
# repeat over multiple random chunks; report PRE-ROPE K diff and V diff at layer 0 and depth.
import statistics
res={li:{'kpre':[],'v':[]} for li in range(m.N)}
for trial in range(8):
    W=48; chunk=[random.randint(0,V-1) for _ in range(W)]
    f1=[random.randint(0,V-1) for _ in range(random.randint(50,150))]
    f2=[random.randint(0,V-1) for _ in range(random.randint(300,500))]
    seq=f1+chunk+f2+chunk; p1=len(f1); p2=len(f1)+W+len(f2)
    Kpre,Ks,Vs=kv(seq)
    for li in range(m.N):
        res[li]['kpre'].append((Kpre[li][:,p1:p1+W,:]-Kpre[li][:,p2:p2+W,:]).abs().max().item())
        res[li]['v'].append((Vs[li][:,p1:p1+W,:]-Vs[li][:,p2:p2+W,:]).abs().max().item())
print("Same chunk at 2 positions, 8 trials. PRE-RoPE K and V divergence (median):")
print("| layer | max|dK_preRoPE| | max|dV| |  (layer0=pure token identity, depth=context mixing)")
for li in [0,1,2,3,5,9,15,21,27]:
    print(f"| {li} | {statistics.median(res[li]['kpre']):.4f} | {statistics.median(res[li]['v']):.4f} |")
