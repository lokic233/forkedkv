import sys; sys.path.insert(0,"src")
import torch, random
from decode_layer import QwenLayerN
from transformers import AutoTokenizer
import os
SNAP=os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots")
SNAP=os.path.join(SNAP,os.listdir(SNAP)[0]); tok=AutoTokenizer.from_pretrained(SNAP,trust_remote_code=True)
m=QwenLayerN(num_layers=28,device="cuda")
@torch.no_grad()
def prefill(ids, split=None):
    T=len(ids); ids_t=torch.tensor(ids,device="cuda"); pos=torch.arange(T,device="cuda").float().view(T,1); h=m.embed[ids_t]
    def rp(t,p):
        ang=p*m.inv_freq;cos=torch.cat([ang.cos(),ang.cos()],dim=-1)[:,None,:];sin=torch.cat([ang.sin(),ang.sin()],dim=-1)[:,None,:]
        t1=t[...,:m.hd//2];t2=t[...,m.hd//2:];return (t.float()*cos+torch.cat([-t2,t1],dim=-1).float()*sin).to(t.dtype)
    rep=m.n_q//m.n_kv
    for li in range(m.N):
        Lw=m.L[li];x=m._rmsnorm(h,Lw["ln1"])
        q=(x@Lw["wq"].T+Lw["bq"]).view(T,m.n_q,m.hd);k=(x@Lw["wk"].T+Lw["bk"]).view(T,m.n_kv,m.hd);v=(x@Lw["wv"].T+Lw["bv"]).view(T,m.n_kv,m.hd)
        q=rp(q,pos);k=rp(k,pos);Kf=k.permute(1,0,2).repeat_interleave(rep,dim=0);Vf=v.permute(1,0,2).repeat_interleave(rep,dim=0);Qf=q.permute(1,0,2)
        if split is None: o=torch.nn.functional.scaled_dot_product_attention(Qf,Kf,Vf,is_causal=True)
        else:
            o1=torch.nn.functional.scaled_dot_product_attention(Qf[:,:split,:],Kf[:,:split,:],Vf[:,:split,:],is_causal=True)
            msk=torch.full((T-split,T),float('-inf'),device="cuda")
            for i in range(T-split): msk[i,:split+i+1]=0.0
            o2=torch.nn.functional.scaled_dot_product_attention(Qf[:,split:,:],Kf,Vf,attn_mask=msk)
            o=torch.cat([o1,o2],dim=1)
        o=o.permute(1,0,2).reshape(T,m.n_q*m.hd);h=h+(o@Lw["wo"].T)
        xn=m._rmsnorm(h,Lw["ln2"]);g=torch.nn.functional.silu((xn@Lw["gate"].T).float()).to(m.dtype);u=xn@Lw["up"].T;h=h+((g*u)@Lw["down"].T)
    return m.logits_of(h[-1])
# Greedy-generate 60 tokens TWO ways from a real prompt: (A) single-pass prefill each step,
# (B) two-chunk prefill each step. Count how many generated tokens DIVERGE (compounding).
prompt="Write a Python function that merges two sorted linked lists into one sorted list, then explain its time complexity in detail and discuss edge cases."
base=tok.encode(prompt)
seqA=list(base); seqB=list(base)
div_step=-1; diverged=0
for step in range(60):
    la=prefill(seqA, split=None)
    lb=prefill(seqB, split=len(seqB)//2)
    na=la.argmax().item(); nb=lb.argmax().item()
    if na!=nb and div_step<0: div_step=step
    if na!=nb: diverged+=1
    seqA.append(na); seqB.append(nb)
print(f"real-prompt greedy gen, single vs chunked prefill: first divergence at step={div_step}, total divergent steps={diverged}/60")
print("A:",repr(tok.decode(seqA[len(base):len(base)+40])))
print("B:",repr(tok.decode(seqB[len(base):len(base)+40])))
