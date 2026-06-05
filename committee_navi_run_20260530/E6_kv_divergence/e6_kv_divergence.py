"""
E6 — T3 core measurement: does TOKEN-prefix sharing (what RadixAttention exploits) OVERSTATE the
true KV-level sharing opportunity in REAL agent branch families, and is the gap NON-TRIVIAL +
LAYER-DEPENDENT (not just 1-(tail/total))?

Tests both T3 YELLOW objections:
- Muse Park "trivial 1-(tail/total)": if token-share != KV-share and the gap grows with layer
  depth, the model is NOT a trivial token ratio.
- claude48 "RadixAttention already does dynamic sharing / no delta": RadixAttention shares on
  token-prefix match. If true KV diverges EARLIER than tokens do, RadixAttention OVER-shares
  (reuses KV that is actually stale at depth) OR the real reusable fraction is smaller -> a
  measurable a-priori-divergence model beats reactive token matching.

Method: REAL Qwen2.5-7B (N layers). For each real agent branch pair (two sibling sessions sharing
an opening prefix), tokenize both, find the token-level common-prefix length Ltok. Then run both
through the model and, per layer, find the KV-level common-prefix length Lkv(layer) = #leading
positions whose K AND V match within fp16 tol. Report Lkv(layer)/Ltok: if <1 and decreasing with
layer, token-sharing overstates KV-sharing by a layer-dependent amount = the non-trivial structure.
"""
import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import torch
from decode_layer import QwenLayerN
from transformers import AutoTokenizer

DEV="cuda"; N=int(os.environ.get("E6_LAYERS","6")); MAXTOK=int(os.environ.get("E6_MAXTOK","1024"))
TOL=float(os.environ.get("E6_TOL","5e-3")); MAXPAIRS=int(os.environ.get("E6_MAXPAIRS","12"))
SNAP=os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots")
SNAP=os.path.join(SNAP, os.listdir(SNAP)[0])
tok=AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
m=QwenLayerN(num_layers=N, device=DEV)

@torch.no_grad()
def perlayer_kv(ids):
    T=len(ids); ids_t=torch.tensor(ids,device=DEV); h=m.embed[ids_t]
    pos=torch.arange(T,device=DEV).float().view(T,1)
    Ks=[];Vs=[]
    for li in range(m.N):
        Lw=m.L[li]; x=m._rmsnorm(h,Lw["ln1"])
        q=(x@Lw["wq"].T+Lw["bq"]).view(T,m.n_q,m.hd); k=(x@Lw["wk"].T+Lw["bk"]).view(T,m.n_kv,m.hd); v=(x@Lw["wv"].T+Lw["bv"]).view(T,m.n_kv,m.hd)
        ang=pos*m.inv_freq; cos=torch.cat([ang.cos(),ang.cos()],dim=-1)[:,None,:]; sin=torch.cat([ang.sin(),ang.sin()],dim=-1)[:,None,:]
        def rope(t):
            t1=t[...,:m.hd//2];t2=t[...,m.hd//2:]; return (t.float()*cos+torch.cat([-t2,t1],dim=-1).float()*sin).to(t.dtype)
        q=rope(q);k=rope(k); rep=m.n_q//m.n_kv
        Kf=k.permute(1,0,2).repeat_interleave(rep,dim=0);Vf=v.permute(1,0,2).repeat_interleave(rep,dim=0);Qf=q.permute(1,0,2)
        o=torch.nn.functional.scaled_dot_product_attention(Qf,Kf,Vf,is_causal=True).permute(1,0,2).reshape(T,m.n_q*m.hd)
        h=h+(o@Lw["wo"].T); xn=m._rmsnorm(h,Lw["ln2"]); gate=torch.nn.functional.silu((xn@Lw["gate"].T).float()).to(m.dtype); up=xn@Lw["up"].T; h=h+((gate*up)@Lw["down"].T)
        Ks.append(k.permute(1,0,2).contiguous());Vs.append(v.permute(1,0,2).contiguous())
    return Ks,Vs

def kv_common_prefix(Ka,Va,Kb,Vb,Ltok):
    """#leading positions (up to Ltok) where K&V match within TOL, at this layer."""
    n=min(Ka.shape[1],Kb.shape[1],Ltok)
    dK=(Ka[:,:n,:]-Kb[:,:n,:]).abs().amax(dim=(0,2))  # [n]
    dV=(Va[:,:n,:]-Vb[:,:n,:]).abs().amax(dim=(0,2))
    bad=((dK>TOL)|(dV>TOL)).nonzero()
    return (bad[0,0].item() if bad.numel()>0 else n)

def first_user_text(path):
    with open(path) as f:
        for line in f:
            try:d=json.loads(line)
            except:continue
            if d.get("type")=="user":
                c=d.get("message",{}).get("content")
                if isinstance(c,str) and len(c)>50: return c
                if isinstance(c,list):
                    for b in c:
                        if isinstance(b,dict) and b.get("type")=="text" and len(b.get("text",""))>50: return b["text"]
    return None
def full_text(path,limit=8000):
    out=[]
    with open(path) as f:
        for line in f:
            try:d=json.loads(line)
            except:continue
            if d.get("type") in ("user","assistant"):
                c=d.get("message",{}).get("content")
                if isinstance(c,str): out.append(c)
                elif isinstance(c,list):
                    for b in c:
                        if isinstance(b,dict): out.append(b.get("text") or b.get("thinking") or json.dumps(b.get("input",b.get("content","")))[:500])
            if sum(len(x) for x in out)>limit: break
    return "\n".join(out)

# build branch pairs
files=sorted(glob.glob(os.path.expanduser("~/.claude/projects/**/*.jsonl"),recursive=True))
from collections import defaultdict
fam=defaultdict(list)
for f in files[:325]:
    t=first_user_text(f)
    if t: fam[t[:200]].append(f)
pairs=[]
for k,v in fam.items():
    if len(v)>=2:
        for i in range(min(len(v)-1,3)):
            pairs.append((v[i],v[i+1]))
pairs=pairs[:MAXPAIRS]
print(f"E6: {len(pairs)} real agent branch pairs, Qwen2.5-7B {N} layers, max {MAXTOK} tok, tol {TOL}")
print("| pair | Ltok (token-shared) | Lkv@L0 | Lkv@Lmid | Lkv@Llast | KVshare/tokshare@last |")
print("|---|---|---|---|---|---|")
import statistics
ratios=[]
for idx,(fa,fb) in enumerate(pairs):
    ta=tok.encode(full_text(fa))[:MAXTOK]; tb=tok.encode(full_text(fb))[:MAXTOK]
    # token common prefix
    Ltok=0
    for x,y in zip(ta,tb):
        if x==y: Ltok+=1
        else: break
    if Ltok<8: continue
    Ka,Va=perlayer_kv(ta); Kb,Vb=perlayer_kv(tb)
    lkv=[kv_common_prefix(Ka[li],Va[li],Kb[li],Vb[li],Ltok) for li in range(m.N)]
    last_ratio=lkv[-1]/Ltok if Ltok else 0
    ratios.append(last_ratio)
    print(f"| {idx} | {Ltok} | {lkv[0]} | {lkv[m.N//2]} | {lkv[-1]} | {last_ratio:.3f} |")
if ratios:
    print(f"\nmedian KVshare/tokshare at last layer: {statistics.median(ratios):.3f}  (1.0 = token-share fully predicts KV-share; <1 = token-share OVERSTATES reusable KV)")
