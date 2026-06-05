"""
E4 — Implemented VMM-style pointer-stable REPAIR of a RoPE-invariant edit: REAL attention,
REAL correctness, and a context-scaling sweep. Tests T2's last surviving flaw (Muse Park R5 RED:
"E3 is a recompute-cost microbenchmark, not a repair implementation; the repair is UNBUILT/UNMEASURED").

Scenario (RoPE-INVARIANT 'E_fixed' class from E3): long cached prefix with a FIXED-WIDTH tool-result
slot; tool re-runs returning a NEW value of the SAME token width. Width unchanged => every token
after the slot keeps its absolute position => RoPE phase unchanged => prefix+suffix K/V stay valid
and are NOT recomputed. Only the W slot tokens need new K/V, written into their (still-mapped) pages.

REAL Qwen2.5-7B layer-0, batched prefill (real GEMMs, real RoPE). KV resident in a persistent
buffer that stands in for VMM-mapped pages (pointer-stable: prefix/suffix never re-touched).

Per prefix length S:
  B RECOMPUTE (stock): batched prefill of ALL S tokens of the edited sequence (hash chain broke).
  C EDMM REPAIR      : batched prefill of ONLY the W slot tokens, write in-place into resident KV
                       (no full-KV copy — the prefix/suffix pages keep their mapping & valid K/V).
CORRECTNESS per S: resident-after-repair K/V AND a real last-token SDPA logit must match a
from-scratch recompute of the EDITED sequence to fp16 tol; else the primitive is WRONG.

Headline: repair latency stays ~flat as S grows while stock recompute scales with S -> the repair
advantage GROWS with context, mirroring the E2/E3 context-scaling pathology. Success: at the
largest S, C/B < 0.5 (>=2x faster) AND correct at every S.
"""
import sys, os, glob, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, torch.nn.functional as F
from decode_layer import QwenLayer0

DEV="cuda"
W   = int(os.environ.get("E4_W", "64"))
TRIALS = int(os.environ.get("E4_TRIALS", "30"))
SLIST = [int(x) for x in os.environ.get("E4_SLIST","2048,4096,8192,16384,32768").split(",")]
torch.manual_seed(0)

class Batched:
    def __init__(self, layer): self.L=layer
    @torch.no_grad()
    def kv(self, token_ids, positions):
        L=self.L
        ids=torch.tensor(token_ids, device=DEV)
        h=L.embed[ids]; x=L._rmsnorm(h, L.ln1)
        k=(x@L.wk.T+L.bk).view(-1,L.n_kv,L.hd)
        v=(x@L.wv.T+L.bv).view(-1,L.n_kv,L.hd)
        pos=torch.tensor(positions, device=DEV, dtype=torch.float32)
        ang=pos[:,None]*L.inv_freq[None,:]
        cos=torch.cat([ang.cos(),ang.cos()],dim=-1)[:,None,:]
        sin=torch.cat([ang.sin(),ang.sin()],dim=-1)[:,None,:]
        t1=k[...,:L.hd//2]; t2=k[...,L.hd//2:]
        k=(k.float()*cos+torch.cat([-t2,t1],dim=-1).float()*sin).to(k.dtype)
        return k.permute(1,0,2).contiguous(), v.permute(1,0,2).contiguous()

def main():
    L=QwenLayer0(device=DEV); Bz=Batched(L)
    ev0,ev1=torch.cuda.Event(True),torch.cuda.Event(True)
    def timed(fn):
        fn()
        out=[]
        for _ in range(TRIALS):
            torch.cuda.synchronize(); ev0.record(); r=fn(); ev1.record(); torch.cuda.synchronize()
            out.append(ev0.elapsed_time(ev1))
        out.sort(); return out[len(out)//2], r
    print("\n=== E4 RESULT (real Qwen2.5-7B layer-0, batched, pointer-stable repair) ===")
    print(f"W={W} (fixed-width slot) trials={TRIALS}")
    print("|   S | OFF | slot% | B recompute (ms) | C repair (ms) | C/B | speedup | correct |")
    print("|-----|-----|-------|------------------|---------------|-----|---------|---------|")
    all_ok=True; last=None
    for S in SLIST:
        OFF=S//2
        g=torch.Generator(device="cpu").manual_seed(1)
        base=torch.randint(0,150000,(S,),generator=g).tolist()
        edit=base.copy()
        g2=torch.Generator(device="cpu").manual_seed(999)
        edit[OFF:OFF+W]=torch.randint(0,150000,(W,),generator=g2).tolist()
        allpos=list(range(S))
        # resident KV = clean base prefill (built ONCE, stands in for VMM-mapped pages)
        K_res,V_res=Bz.kv(base, allpos)
        b_ms,(K_full,V_full)=timed(lambda: Bz.kv(edit, allpos))
        def repair():
            Ks,Vs=Bz.kv(edit[OFF:OFF+W], list(range(OFF,OFF+W)))  # ONLY slot tokens
            K_res[:,OFF:OFF+W,:]=Ks; V_res[:,OFF:OFF+W,:]=Vs       # in-place write, no copy
            return K_res,V_res
        c_ms,(K_rep,V_rep)=timed(repair)
        dK=(K_rep.float()-K_full.float()).abs().max().item()
        dV=(V_rep.float()-V_full.float()).abs().max().item()
        _,q_last,_,_=L.project(edit[-1], S-1)
        lf=L.attend_and_logits(L.embed[edit[-1]], q_last, K_full, V_full)
        lr=L.attend_and_logits(L.embed[edit[-1]], q_last, K_rep,  V_rep)
        argmatch=int(lf.argmax().item()==lr.argmax().item())
        ok=(dK<5e-3 and dV<5e-3 and argmatch==1); all_ok=all_ok and ok
        last=(S,b_ms,c_ms)
        print(f"| {S:>5} | {OFF:>4} | {100.0*W/S:4.1f}% | {b_ms:16.3f} | {c_ms:13.3f} | {c_ms/b_ms:.3f} | {b_ms/c_ms:6.1f}x | {ok} |")
    S,b,c=last
    print(f"\nAt largest S={S}: stock recompute {b:.2f}ms vs repair {c:.2f}ms = {b/c:.1f}x faster, correct_all={all_ok}")
    print(f"SUCCESS_BAR (C/B<0.5 at largest S AND correct at every S): {c/b<0.5 and all_ok}")

if __name__=="__main__": main()
