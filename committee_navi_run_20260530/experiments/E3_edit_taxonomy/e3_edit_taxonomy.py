#!/usr/bin/env python3
"""
E3 — Edit-type taxonomy: per-class recompute cost on a LIVE engine (SGLang RadixAttention).
Tests T2's surviving flaw (Muse Park RED + claude48/gemini YELLOW): is the RoPE-INVARIANT
repairable subclass VACUOUS, or does it have a fundamentally cheaper cost than position-shifting
edits? We measure TTFT for 4 edit classes against a clean cache hit (C0):

  C0  clean cache hit (RadixAttention best case)
  E_append   tool result APPENDED at end of cached prefix (nothing after it)        [RoPE-INVARIANT]
  E_fixed    FIXED-WIDTH interior replace: overwrite a pre-sized slot, |new|==|old| [RoPE-INVARIANT]
  E_varins   VARIABLE-LENGTH interior insert: salt grows context, shifts h2          [RoPE-SHIFTING]
  E_varins0  variable insert at position 0 (worst case, whole context shifts)        [RoPE-SHIFTING]

Hypothesis: E_append ~ C0 (no penalty); E_fixed << E_varins (fixed-width slot is cheaper because
downstream positions don't move -> the repairable subclass is REAL, not vacuous). If E_fixed ~=
E_varins, Muse Park is right and the subclass is vacuous.
"""
import os, time, uuid
from sglang import Engine
from transformers import AutoTokenizer

MODEL = os.environ.get("E3_MODEL", "/tmp/qwen15b")
SCALE = int(os.environ.get("E3_SCALE", "16384"))
TRIALS = 5
SLOT = 64  # fixed-width tool-result slot, in tokens

CODE_UNIT = ("class HTTPRequestHandler:\n    def dispatch(self, method, path):\n"
             "        handler = self._resolve_route(method, path)\n"
             "        return handler(self.request)\n\n")
SUFFIX = "Identify the root cause and produce a minimal unified diff."

def halves(tok, target):
    half = (target - 100)//2
    text = CODE_UNIT*500
    while len(tok.encode(text)) < half:
        text += CODE_UNIT*100
    htext = tok.decode(tok.encode(text)[:half])
    return htext, htext

def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    eng = Engine(model_path=MODEL, mem_fraction_static=float(os.environ.get("E3_MEM","0.5")))
    sp = {"max_new_tokens":1, "temperature":0.0}
    eng.generate("warmup", sp)
    h1, h2 = halves(tok, SCALE)
    clean = h1 + h2 + SUFFIX
    # fixed-width filler the size of SLOT tokens (deterministic, reused -> RoPE-invariant slot)
    slot_fill = tok.decode(tok.encode("X "*200)[:SLOT])

    def ttft(prompt):
        eng.generate(prompt, sp); time.sleep(0.25)
        t0=time.perf_counter(); eng.generate(prompt, sp); return (time.perf_counter()-t0)*1000

    def measure(make):
        # warm the cache with clean once, then hit the variant
        out=[]
        for _ in range(TRIALS):
            eng.generate(clean, sp); time.sleep(0.25)
            t0=time.perf_counter(); eng.generate(make(), sp); out.append((time.perf_counter()-t0)*1000)
        return sum(out)/len(out), out

    # warm clean
    for _ in range(2): eng.generate(clean, sp)
    c0,_ = measure(lambda: clean)
    # E_append: result appended at END (after suffix) -> nothing cached after it
    ap,_ = measure(lambda: clean + f"\nTOOL: {slot_fill}\n")
    # E_fixed: overwrite a FIXED-WIDTH slot already present in the cached prefix (|new|==|old|)
    #   build a cached prefix that CONTAINS a slot, then replace slot contents with same width
    clean_slot = h1 + f"\nTOOL[{slot_fill}]\n" + h2 + SUFFIX
    for _ in range(2): eng.generate(clean_slot, sp)
    fx,_ = measure_fixed(eng, sp, tok, h1, h2, SUFFIX, slot_fill, SLOT, clean_slot, TRIALS)
    # E_varins: variable-length insert between h1 and h2 (shifts h2)
    vi,_ = measure(lambda: h1 + f"\nTOOL: {uuid.uuid4()} {uuid.uuid4()}\n" + h2 + SUFFIX)
    # E_varins0: insert at very start (shifts everything)
    vi0,_ = measure(lambda: f"PREPEND {uuid.uuid4()}\n" + clean)

    print("\n===E3 RESULT=== model=%s scale=%d slot=%d trials=%d"%(MODEL,SCALE,SLOT,TRIALS))
    print("| edit class | RoPE | TTFT ms | x C0 |")
    print("|---|---|---|---|")
    for name,rope,v in [("C0 clean hit","-",c0),("E_append","invariant",ap),
                        ("E_fixed (fixed-width replace)","invariant",fx),
                        ("E_varins (var interior insert)","SHIFTING",vi),
                        ("E_varins0 (prepend)","SHIFTING",vi0)]:
        print("| %s | %s | %.1f | %.2fx |"%(name,rope,v,v/c0))

def measure_fixed(eng, sp, tok, h1,h2,SUFFIX, slot_fill, SLOT, clean_slot, TRIALS):
    import time
    out=[]
    for _ in range(TRIALS):
        eng.generate(clean_slot, sp); time.sleep(0.25)
        # replace slot content with a DIFFERENT same-width token block -> positions of h2 unchanged
        newfill = tok.decode(tok.encode("Y "*200)[:SLOT])
        variant = h1 + f"\nTOOL[{newfill}]\n" + h2 + SUFFIX
        t0=time.perf_counter(); eng.generate(variant, sp); out.append((time.perf_counter()-t0)*1000)
    return sum(out)/len(out), out

if __name__=="__main__":
    main()
