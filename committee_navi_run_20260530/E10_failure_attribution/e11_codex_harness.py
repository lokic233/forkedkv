import json, glob, os
from collections import defaultdict, Counter
files=sorted(glob.glob(os.path.expanduser("~/.codex/sessions/**/*.jsonl"),recursive=True))
def errkind(txt):
    t=txt[:400].lower()
    for k,pats in [("size_exceed",["exceed","too large","maximum allowed","max_output","token count"]),
                   ("path_missing",["no such file","not found","enoent","does not exist","cannot find","cannot stat"]),
                   ("permission",["permission denied","eacces","not permitted","operation not permitted"]),
                   ("syntax_lint",["syntaxerror","parse error","unexpected","invalid syntax","traceback"]),
                   ("test_fail",["test failed","assertionerror","failed test","tests failed","failures="]),
                   ("string_match",["no replacement","string not found","did not match","no match"]),
                   ("nonzero_exit",["exited with code 1","exited with code 2","exit code: 1","non-zero","command failed"]),
                   ("timeout",["timed out","timeout","deadline","killed"])]:
        if any(p in t for p in pats): return k
    return "other"
def is_err(out):
    t=out[:400].lower()
    if "process exited with code 0" in t: return False
    return ("error" in t or "exited with code" in t and "code 0" not in t or "traceback" in t
            or "no such" in t or "denied" in t or "failed" in t or "not found" in t or "exceed" in t)
repeat=defaultdict(int); adapt=defaultdict(int); tot_calls=0; tot_err=0
for f in files:
    calls=[]  # ordered (sig, output)
    pend={}   # call_id -> sig
    with open(f) as fh:
        for line in fh:
            try:d=json.loads(line)
            except:continue
            p=d.get("payload")
            if not isinstance(p,dict): continue
            pt=p.get("type")
            if pt in ("function_call","custom_tool_call"):
                sig=(p.get("name","")+"|"+str(p.get("arguments",""))[:200])
                pend[p.get("call_id")]=sig
            elif pt in ("function_call_output","custom_tool_call_output"):
                sig=pend.get(p.get("call_id"),"?")
                out=str(p.get("output",""))
                calls.append((sig,out))
    tot_calls+=len(calls)
    for k in range(1,len(calls)):
        sp,op=calls[k-1]; sc,oc=calls[k]
        if is_err(op):
            tot_err+=1
            kind=errkind(op)
            if sc==sp: repeat[kind]+=1
            else: adapt[kind]+=1
print(f"CODEX harness: {len(files)} sessions, {tot_calls} tool calls, {tot_err} errors")
print(f"{'error class':14} {'n':>4} {'repeat%':>8}")
allr=alla=0
rows=[]
for k in set(list(repeat)+list(adapt)):
    r=repeat[k];a=adapt[k];n=r+a;allr+=r;alla+=a
    if n>=8: rows.append((k,n,100*r/n))
for k,n,p in sorted(rows,key=lambda x:-x[2]): print(f"{k:14} {n:4} {p:7.1f}%")
print(f"baseline repeat%: {100*allr/max(allr+alla,1):.1f}%  (pairs={allr+alla})")
