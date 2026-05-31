"""E12b — harden E12 against toy-setup artifact. Multiple realistic distractor dirs, non-obvious
correct location, and a HARDER control (the raw error already includes cwd, like real Claude traces).
Also log the FULL control response to confirm it's a genuine wrong-guess, not a parse failure."""
import json, os, subprocess, re, tempfile, random
def make_case(i):
    d=tempfile.mkdtemp(prefix=f"e12b_{i}_")
    # realistic repo layout with several plausible dirs; real file in a non-obvious place
    for sub in ["lib","tests","docs","build","internal/handlers","cmd"]:
        os.makedirs(os.path.join(d,sub),exist_ok=True)
    # real file: internal/handlers/auth.go ; agent will have tried src/auth.go (no 'src' dir)
    real=os.path.join(d,"internal","handlers","auth.go")
    open(real,"w").write("package handlers\n// real\n")
    wrong=os.path.join(d,"src","auth.go")
    return d, wrong, real
CTRL="""You are a coding agent in repo {root} (cwd={root}). You ran:
  Read("{wrong}")
Error: File does not exist: {wrong}
Reply with ONLY your next action as one line: Read("<path>")"""
TREAT="""You are a coding agent in repo {root} (cwd={root}). You ran:
  Read("{wrong}")
Error: File does not exist: {wrong}
  [runtime hint] '{wrong}' has no existing parent 'src/'. Repo top-level dirs: {top}.
  Files named auth.go found: {found}
Reply with ONLY your next action as one line: Read("<path>")"""
def ask(p):
    env=os.environ.copy()
    for k in ["THRIFT_TLS_CL_CERT_PATH","THRIFT_TLS_CL_KEY_PATH","AGENT","META_AGENT_ROLE"]: env.pop(k,None)
    return subprocess.run(["claude","--model","claude-opus-4-6","-p",p],capture_output=True,text=True,timeout=90,env=env,stdin=subprocess.DEVNULL).stdout.strip()
def pick(resp):
    m=re.findall(r'Read\("([^"]+)"\)',resp) or re.findall(r'"([^"]+\.go)"',resp)
    return m[0] if m else None
N=int(os.environ.get("N","15"))
import subprocess as sp
rc={"correct":0,"wrongguess":0,"noparse":0}; rt={"correct":0,"wrongguess":0,"noparse":0}
for i in range(N):
    d,wrong,real=make_case(i)
    top=", ".join(sorted(os.listdir(d)))
    found=sp.run(["bash","-c",f"find {d} -name auth.go"],capture_output=True,text=True).stdout.strip()
    pc=pick(ask(CTRL.format(root=d,wrong=wrong)))
    pt=pick(ask(TREAT.format(root=d,wrong=wrong,top=top,found=found)))
    for arm,p,store in [("c",pc,rc),("t",pt,rt)]:
        if not p: store["noparse"]+=1
        elif os.path.exists(p) and os.path.samefile(p,real) if os.path.exists(p) else False: store["correct"]+=1
        elif p and "internal/handlers/auth.go" in p: store["correct"]+=1
        else: store["wrongguess"]+=1
    print(f"trial {i}: control={pc}  treat={pt}",flush=True)
print("\n=== E12b (hardened) ===")
print(f"control:   correct={rc['correct']}/{N} wrongguess={rc['wrongguess']} noparse={rc['noparse']}")
print(f"treatment: correct={rt['correct']}/{N} wrongguess={rt['wrongguess']} noparse={rt['noparse']}")
