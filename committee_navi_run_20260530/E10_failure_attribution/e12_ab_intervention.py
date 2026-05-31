"""
E12 — T5 intervention A/B: does ERROR-CLASS-AWARE reformatting reduce identical-retry rate?
For each real path_missing failure, present the agent (claude CLI) a controlled mini-task:
it just issued tool call C and got an error. Two arms:
  CONTROL  : raw error string (what the agent saw in real traces).
  TREATMENT: same error + a class-aware augmentation (the actual `ls` of the parent dir +
             "the path does not exist; nearest existing paths are: ...").
Ask the agent for its NEXT action as a single tool call. Classify: REPEAT (reissues the same
failing path) vs ADAPT (different path/strategy). Measure repeat-rate per arm. Done many trials.
"""
import json, os, subprocess, re, tempfile
EX=json.load(open("/tmp/pathmiss_examples.json"))
# build synthetic-but-grounded path_missing cases with a KNOWN correct neighbor, so "adapt" is
# objectively detectable. Use a temp dir we control.
import random
def make_case(i):
    d=tempfile.mkdtemp(prefix=f"e12_{i}_")
    # real file exists at d/src/app.py ; agent will be told it tried d/source/app.py (wrong)
    os.makedirs(os.path.join(d,"src"),exist_ok=True)
    open(os.path.join(d,"src","app.py"),"w").write("# real file\nprint('hi')\n")
    wrong=os.path.join(d,"source","app.py")  # nonexistent dir 'source'
    return d, wrong, os.path.join(d,"src","app.py")

CONTROL_TMPL="""You are a coding agent. You ran:
  Read("{wrong}")
and got this error:
  Error: File does not exist: {wrong}
Output ONLY your next single action as: Read("<path>")  (one line, nothing else)."""

TREAT_TMPL="""You are a coding agent. You ran:
  Read("{wrong}")
and got this error (augmented by the runtime):
  Error: File does not exist: {wrong}
  [runtime hint] Parent dir listing of {parent}: {listing}
  Nearest existing paths: {nearest}
Output ONLY your next single action as: Read("<path>")  (one line, nothing else)."""

def ask(prompt):
    env=os.environ.copy()
    for k in ["THRIFT_TLS_CL_CERT_PATH","THRIFT_TLS_CL_KEY_PATH","AGENT","META_AGENT_ROLE"]: env.pop(k,None)
    r=subprocess.run(["claude","--model","claude-opus-4-6","-p",prompt],
                     capture_output=True,text=True,timeout=90,env=env,stdin=subprocess.DEVNULL)
    return r.stdout.strip()

def classify(resp, wrong, correct):
    # REPEAT if it reissues the wrong path; ADAPT if it points elsewhere (esp. the correct dir)
    paths=re.findall(r'Read\("([^"]+)"\)', resp) or re.findall(r'"([^"]+\.py)"', resp)
    if not paths: return "noparse", resp[:60]
    p=paths[0]
    if p==wrong: return "repeat", p
    return "adapt", p

import sys
N=int(os.environ.get("E12_N","10"))
res={"control":{"repeat":0,"adapt":0,"noparse":0},"treatment":{"repeat":0,"adapt":0,"noparse":0}}
for i in range(N):
    d,wrong,correct=make_case(i)
    parent=os.path.dirname(wrong)  # nonexistent 'source'
    gp=os.path.dirname(parent)     # the case root (exists, contains src/)
    listing=", ".join(os.listdir(gp))
    nearest=os.path.join(gp,"src","app.py")
    c=classify(ask(CONTROL_TMPL.format(wrong=wrong)), wrong, correct)
    t=classify(ask(TREAT_TMPL.format(wrong=wrong,parent=gp,listing=listing,nearest=nearest)), wrong, correct)
    res["control"][c[0]]+=1; res["treatment"][t[0]]+=1
    print(f"trial {i}: control={c[0]}({c[1][:40]})  treatment={t[0]}({t[1][:40]})", flush=True)
print("\n=== E12 RESULT ===")
for arm in ["control","treatment"]:
    r=res[arm]; n=r["repeat"]+r["adapt"]
    print(f"{arm:10} repeat={r['repeat']} adapt={r['adapt']} noparse={r['noparse']}  repeat%={100*r['repeat']/max(n,1):.0f}%")
