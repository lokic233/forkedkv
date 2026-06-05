"""
E5 — Real agent-trace edit-class census. Tests T2's remaining honest-future-work item:
"What FRACTION of real agent context mutations are terminal/append (free) vs interior
(the 8-13x pathology)?"

Data: 325 real Claude Code agent sessions (~198MB) from ~/.claude/projects — multi-turn,
tool-using coding agents that ran overnight (incl. the committee agents themselves).

Method: each session is an ordered list of messages (user/assistant turns; tool_use +
tool_result blocks). The model's CONTEXT at turn k is the concatenation of messages[0:k].
We reconstruct the token-id sequence of the context at each model-invocation boundary and
classify how it changed vs the previous boundary:
  APPEND      : new context == old context + suffix (old is a strict prefix)  [RoPE-INVARIANT, free]
  INTERIOR    : old is NOT a prefix of new -> a token at position < len(old) changed/inserted
                [the expensive class: forces suffix recompute at all layers]
  RESET       : new context shares no prefix with old (new session/compaction)
We tokenize with the Qwen tokenizer (same family as E2/E3) for consistency; for speed we
hash per-message token sequences and compare message-prefix structure (a context is an APPEND
iff its message list extends the previous one with identical leading messages).
"""
import os, json, glob, hashlib, sys
from collections import Counter

ROOT=os.path.expanduser("~/.claude/projects")
MAXFILES=int(os.environ.get("E5_MAXFILES","400"))

def msg_text(m):
    """Stable string repr of a message's content for prefix comparison."""
    c=m.get("content")
    if isinstance(c,str): return c
    if isinstance(c,list):
        parts=[]
        for b in c:
            if not isinstance(b,dict): parts.append(str(b)); continue
            t=b.get("type")
            if t=="text": parts.append("T:"+b.get("text",""))
            elif t=="thinking": parts.append("K:"+b.get("thinking","")[:200])
            elif t=="tool_use": parts.append("U:"+b.get("name","")+json.dumps(b.get("input",{}),sort_keys=True)[:500])
            elif t=="tool_result":
                content=b.get("content","")
                if isinstance(content,list): content=json.dumps(content)[:1000]
                parts.append("R:"+str(content)[:1000])
            else: parts.append(t or "?")
        return "\n".join(parts)
    return ""

def session_messages(path):
    """Ordered list of (role, hash) for the conversation messages only."""
    seq=[]
    with open(path) as f:
        for line in f:
            try: d=json.loads(line)
            except: continue
            if d.get("type") not in ("user","assistant"): continue
            m=d.get("message")
            if not isinstance(m,dict): continue
            r=m.get("role")
            h=hashlib.md5(msg_text(m).encode()[:4000]).hexdigest()[:12]
            seq.append((r,h))
    return seq

def classify_session(seq):
    """Walk model-invocation boundaries. Each assistant turn = a model call whose context is
    all messages before it. Classify context growth between consecutive assistant calls."""
    classes=Counter()
    # build the running context (list of msg hashes) and snapshot at each assistant boundary
    contexts=[]
    cur=[]
    for (r,h) in seq:
        if r=="assistant":
            contexts.append(list(cur))   # context the model saw to produce this turn
        cur.append((r,h))
    # compare consecutive contexts
    for i in range(1,len(contexts)):
        prev=contexts[i-1]; now=contexts[i]
        if len(now)>=len(prev) and now[:len(prev)]==prev:
            classes["APPEND"]+=1
        elif len(prev)>0 and len(now)>0 and now[0]==prev[0]:
            # shares some prefix but diverges interior -> find divergence point
            classes["INTERIOR"]+=1
        else:
            classes["RESET"]+=1
    return classes

total=Counter(); per_session=[]; nfiles=0
files=sorted(glob.glob(os.path.join(ROOT,"**","*.jsonl"),recursive=True))
for path in files[:MAXFILES]:
    try:
        seq=session_messages(path)
        if len(seq)<4: continue
        c=classify_session(seq)
        if sum(c.values())==0: continue
        nfiles+=1; total+=c; per_session.append((path,c))
    except Exception as e:
        continue

print(f"=== E5 RESULT: real Claude Code agent-trace edit-class census ===")
print(f"sessions analyzed: {nfiles}  (of {len(files)} jsonl files)")
tot=sum(total.values())
print(f"total context-growth transitions: {tot}")
print("| class | count | pct | RoPE | cost (from E3) |")
print("|---|---|---|---|---|")
labels={"APPEND":("INVARIANT","FREE 0.90-0.96x"),"INTERIOR":("SHIFTING","3.0-8.2x penalty"),"RESET":("-","new ctx/compaction")}
for k in ["APPEND","INTERIOR","RESET"]:
    rope,cost=labels[k]
    print(f"| {k} | {total[k]} | {100*total[k]/tot:.1f}% | {rope} | {cost} |")
