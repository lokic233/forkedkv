"""experiment_log.jsonl writer — every run logged with timestamp+git hash+params+result."""
import json, time, subprocess, os
LOG = os.path.join(os.path.dirname(__file__), "..", "experiment_log.jsonl")

def _git_hash():
    try:
        return subprocess.check_output(["git","rev-parse","--short","HEAD"],
                cwd=os.path.dirname(LOG)).decode().strip()
    except Exception:
        return "nogit"

def log(experiment, params, result, hardware="H100-97GB-1xGPU-cuda12.8-driver580.82"):
    rec = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               git=_git_hash(), experiment=experiment, hardware=hardware,
               params=params, result=result)
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec
