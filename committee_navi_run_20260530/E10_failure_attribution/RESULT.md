# E10 — Agent tool-failure structure from REAL traces (agent-failure-attribution domain, H100-free)

## Data: 325 real Claude Code agent sessions, 6,510 tool calls, 847 error->next-action pairs.

## Measured facts:
- Tool-call error rate: 13.2% (857/6510). Sessions with >=1 error: 157/325 (48%).
- Identical-call SPIRALS (>=3 consecutive identical calls): 178; median length 4, MAX 35.
- After a tool error, does the agent REPEAT the identical failing call (spiral) or ADAPT?
  Baseline repeat rate: 8.7%. By refined error class (n>=15):
    | error class   | n   | repeat% |
    | path_missing  | 118 | 18.6%   |
    | size_exceed   | 84  | 14.3%   |
    | other         | 621 | 6.1%    |
  => error CLASS predicts spiral risk by ~3x (6% -> 19%). REAL but MODEST signal.

## HONEST ASSESSMENT (pre-committee, anti-overclaim):
- The phenomenon (agents loop on failures) is KNOWN; the agent-failure-attribution literature
  covers it. The candidate DELTA is the per-error-CLASS repeat-rate quantification on real traces.
- BUT the spread is only ~3x (path_missing 18.6% vs other 6.1%), and "other" (621) dominates the
  sample, so the class signal is weak. A first coarser cut showed 'no such' at 46.8% but better
  classification dissolved most of it -> the strong-signal version was a classification artifact.
- Single harness (Claude Code), single agent family. No causal claim — just outcome correlation.
- Likely a YELLOW characterization at best; flagging honestly for committee disposition.

## CORRECTION (parsing-bug audit): the "spiral / wasted-compute" angle is NEGLIGIBLE.
Initial probe counted 178 "identical-call spirals" — but that counted identical tool_use SIGNATURES
without verifying the prior call had ERRORED. Tool_use lives in an assistant msg, its tool_result in
the NEXT user msg (separate JSONL lines); the redundancy probe had a parse bug. Corrected:
- STRICTLY-redundant retries (identical call issued AFTER seeing it error) = 10 / 6510 = 0.2%.
- sessions affected: 7/325 (2%). wasted context tokens: ~4,581 (trivial).
=> The "agents waste compute spiraling on identical failing calls" thesis is FALSE on real traces.
   Killed by correct measurement before any claim.

## NET surviving signal from the failure-attribution probe: ONLY the modest error-class repeat-rate
## spread (path_missing 18.6% vs other 6.1%, ~3x). Borderline. Putting to committee as candidate T5
## with these HONEST modest numbers — likely YELLOW/KILL.
