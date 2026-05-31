# ROUND 14 — candidate thesis T5 (agent-failure-attribution domain). HONEST modest numbers + two
# self-caught parsing bugs disclosed. Do NOT reward overclaim. Vote GREEN/YELLOW/RED/KILL.

## T5 — "Error-class predicts agent recovery competence"
One-sentence: A coding agent's probability of REPEATING an identical failing tool call (vs adapting)
is determined by the ERROR CLASS it received — measured on 6,510 real Claude Code tool calls,
path_missing errors are re-issued 18.6% of the time vs 6.1% for the baseline 'other' class (~3x) —
suggesting agents parse some error semantics but are systematically blind to others, an attributable
and potentially harness-fixable failure mode.

## MEASURED EVIDENCE (E10, 325 real Claude Code sessions):
- tool-call error rate 13.2% (857/6510); 48% of sessions hit >=1 error.
- After an error, repeat-identical-call rate by class (n>=15):
  path_missing 18.6% (n=118) | size_exceed 14.3% (n=84) | other 6.1% (n=621). Baseline 8.7%.

## SELF-DISCLOSED WEAKNESSES (two parsing bugs already caught + corrected this session):
1. A first coarse cut showed 'no such' at 46.8% repeat — DISSOLVED to 18.6% under better
   classification. The strong signal was a classification artifact. Current spread is only ~3x.
2. A "178 spirals / wasted compute" angle was a parse bug (counted identical signatures without
   verifying prior error); corrected = 10 redundant retries (0.2%), NEGLIGIBLE. That sub-thesis is dead.
3. "other" (621) dominates the sample -> the class signal rests on ~200 well-classified errors.
4. Single harness (Claude Code), single agent family. Correlation, not causal. No intervention shown.
5. Prior art: agent-failure-attribution / reflexion / self-debugging literature is crowded.

## ANTI-FLASHINFER: YES (about agent behavior, not kernel speed).
## GREEN bar would need: a NAMED experiment proving (a) the class->repeat signal holds across >=2
## harnesses or agent families, AND (b) an intervention (e.g. error-class-aware reformatting) that
## measurably reduces repeat-rate. Neither is done.

## VOTE T5. One line. KILL if too marginal/crowded; YELLOW if bounded characterization; GREEN only
## if a defensible non-trivial contribution survives + name the confirming experiment.
T5: <GREEN|YELLOW|RED|KILL> — <=2 lines.
