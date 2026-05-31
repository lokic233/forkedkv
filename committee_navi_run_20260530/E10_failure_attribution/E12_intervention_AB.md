# E12 — T5 intervention A/B: does error-class-aware reformatting improve recovery? -> ARTIFACT.

## Design: real path_missing failures; agent (claude-opus-4-6) given CONTROL (raw error) vs
## TREATMENT (error + parent-dir listing + nearest path). Measure correct recovery of the real file.

## E12 (first cut, IMPOVERISHED control prompt): control 0/20 correct, treatment 20/20 (+100 pts).
Looked like a decisive intervention win.

## E12b (HARDENED control = adds cwd + realistic repo layout; the artifact check):
8/8 completed trials: CONTROL recovered correctly 8/8 (agent finds internal/handlers/auth.go on its
OWN). Treatment ~equal. => E12's 0->100% delta was a TOY-SETUP ARTIFACT: the original control prompt
was impoverished (no cwd/layout), so the agent guessed blindly. With a FAIR control that includes the
context a real harness already provides, the agent recovers WITHOUT the hint and the intervention's
marginal effect collapses. (Run stalled at 8/15 on a hung CLI call; trend unambiguous.)

## HONEST CONCLUSION: the intervention A/B does NOT support T5's GREEN-gate. The apparent win was
## control-impoverishment, caught by the hardened re-run. T5 stays YELLOW. No GREEN manufactured.

## Also E11 (cross-harness, codex traces): only 9 errors / 208 calls (codex error rate 4.3% vs
## Claude Code 13.2%) -> underpowered, cannot confirm the class->repeat pattern on a 2nd harness.
