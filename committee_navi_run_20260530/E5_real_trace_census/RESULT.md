# E5 — Real agent-trace edit-class census (T2's remaining future-work item)

## Question: what FRACTION of real agent context mutations are terminal/append (free, per E3) vs
## interior (the 8-13x pathology)? Sizes T2's real-world impact.

## Data
325 REAL Claude Code agent sessions (~198MB) from ~/.claude/projects — multi-turn tool-using
coding agents that ran overnight on real fbsource/repo tasks (including the committee agents
themselves). 12,380 assistant turns, 6,510 tool_use / 6,508 tool_result blocks across 225 analyzable
sessions (>=4 turns).

## Method + DISCIPLINE NOTE (a too-clean first result forced a deeper look)
First pass (message-prefix comparison): 100% APPEND, 0% interior. That was SUSPICIOUSLY clean, so
we audited for KV-invalidating events the coarse method would miss. Found the real ones:
  - system-reminder injections: 15   (injected as TAIL user-turn content -> APPEND)
  - compactSummary / compact_boundary: 8 / 4  (REPLACE whole prefix with summary -> RESET, new cache)
  - 29,594 attachments: file contents, injected at the TAIL before the next prompt -> APPEND
Verified: in Claude Code's append-only loop EVERY tool_result is appended at the tail immediately
after its tool_use (6510 tool_use ~ 6508 tool_result, always tail-adjacent). No tool result is ever
spliced BEFORE already-cached tokens within a live context.

## RESULT (honest classification)
| context-mutation class | share | KV effect |
|---|---|---|
| APPEND (tool result / attachment / reminder at tail) | ~99.9% | RoPE-INVARIANT — FREE under prefix caching (E3: 0.90-0.96x) |
| RESET (compaction: prefix replaced by summary) | ~0.1% (12 events / 12380 turns) | new cached prefix, not an interior edit |
| INTERIOR (tool result spliced before cached tokens) | 0% observed | the 3-8.2x pathology — DID NOT OCCUR in these agent loops |

## FINDING — and its HONEST INTERPRETATION (this CUTS BOTH WAYS for T2)
1. In current append-only agent harnesses (Claude Code), context mutation is ~entirely APPEND, which
   prefix caching ALREADY handles for free. Compaction is a rare full RESET, not an interior splice.
2. => The 8-13x INTERIOR pathology that T2's E2/E3 measured is NOT triggered by today's dominant
   coding-agent loop. The pathology is REAL (measured) but its real-world INCIDENCE in current
   harnesses is ~zero. T2's penalty is LATENT, not active, under append-only orchestration.
3. WHERE it WOULD bite (not measured here, honest scope): harnesses that DO interior-mutate —
   editable scratchpads, RAG re-ranking that re-orders retrieved docs mid-context, structured-memory
   rewriting, multi-agent shared-context editing, speculative branch rollback. EDMM's original 8.21x
   came from a vLLM path that re-injected tool output mid-prompt — an interior edit that THIS harness
   avoids by appending.

## NET FOR THE COMMITTEE (anti-overclaim):
T2's workload-model + characterization is MEASURED and correct (GREEN, R9). E5 adds the honest
incidence caveat: the pathology's frequency is HARNESS-DEPENDENT and ~0% in append-only Claude Code;
it binds only for interior-mutating orchestration patterns. This STRENGTHENS T2's honesty (we now
know exactly when it matters) but it does NOT let T2 claim broad real-world impact for current
agents. Report as: "the penalty is an architectural latent cost that activates iff the harness
performs interior context mutation; mainstream append-only loops avoid it, by structure not by luck."
