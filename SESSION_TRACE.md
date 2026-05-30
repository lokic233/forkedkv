# ForkedKV Research Session Trace
## Full history of the autonomous research factory run (2026-05-29 to 2026-05-30)

---

## Phase 1: Multi-Agent Academic Committee (8.76h)

**Start:** 2026-05-29 05:42 UTC
**End:** 2026-05-29 14:28 UTC

### What happened:
- Built a SOTA multi-agent research committee using 5 model lineages (Claude 4.8/4.7/4.6, gpt-5.5 via codex, Avocado via metacode, Gemini)
- Ran 11 seed proposals through 10-phase adversarial pipeline
- Generated 27 fresh research directions via 3 ideation sub-agents
- Phase 9: backtest consensus on seeds (A=RED, G=RED, I=YELLOW)
- Phase 10: backtest on 12 fresh ideas → 2 GREEN (DET_8 Determinism Bisect, DET_6 LSM-for-Agents)
- Phase 11: reframed 5 YELLOWs → 1 more GREEN (RF_4 AgentTraceStore)

### Key deliverables:
- `~/public_agent_infra_research_engagement/` (full research engagement, 72+ files)
- `reports/FINAL_PREMIUM_RESEARCH_REPORT_v4.md`
- 3 GREEN research directions identified

---

## Phase 2: ASPLOS Framing Debate (2h)

**Start:** 2026-05-29 ~17:00 UTC

### What happened:
- User's spec asked: can DET_8 be elevated into ASPLOS-grade contribution?
- 3-CLI debate (codex/claude/gemini) → unanimous: Framing B (Counterfactual Execution Engine)
- 4-CLI ASPLOS-specific debate → unanimous: angle (a) GPU-state CoW with KV-page aliasing
- Title converged: "Forkable GPU Memory for Replayable Agent Execution"
- ASPLOS probability estimated: 35-50% with strong hardware measurements

### Key deliverables:
- `~/public_agent_infra_research_engagement/framing_debate/CHAIR_SYNTHESIS.md`
- `~/public_agent_infra_research_engagement/asplos_debate/CHAIR_SYNTHESIS.md`

---

## Phase 3: Prototype Build + Committee Loop (R0→R1→R2→R3, ~8h)

**Start:** 2026-05-29 20:06 UTC
**R3 complete:** 2026-05-30 03:22 UTC

### Revision history:
| Round | Verdicts | Key changes |
|-------|----------|-------------|
| R0 (v0.1) | 4×YELLOW (15-30%) | CUDA VMM CoW primitives, 5 metrics, honest null results |
| R1 (v0.2) | 1G+3Y (35-45%) | Real Qwen decode loop, dynamic VA, tail divergence, clock fix |
| R2 (v0.3) | 3G+1Y (55-70%) | 4-layer multi-block, unaligned prefix, B8 retraction, forensic OOM |
| R3 (v0.4) | 4×GREEN | 28-layer full model, ceiling model K≈520K, waste quantification |

### Key deliverables:
- `devgpu014:~/branchable_replay/` (18+ commits, head 5cc094f at R3)
- Real H100 measurements across ALL metrics
- 28-layer bit-identical correctness verified

---

## Phase 4: Extended Lab Experiments (2h)

**Start:** 2026-05-30 03:23 UTC

### Experiments completed:
- **Exp 1 (TLB pressure):** ±0.02% overhead at peak — TLB absorbs VMM indirection
- **Exp 2 (Driver contention):** 258× serialization at 128 threads; 100% ioctl time
- Both committed as `a7d3db1`

---

## Phase 5: Lab 1 — vm.max_map_count (1.5h)

**Finding:** vm.max_map_count = 67M on host. VMA count at OOM = 392. Ceiling is DRIVER-INTERNAL, not Linux VMA.
**Commit:** `ec90eee`

---

## Phase 6: Lab 1 Full-Scope Committee + Revision (3h)

### Committee result: 2 GREEN, 2 YELLOW, 1 RED (gemini)
**Gemini's 3 attacks:**
1. Strawman baseline (vLLM APC already does this in software)
2. Solving non-problem (LLMs are append-only)
3. OS transparency illusion (software refcounts, not HW faults)

### Lab 1 R1 Revision (the intellectual repositioning):
- Built software prefix-sharing baseline — honest comparison shows software wins 3/4 axes
- REPOSITIONED paper: "forensic GPU VMM characterization + kernel-transparent VA"
- Retracted "OS-style CoW" language globally
- Added workload justification for branch-and-edit (not just append)
- **Commit:** `3028567`

### Current status: Committee reviewing repositioned prototype (4 reviewers running)

---

## Architecture Summary

```
devgpu014 (H100 ×8):     ~/branchable_replay/  (the prototype, 20+ commits)
dengcchi-mac (laptop):    ~/branchable_replay_review/  (committee reviews, packets, loop state)
                          ~/public_agent_infra_research_engagement/  (research committee output)
GitHub target:            https://github.com/lokic233  (new repo for final artifact)
```

## Nodes Used:
- cli:devgpu014 — H100 GPU, prototype builder, all measurements
- cli:dengcchi-mac — committee reviews, orchestration, file transfer hub
- cli:devgpu499 (AMD MI350) — NOT USED (reserved for cross-vendor experiments)

## Models Used:
- Claude Opus 4.8/4.7/4.6 — builder (H100), native sub-agents, committee
- gpt-5.5 via codex CLI — hostile reviewer, ideation
- Gemini via gemini CLI — hostile reviewer (caught the RED)
- Avocado (Muse Spark) via metacode CLI — reviewer, ideation

## Key Metrics (final prototype v0.4+Lab1R1, commit 3028567):
- 14× capacity gain (84 vs 6 branches, driver-internal ceiling at K≈520K)
- 28-layer full-model bit-identical decode
- ±0.02% TLB/attention overhead
- 258× driver serialization under concurrent load
- Software baseline HONESTLY shown to win on latency (700×) and capacity (6×)
- Hardware wins: kernel-transparent contiguous VA (FlashAttention OOB)
- Paper repositioned: "architectural characterization" not "practical capacity gain"

## Total session elapsed: ~24 hours continuous autonomous operation
