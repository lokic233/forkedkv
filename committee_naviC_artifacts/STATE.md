# COMMITTEE STATE — committee_naviC (live, self-loop)
Updated: 2026-05-30. Goal: 3 NEW theses at 6/6 GREEN.

## GREEN SCOREBOARD
- C* (when-NOT-to-use HW GPU CoW; measured negative result): **6/6 GREEN ✓** [1 of 3]
- A* (NVIDIA VMM 520K ceiling = vendor portability cliff): **6/6 GREEN ✓** [2 of 3] (R4 CC48 flipped)
  Holdout pieces: (1) NVIDIA mechanism — DONE (E-A2 independent repro 523,404 ≈ 520K, regime-
  sensitive 5637 single-handle). (2) AMD true ceiling — E-A1 RUNNING on devgpu499.
- B (prefix-injection workload model): 3-ENGINE COMPLETE (45 rows). Penalty k=0.67–0.79 R²≥0.97
  cross-engine reproduced; position ratio 2.2–2.8×. BUT absolute recompute ~1.3 << quadratic 2.0.
  Honest verdict: strong characterization, NOT a discovery → caps at YELLOW. NOT forcing it.

## EXPERIMENTS IN FLIGHT
- ROUND 4B (B final vote): LAUNCHED with full 3-engine data. penalty∝L^0.67/0.72/0.79 (R²≥0.97),
  but absolute recompute SUB-quadratic k≈1.3. Likely YELLOW (honest).
- E-E (Cluster E, 3rd-thesis candidate): sub-agent building isolation-primitive gating experiment
  on devgpu014. Tests if HW CoW gives a correctness/isolation primitive SW can't match, or collapses.
- [OLD-below]
- E-A1 (devgpu499): PARTIAL — reached 50M maps/191GiB, NO failure (96x NVIDIA), then node
  CRASHED (probe thrashed host). devgpu499 OFFLINE again. AMD divergence firmly established.
  Expect: either AMD_TRUE_CEILING_MAPPINGS (a real wall) or VA_RESERVE_FAIL (no mapping
  ceiling in reachable VA → proves AMD has no 520K-equivalent → flips CC48 → A* GREEN).
- E-B finish (devgpu014): SGLang + HF sweeps. logs/eb/finish_status.txt = EB_FINISH_DONE.

## ROUND 4 A* FINAL VOTE: LAUNCHED (votes/r4a_*.md) with both CC48 objections addressed
## (E-A2 independent 523,404 repro + E-A1 two-run AMD no-wall at 4M & 50M). Tally when done.

## NEXT STEPS (self-loop)
1. When E-A1 done → write A* mechanism+ceiling artifact → run final A* committee vote round.
   If 6/6 → A* GREEN [2 of 3].
2. When E-B finish done → assemble 3-engine CSV → cross-engine exponent agreement. Run B vote.
   B likely stays YELLOW (honest) — so we need a DIFFERENT 3rd thesis for the 3rd GREEN.
3. THIRD GREEN candidate: mine round-1 clusters D (cross-domain heterogeneous CoW) and E
   (write-after-fork exact isolation as a primitive). Both higher-ceiling. Pick the one with
   a clean H100-only gating experiment, build+run it, then vote.
   - Cluster E (Metric 5c bit-identical rollback isolation) is most built-already → cheapest.
4. Keep iterating rounds until 3× 6/6 GREEN or evidence says kill+replace.

## NODES: devgpu014 (H100) + devgpu499 (MI350X) both ONLINE. /tmp/agentenv.sh on both.
## Committee backends: claude-opus-4-8/4-7/4-6, metacode(Muse), codex, gemini. run_agent.sh.


## 2026-05-30 UPDATE: B=YELLOW(final), Cluster-E=KILLED (collapses to SW-equivalence).
## GREEN: C*, A* (2 of 3). Need a fresh 3rd candidate — see DEAD list, pick from round-1
## clusters NOT yet killed, or derive a new one from C*/A* measured data.

## ROUND 5: 3rd-GREEN candidate selection LAUNCHED. Candidates: I (unified A*+C* decision
## procedure), J (driver-handle physical provenance/attestation, from E-E residue), or agent's own.
## Committee picks the most GREEN-able; then run its gating experiment.

## ROUND 5 PICK: Candidate J (driver-handle provenance/attestation), 4/6 pick. Gating exp E-J:
## untrusted-runtime external verifier + adversarial bookkeeping-lie + the FATAL forge-resistance
## test (is the handle HW-rooted or a fabricable userspace int?). Building E-J now.

## E-J PRE-FLIGHT: Candidate J KILLED. cuMemRetainAllocationHandle == int(cuMemCreate handle),
## a process-local userspace integer, NOT hardware-attested. Fails CC48's forge-resistance test.
## Same death as E. Survivors for 3rd GREEN: I (decision procedure), Gemini throughput-collapse.

## 3rd-GREEN PIVOT: J killed (preflight). New target = T-TAX (Gemini's 'Contiguous-VA VMM
## Tax' = unified A*+C*+E architectural indictment). E-T experiment LAUNCHED (high-fanout
## throughput collapse, sub-agent 63f3d6c1). Resolves both I and T-TAX: does HW throughput-
## collapse at the K-ceiling while SW scales on, and is VMM win-region empty end-to-end?

## E-T IN PROGRESS (sub-agent 63f3d6c1): high-fanout throughput sweep building. EARLY SIGNAL:
## ARM-HW crashes NON-DETERMINISTICALLY below the 520K ceiling under real decode+CoW (branch 72/
## 193/ok@100; mixed cuMemSetAccess-OOM + illegal-access). Sub-agent isolating fork-only vs
## fork+decode+CoW to attribute (confound vs genuinely-lower effective ceiling w/ torch+model).
## At B=4 HW already 110 tok/s vs SW 185 tok/s (HW slower, consistent w/ C*). Healthy, debugging.

## E-T DATA COMPLETE (CSV 16 rows). T-TAX STRONGLY SUPPORTED end-to-end:
## - HW VMM CoW crashes at cuMemSetAccess for EVERY B>=128 (branch-at-crash variable 9-393),
##   throughput capped 68-136 tok/s, NEVER wins. SW: zero crashes thru B=1124, 280-800 tok/s
##   RISING with fanout. VMM-CoW win-region EMPTY end-to-end. SW wins at every B.
## - Caveat: one B=16 HW crash was transient cublas (not ceiling; 2/3 reps OK). Crash-branch
##   variability high (torch+CoW compete for descriptor budget) — itself part of indictment.
## Sub-agent finalizing ET_RESULT.md + chart. NEXT: run committee vote on T-TAX (candidate for GREEN #3).

## ROUND 6 (T-TAX FINAL VOTE) LAUNCHED — candidate for GREEN #3. Prompt gives both the
## strong case AND 4 counterarguments (A*+C*-restatement risk, single-layer confound, venue bar,
## crash non-reproducibility) so the vote is honest. Tally when done.

## ★★★ GOAL REACHED 2026-05-30: THREE 6/6 GREEN theses — C*, A*, T-TAX. FINAL_REPORT.md written. ★★★
