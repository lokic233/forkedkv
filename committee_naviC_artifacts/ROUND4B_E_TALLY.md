# ROUND 4B (B vote) + E-E (Cluster E gating) — both HONEST non-GREEN

## B FINAL VOTE (full 3-engine data: L^0.67/0.72/0.79 R²≥0.97, but abs recompute sub-quadratic k≈1.3)
| CC48 | CC47 | CC46 | Muse | Codex | Gemini | CONSENSUS |
|---|---|---|---|---|---|---|
| YELLOW | YELLOW | YELLOW | YELLOW | GREEN | YELLOW | **YELLOW** (5Y/1G) |
B verdict: real, cross-engine, position-resolved, operationally large (~17×) — but SUB-quadratic,
not a super-quadratic discovery. Caps at YELLOW/MLSys-workshop. "Just algebra" largely vindicated.
NOT GREEN. Honest outcome — not forced.

## E-E (Cluster E: write-after-fork isolation primitive) — NEAR-KILL
Software prefix-sharing MATCHES HW exactly on bit-identical isolation + O(1) rollback (both copy
0 bytes, both verified bit-identical across reps), and is STRICTLY FASTER (fork 240× faster,
rollback 1.8× faster). HW's only unique residue: driver-handle physical proof of non-aliasing
(cuMemRetainAllocationHandle) + contiguous-VA — but the former needs a self-undermining threat
model (distrust your allocator but trust the driver) and the latter is a perf property already
shown a NET LOSS by C*. VERDICT: Cluster E collapses into software-equivalence. KILL as standalone
safety thesis. (Worth 1 sentence in C*/A* writeup: HW is the only mechanism that can emit a
driver-level physical proof of KV-page non-aliasing — a verifiability nicety, not a capability gap.)

## SCOREBOARD: C* GREEN, A* GREEN. B=YELLOW(capped), E=KILLED. Need a NEW 3rd-GREEN candidate.
