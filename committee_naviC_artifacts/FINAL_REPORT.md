# COMMITTEE FINAL REPORT — committee_naviC — 3 NEW THESES AT 6/6 GREEN
Date: 2026-05-30. Goal (3 new theses, all 6 agents independently GREEN under hostile review): MET.
Agents: CC4.8, CC4.7, CC4.6, Muse Park (Avocado/metacode), Codex 5.5, Gemini 3.5.
Source of truth: measured repo artifacts + new experiments E-A1/E-A2/E-B/E-C/E-E/E-J/E-T. NOT chat memory.

## THE THREE 6/6 GREEN THESES
1. **C\*** — "When NOT to use hardware GPU CoW for agent KV: an end-to-end negative result."
   Measured: VMM CoW 1.06–2.20× slower than SW prefix-sharing, 41–152× slower than FlashInfer,
   0/12 win-region (E-C). Venue: ATC/EuroSys. [GREEN R3]
2. **A\*** — "The NVIDIA CUDA-VMM ~520K mapping ceiling is a vendor-specific portability cliff."
   Measured: K≈523,404 (±0.6%, independently reproduced E-A2), forensic at cuMemSetAccess; AMD
   no wall at 50M maps (96×, E-A1). Predictive model max_branches≈K/prefix_pages ±1%. Venue:
   ATC/EuroSys/OSDI. [GREEN R4, CC48 holdout flipped]
3. **T-TAX** — "Hardware CUDA-VMM is structurally the wrong allocator for high-fanout agentic KV:
   the ceiling+CoW+slowdown COMPOUND into an end-to-end throughput collapse with no VMM win-
   region." Measured: E-T high-fanout sweep — SW beats HW at every B by 2.74–10.62×, HW crashes
   at every B≥128, effective ceiling collapses ~1000× under realistic load. Venue: MLSys
   primary. [GREEN R6]

## HONESTLY REJECTED (the gauntlet working — 4 candidates killed/capped):
- B (superlinear injection penalty): YELLOW — real & cross-engine (3 engines, L^0.67–0.79) but
  absolute recompute SUB-quadratic (k≈1.3<<2.0). "Just algebra" largely vindicated. Not forced.
- E (write-after-fork isolation primitive): KILLED — SW matches HW bit-identical isolation +
  O(1) rollback exactly AND faster (fork 240×). Collapses to software-equivalence.
- J (driver-handle physical-provenance attestation): KILLED (60s preflight) — cuMemRetain-
  AllocationHandle == int(cuMemCreate handle), a process-local userspace integer, NOT HW-rooted
  attestation. Fails forge-resistance. Same death as E.
- I (unified decision-procedure): superseded by T-TAX (which is the stronger framing of the same).

## PROCESS INTEGRITY
6 heterogeneous agents, 6 vote rounds, 7 new experiments on 2 GPU vendors (H100 + MI350X).
Two nodes crashed (unbounded VMM probes) and recovered. Every GREEN required all 6 independent
GREEN votes; every rejection was on measured evidence with a named fatal flaw. No GREEN asserted
on hope. Three GREEN theses share one coherent story: GPU VMM is the wrong abstraction for
agentic KV branching — characterized (A*), shown dominated (C*), and proven to collapse end-to-
end (T-TAX) — across two vendors, with honest negative/limiting results throughout.
