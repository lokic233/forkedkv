# REFINED THESES (Round 2) — each rewritten to neutralize the EXACT objections
# raised in Round 1. Vote independently. GitHub artifacts are source of truth.

These 3 are the survivors of Round 1 clustering, refined to address every recorded
RED/YELLOW dissent. Your job: vote GREEN/YELLOW/RED on EACH, independently and hostilely.

---

## THESIS T1 — "The GPU VMM Mapping-Table Wall"
One-sentence: Per-context GPU VMM mapping-table entries are a distinct, exhaustible,
OS-independent accelerator resource governed by an invariant capacity law
(branches x prefix_pages = K ~= 520K on H100), and ANY VMM-backed KV system — including the
already-deployed vAttention (ASPLOS'25) — must account for and schedule against this resource,
which no current serving scheduler (vLLM/SGLang/TRT-LLM) tracks.
- Contribution type: characterization + capacity-planning abstraction
- Evidence: forkedkv M4b (K=520K invariant within 1%), M4 (OOM call site = cuMemSetAccess,
  live HBM flat at 12GiB), Lab1 (independent of vm.max_map_count: 392 VMAs vs 67M).
- OBJECTION NEUTRALIZED (Round 1, claude48 YELLOW): "R4 says software fork is 700x faster, so
  who ever hits this ceiling?" -> The ceiling binds for vAttention, a PUBLISHED + DEPLOYED
  ASPLOS'25 system that uses CUDA VMM for KV serving. The characterization is about the resource
  class, portable to every VMM-based design, NOT about forkedkv's retracted fork mechanism.
- OBJECTION NEUTRALIZED (all agents, "vendor-specific/ephemeral"): claim is scoped as a
  measured law on H100/580.82 WITH a cross-vendor reproduction experiment (A100/L40S/MI300X)
  as the GREEN-gating experiment, not assumed universal.
- Anti-FlashInfer: YES (metadata exhaustion, not compute).
- Highest-EV experiment: reproduce K·P invariant on >=2 additional GPU/driver stacks; GREEN
  iff invariant holds within +-15% on >=2 stacks (=> cross-arch law) OR is cleanly explained
  as a driver-config constant (still a schedulable resource).
- Venue: ASPLOS (fallback EuroSys/ATC).

## THESIS T2 — "Mid-Prompt Cache Invalidation is a First-Class Serving Pathology"
One-sentence: Tool-call mid-prompt injection is a previously-uncharacterized cache-invalidation
event class that inflicts a measured 8.21x live-engine TTFT penalty (vLLM 0.6.6, Qwen2.5-7B,
H100) because prefix-hash chains assume append-only growth, and microsecond-scale VMM pointer
remap repairs it (recovery to 1.17x; 58us swap vs 274ms recompute) for the RoPE-safe edit class.
- Contribution type: workload-model + runtime-primitive
- Evidence: edmm live vLLM B/A=8.21x, C/A=1.17x; cuMemMap 58us vs recompute 274ms; standalone
  4.18x->0.96x; vLLM fork integration +14 lines/31 tests.
- OBJECTION NEUTRALIZED (claude48/codex/gemini, RoPE): we DO NOT claim arbitrary insertion is
  free. We scope the repairable class to edits where downstream RoPE positions are preserved
  (append-at-boundary / fixed-width tool-result slots), AND require the experiment to MEASURE the
  RoPE re-rotation cost for position-shifting edits so the boundary is quantified, not assumed.
  The 8.21x WORKLOAD PATHOLOGY stands regardless of the repair's edit-class breadth.
- OBJECTION NEUTRALIZED ("just a benchmark"): paired with the repair primitive, not dataset-only.
- Anti-FlashInfer: YES (penalty is prefill recompute + weight reads, not attention-kernel speed).
- Highest-EV experiment: replay real tool-call traces through vLLM+SGLang, inject at 5 positions,
  measure TTFT + RoPE re-rotation cost by edit type; GREEN iff (a) >=2 engines show >5x penalty
  AND (b) repair holds <=1.3x baseline on the RoPE-safe class with the unsafe class cost bounded.
- Venue: MLSys / NSDI.

## THESIS T3 — "Tool-Idle Windows are Reclaimable GPU Time" (speculative prefill scheduling)
One-sentence: External tool-call latency (100ms-2s) is a structurally-reclaimable GPU idle
window distinct from chat workloads, and speculative KV prefill scheduled into it — recovered
via 58us VMM pointer-swap on a hit, discarded at near-zero HBM on a miss — converts the measured
8.21x invalidation penalty into hidden latency when a predictor clears a stated break-even.
- Contribution type: mechanism + scheduling optimization
- Evidence: edmm 8.21x penalty + 58us swap + 1.17x recovery (the idle window and cheap
  recovery are both measured); forkedkv M2b (95%/90% fewer bytes => discarding wrong
  speculations is near-free) + M3 (~0% attn overhead on forked branch).
- OBJECTION NEUTRALIZED (claude46/codex/gemini, "tool returns unpredictable"): the thesis is
  gated on an EXPLICIT break-even model (remap cost, recompute cost, predictor accuracy) and
  scoped to structured/schema-known tool returns (SQL, typed APIs, file-read) where hit rate is
  measurable; we do NOT claim it works for arbitrary web scraping.
- OBJECTION NEUTRALIZED (mapping-budget contention w/ T1): speculative branches are bounded by
  the K/P budget from T1 — the two theses compose into a single admission policy.
- Anti-FlashInfer: YES (latency hiding during I/O wait; kernel speed doesn't shorten network RTT).
- Highest-EV experiment: SWE-bench trace replay with real tool latencies, 3-way speculative
  prefill on structured-API tasks; GREEN iff >=1.5x E2E TTFT reduction vs EDMM-repair baseline at
  >=25% predictor hit rate AND <10% wasted prefill.
- Venue: EuroSys/ATC / MLSys.

---
VOTE INSTRUCTIONS: For T1, T2, T3 output EXACTLY:
T1: <GREEN|YELLOW|RED> — <=2 lines why, citing the specific surviving flaw or why it's neutralized.
T2: <GREEN|YELLOW|RED> — ...
T3: <GREEN|YELLOW|RED> — ...
Be hostile. A YELLOW/RED must name the precise unfixed flaw. Output ONLY the 3 vote lines.
