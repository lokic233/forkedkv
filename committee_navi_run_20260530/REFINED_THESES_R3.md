# ROUND 3 THESES — sharpened to kill the EXACT Round-2 dissents. Vote independently & hostile.
# GitHub artifacts = source of truth. GREEN = "the FRAMING survives hostile review AND has a
# defined, fundable validating experiment." GREEN does NOT require the experiment be already run;
# it requires that NO unfixable flaw remains and the experiment, if it hits its stated bar, makes
# this submission-grade. Vote RED only if a flaw is UNFIXABLE by the stated experiment.

## THESIS T1 (sharpened) — "Mapping-table entries are a schedulable accelerator resource"
One-sentence: GPU VMM exposes a per-context mapping-table whose entries are an exhaustible,
HBM-orthogonal, OS-independent resource (forensically: OOM at cuMemSetAccess, live HBM flat,
independent of vm.max_map_count), and the measured invariant branches x prefix_pages = K (~520K
on H100/580.82) is a capacity-planning model that any VMM-backed KV system (e.g. vAttention,
ASPLOS'25) must schedule against — a resource axis no current serving scheduler tracks.
- ROUND-2 DISSENT (4x YELLOW): "single-stack; cross-vendor unrun; vAttention bindingness asserted."
- SHARPENED CLAIM: The contribution is the EXISTENCE + FORENSIC ATTRIBUTION + ACCOUNTING MODEL of
  a new resource axis, NOT a universality claim. We explicitly state K's VALUE is stack-specific;
  the LAW we claim is "this resource exists, is orthogonal to HBM/OS limits, and follows
  branches x pages = const within a stack." The cross-vendor sweep tests whether the FORM
  (multiplicative invariant) generalizes; even if K differs per stack, a per-stack-calibrated
  schedulable resource is the contribution. vAttention is cited as an EXISTENCE PROOF that
  deployed VMM-KV systems exist; we do NOT claim it currently saturates K — we claim any such
  system that branches/shares heavily WILL, and none can currently see the resource.
- GREEN bar: cross-vendor sweep (>=2 of A100/L40S/MI300X) confirms the multiplicative-invariant
  FORM holds (per-stack K), AND a microbench shows a byte/HBM-only scheduler cannot predict the
  OOM that a K/P-aware one can. RED only if the invariant FORM fails to reproduce on any 2nd stack.
- Anti-FlashInfer: YES. Contribution type: characterization + capacity abstraction. Venue: ASPLOS/ATC.

## THESIS T2 (sharpened) — "Mid-prompt cache invalidation: live-engine quantification + bounded repair"
One-sentence: We do not claim prefix-cache invalidation is undiscovered; we claim the FIRST
live-engine quantification of tool-call mid-prompt invalidation (8.21x TTFT, vLLM 0.6.6 / Qwen2.5-7B
/ H100), a POSITION-SHIFT TAXONOMY of edits (append-boundary vs interior-insert vs replace), and a
microsecond VMM pointer-remap that PROVABLY repairs the RoPE-invariant subclass (recovery 1.17x; 58us
vs 274ms) WITH the position-shifting subclass's residual recompute cost measured, not hand-waved.
- ROUND-2 DISSENT: metacode "not a new class"; gemini "RoPE-safe scoping cripples utility."
- SHARPENED CLAIM (kills metacode): novelty is NOT the phenomenon — it is (a) the first LIVE-ENGINE
  magnitude (8.21x is not in any public benchmark), (b) the edit-position taxonomy mapping each edit
  type to its exact recompute cost, (c) the repair primitive for the invariant subclass. These are
  three measured artifacts, not a relabel.
- SHARPENED CLAIM (kills gemini "cripples utility"): we QUANTIFY the split. The experiment measures
  what FRACTION of real tool-call edits fall in the RoPE-invariant subclass (fixed-width/boundary
  results) vs position-shifting, on real SWE-bench/tool traces. If the invariant subclass is a small
  fraction, that is itself a PUBLISHABLE characterization result (the pathology + its true addressable
  surface), and the repair is honestly scoped. Utility is MEASURED, so "cripples" is testable, not fatal.
- GREEN bar: >=2 engines (vLLM+SGLang) show >5x penalty; edit taxonomy with per-class recompute cost;
  repair <=1.3x baseline on invariant subclass; addressable-fraction reported on real traces.
- Anti-FlashInfer: YES (prefill recompute + weight reads). Contribution: workload-model + primitive. Venue: MLSys/NSDI.

## THESIS T3-NEW (replaces dead speculative-prefill) — "Agentic KV-divergence workload model"
One-sentence: Agentic LLM execution has a characterizable, low-and-right-skewed KV-divergence
structure (branching factor, divergence depth, tail-divergence fraction) that determines memory-
sharing opportunity, and this divergence-distribution workload model — absent from all current
KV-cache research, which assumes static shared prefixes — is what should parameterize KV-sharing
system design and admission control.
- WHY IT REPLACES OLD T3: old T3 (speculative prefill) died 3R because it needs an infeasible
  exact-token predictor and "request-idle != GPU-idle under batching." This thesis makes NO predictor
  claim and NO idle-window claim. It is a pure measurement/characterization of the divergence
  distribution that all the OTHER theses implicitly assume.
- Evidence: forkedkv M2b (95%/90% fewer bytes at 5%/10% TAIL divergence = a direct divergence->savings
  curve); M5 (24 real SWE-bench-Verified, 90% fewer KV bytes, 80% lower peak HBM => real divergence is
  low + right-skewed); M5b confirms branches stay largely shared at full 28-layer depth.
- Closest prior work + delta: ChunkAttention/RadixAttention assume STATIC shared prefixes; no prior
  work models the DYNAMIC tree-structured divergence distribution of agent forks that makes sharing
  profitable, nor uses it to size allocators a priori.
- Fatal-flaw candidates + why fixable: "n=24 is a toy sample / harness-specific" -> the GREEN bar is
  >=3 harnesses (SWE-bench + web-nav + multi-tool) with an out-of-sample fit, so generalization is
  tested not assumed.
- Anti-FlashInfer: YES (divergence governs capacity/sharing, not kernel speed).
- No-code test: the divergence-distribution -> sharing-opportunity formula + the empirical
  low-divergence regime of coding agents survive as a durable workload characterization.
- GREEN bar: extract per-step KV divergence distributions across >=3 agent harnesses, fit a
  sharing-opportunity model; success iff it predicts measured byte savings within 15% out-of-sample.
- Contribution: workload-model. Venue: MLSys.

---
VOTE: output EXACTLY 3 lines, hostile, naming any UNFIXABLE flaw:
T1: <GREEN|YELLOW|RED> — <=2 lines
T2: <GREEN|YELLOW|RED> — <=2 lines
T3: <GREEN|YELLOW|RED> — <=2 lines
