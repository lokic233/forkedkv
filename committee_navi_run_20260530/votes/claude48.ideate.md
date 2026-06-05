THESIS 1:
- One-sentence thesis: The cost of every CUDA-VMM KV operation (fork-CoW, mid-sequence insert, cache growth) is governed by a single driver primitive — `cuMemSetAccess`/`cuMemMap` map-op latency — not by data movement, and a crossover model over (map-op cost, bytes-otherwise-moved, mapping granularity) predicts when VMM beats memcpy/recompute for any KV system.
- Repo it builds on: cross-repo
- Contribution type: characterization
- Supporting evidence (cite specific repo metric/result): forkedkv CoW cost ~178us/page with D2D copy only **7%** (map-op bound); fork latency linear, map floor ~50us/page (M1); edmm cuMemMap **58us** vs memcpy+recompute **202us** — two independent repos confirm the map-op, not the copy, is the cost center.
- Closest prior work + the delta: vAttention (ASPLOS'25) uses VMM for read-only KV growth and asserts mapping is cheap; delta = it never models the map-op as the *scaling wall* nor gives a crossover rule for mutation (fork/insert) vs byte movement. This is the missing decision model for the whole VMM-KV class.
- Closest OSS competitor + why it doesn't kill this: vLLM/PagedAttention block allocator avoids the driver entirely (software refcount) — but precisely because it can't characterize the driver, it can't tell an implementer *when* the contiguous-VA VMM path is worth the map-op tax; the model does.
- Fatal flaw (the strongest reason a reviewer kills it): "This is microbenchmarking one vendor's driver call; NVIDIA can change `cuMemSetAccess` cost in a driver bump and your crossover constants evaporate."
- Anti-FlashInfer test: YES — map-op cost is a memory-bandwidth/driver-lock quantity; infinitely fast attention kernels don't move pages or take the driver mapping path.
- No-code test: The crossover inequality (VMM wins iff bytes_moved/BW > map_ops × t_map) and the "7% copy / 93% map" breakdown are durable engineering laws.
- Highest-EV experiment: Sweep map-op latency vs granularity (2MiB→64KiB) and vs concurrent mapping streams on H100; success = a closed-form crossover that correctly classifies forkedkv-M5 and edmm-micro regimes from first principles.
- Realistic venue: MLSys
- Your self-vote: YELLOW — strong cross-repo evidence and clean anti-FlashInfer pass, but vendor-specificity caps the ambition below ASPLOS.

THESIS 2:
- One-sentence thesis: VA-mapping-table entries are a distinct, exhaustible serving resource — invisible to byte/KV-block accounting — and the measured invariant branches×prefix_pages ≈ const (~520K) is a capacity-planning law that should drive admission control for any fork/share-based LLM server.
- Repo it builds on: forkedkv
- Contribution type: abstraction
- Supporting evidence (cite specific repo metric/result): M4b K/P invariant (branches×prefix_pages ≈ **520K** constant); M4 OOM forensically at `cuMemSetAccess` = VA-mapping-metadata exhaustion, NOT data HBM (live HBM flat at 12GiB); Lab1 ceiling independent of `vm.max_map_count` (392 VMAs vs 67M limit) — proves a pure driver mapping-table resource orthogonal to bytes.
- Closest prior work + the delta: vLLM/RadixAttention schedulers admit on KV-block/HBM-byte budgets only; delta = a second resource dimension (mapping-table entries) with its own conservation law, making admission a 2D constraint the moment VMM sharing is used.
- Closest OSS competitor + why it doesn't kill this: RadixAttention never allocates VA mappings, so it appears immune — but the moment any system adopts contiguous-VA sharing (vAttention, EDMM) to dodge the PagedAttention tax, it inherits the 520K ceiling with no scheduler that models it; this fills that gap.
- Fatal flaw (the strongest reason a reviewer kills it): "It only binds if you've already chosen a losing mechanism — repo's own R4 says software fork is ~700x faster and ~6x higher capacity, so why would anyone hit this ceiling in production?"
- Anti-FlashInfer test: YES — a mapping-table-entry limit is structural; infinite kernel speed gives you zero additional table entries.
- No-code test: The K/P conservation law and "mapping-table entries are a schedulable resource" abstraction survive as a planning model.
- Highest-EV experiment: Drive a VMM-sharing server to OOM under mixed branch/prefix mixes and show a K/P-based admission controller raises sustained branch count vs byte-only admission; success = ≥20% higher admitted concurrency at zero OOM.
- Realistic venue: EuroSys/ATC
- Your self-vote: YELLOW — the law is real and anti-FlashInfer-clean, but the R4 retraction means the resource may never bind in practice; needs a deployment regime where contiguity is mandatory to be GREEN.

THESIS 3:
- One-sentence thesis: Real agent execution has a characterizable KV-divergence structure (branching factor, divergence depth, tail-divergence fraction) that determines memory-sharing opportunity, and this workload model — not static-prefix assumptions — should parameterize KV-sharing system design.
- Repo it builds on: forkedkv
- Contribution type: workload-model
- Supporting evidence (cite specific repo metric/result): M2b bytes-written drops **95%/90%** at **5%/10%** tail divergence (a direct divergence→savings curve); M5 on 24 real SWE-bench-Verified instances = 90% fewer KV bytes, 80% lower peak HBM — empirical divergence is low and right-skewed.
- Closest prior work + the delta: ChunkAttention/RadixAttention assume *static* shared prefixes (shared system prompt, few-shot); delta = agentic workloads share *dynamically* via tree-structured forks, and no prior work models the divergence distribution that makes this profitable.
- Closest OSS competitor + why it doesn't kill this: SGLang RadixAttention exploits prefix sharing but treats it as opportunistic dedup with no workload model; it can't predict savings or size allocators a priori — a divergence model can.
- Fatal flaw (the strongest reason a reviewer kills it): "n=24 SWE-bench instances is a toy sample; divergence structure may be agent-harness-specific and won't generalize across frameworks/tasks."
- Anti-FlashInfer test: YES — divergence governs memory capacity/sharing, which infinite attention speed cannot manufacture.
- No-code test: The divergence-distribution → sharing-opportunity formula and measured low-divergence regime of coding agents survive as a workload characterization.
- Highest-EV experiment: Extract per-step KV divergence distributions across ≥3 agent harnesses (SWE-bench, web-nav, multi-tool) and fit a sharing-opportunity model; success = model predicts measured byte savings within 15% out-of-sample.
- Realistic venue: MLSys
- Your self-vote: YELLOW — genuinely under-characterized and clean on both tests; only the thin sample and generalization risk keep it from GREEN.

THESIS 4:
- One-sentence thesis: Content-addressed (hash-chain) prefix caches are structurally brittle to mid-sequence context mutation — the dominant access pattern of agentic workloads — and an insertion-robust KV index is the correct mechanism, not per-incident pointer-swap patching.
- Repo it builds on: edmm
- Contribution type: mechanism
- Supporting evidence (cite specific repo metric/result): edmm measured **8.21x** TTFT penalty through *live* vLLM 0.6.6 when a tool-call injection breaks the prefix hash chain (Radix B/A=8.21x); recovery to 1.17x via pointer-swap shows the loss is purely index-structural, not compute-fundamental.
- Closest prior work + the delta: RadixAttention's radix tree handles prefix *extension* but invalidates on *insertion* (hash chain breaks downstream); delta = an index whose keys are position-stable under mid-sequence edits, generalizing edmm's single-case swap into a cache-structure contribution.
- Closest OSS competitor + why it doesn't kill this: LMCache/RadixAttention both key on contiguous-prefix hashes and recompute on insertion; neither offers insertion-stable keying, so the 8.21x cliff is latent in all of them.
- Fatal flaw (the strongest reason a reviewer kills it): "RoPE makes K position-dependent — you cannot make the index position-invariant without recomputing the rotated keys, so the mechanism collapses to recompute for the shifted suffix anyway."
- Anti-FlashInfer test: YES (partial) — recompute cost is prefill GEMM/FFN-bound, which FlashInfer (attention only) does not eliminate; infinite attention speed still leaves the FFN recompute the insertion forces.
- No-code test: The "hash-chain caches have an insertion cliff; agentic mutation is insertion-dominated" taxonomy survives even if the fix doesn't.
- Highest-EV experiment: Measure recompute fraction attributable to FFN vs attention on insertion-invalidated prefixes; success = FFN ≥40% of the penalty (proving anti-FlashInfer survival) AND a position-stable index recovers ≥2x of the cliff.
- Realistic venue: EuroSys/ATC (Workshop if the index mechanism fails the RoPE test)
- Your self-vote: YELLOW — the brittleness characterization is solid and edmm-backed, but the RoPE fatal flaw could demote the *mechanism* half to a recompute-the-suffix triviality.

THESIS 5:
- One-sentence thesis: The VMM map-op is not just per-op expensive but driver-serialized per context, making `cuMemSetAccess` a multi-tenant *throughput* ceiling — a concurrency wall distinct from the 520K capacity ceiling — for any server doing concurrent fork/insert/remap.
- Repo it builds on: cross-repo (forkedkv map-op + edmm live-server integration)
- Contribution type: characterization
- Supporting evidence (cite specific repo metric/result): forkedkv 178us/page map-bound with linear (not flat) fork latency (M1) and a *per-context* mapping table (M4b 520K/context) — both point to a shared driver structure; edmm integrated into live vLLM (+14 lines, 31 tests) where concurrent requests would contend that same path.
- Closest prior work + the delta: All VMM-KV work (vAttention, forkedkv, edmm) reports *single-stream* map latency; delta = the contention/serialization behavior under N concurrent mappers, which determines real multi-tenant viability and is currently unmeasured.
- Closest OSS competitor + why it doesn't kill this: vLLM software allocators are lock-light and parallel; this characterization is exactly what tells you the VMM path's concurrency penalty vs that baseline — competitor existence motivates, not kills.
- Fatal flaw (the strongest reason a reviewer kills it): "This is an unmeasured extrapolation — neither repo tested concurrent mapping, so it may be a non-effect if the driver maps lock-free per context."
- Anti-FlashInfer test: YES — driver lock contention is independent of attention-kernel speed.
- No-code test: If real, "VMM KV management has a per-context map-op serialization ceiling" is a durable systems constraint guiding architecture (per-context sharding).
- Highest-EV experiment: Scale concurrent `cuMemSetAccess` issuers 1→64 on H100 and measure aggregate map-op throughput; success = sub-linear scaling (>2x degradation at 16 streams) confirming a serialization wall, else KILL.
- Realistic venue: Workshop (MLSys if the wall is sharp and a per-context sharding fix is shown)
- Your self-vote: RED — highest-risk of the five: the core claim is currently unmeasured in either repo, so until the concurrency experiment lands it's a hypothesis, not a thesis.
