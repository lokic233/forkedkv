THESIS 1:
- One-sentence thesis: GPU KV-serving systems need admission control over driver VA-map entries, because branch capacity is bounded by mapping-table entries rather than HBM bytes.
- Repo it builds on: forkedkv
- Contribution type: workload-model
- Supporting evidence (cite specific repo metric/result): forkedkv M4b shows `branches x prefix_pages ~= 520K`; Lab1 shows independence from `vm.max_map_count`; M4 OOM occurs at `cuMemSetAccess`, not data allocation.
- Closest prior work + the delta: vAttention uses CUDA VMM for KV serving, but does not characterize the per-context map-entry ceiling as a schedulable resource under branching.
- Closest OSS competitor + why it doesn't kill this: vLLM/SGLang avoid this driver ceiling with software paging, but that is exactly the contrast: the thesis defines when VMM-backed systems fail and how to schedule around it.
- Fatal flaw (the strongest reason a reviewer kills it): It may be a one-GPU, one-driver artifact unless reproduced across A100/H100/B200, MIG, CUDA versions, and multiple VMM users.
- Anti-FlashInfer test (would it matter if FlashInfer were infinitely fast? YES + why): YES, because this is about driver metadata exhaustion before attention kernels run.
- No-code test (if the impl vanished, what knowledge remains?): A validated `K/P` capacity law for GPU VMM KV systems and a failure taxonomy separating HBM, VA, and map-entry exhaustion.
- Highest-EV experiment (1 sentence: what to run, min success criterion): Sweep page size, prefix length, branch count, CUDA version, and GPU generation; success requires the map-entry model to predict OOM within 10% on at least 3 GPU/config pairs.
- Realistic venue: ASPLOS
- Your self-vote: GREEN (+ Strong if generalized beyond one H100; otherwise it collapses into an artifact note.)

THESIS 2:
- One-sentence thesis: Mid-prompt KV repair is a distinct serving primitive: pointer-remapping can preserve reusable KV around tool-injected spans where prefix-hash caches force recomputation.
- Repo it builds on: edmm
- Contribution type: runtime-primitive
- Supporting evidence (cite specific repo metric/result): edmm measures vLLM live Radix penalty `B/A=8.21x`, EDMM recovery `C/A=1.17x`, and CUDA VMM `cuMemMap` at `58us` versus recompute around `274ms`.
- Closest prior work + the delta: PagedAttention/RadixAttention/ChunkAttention optimize append-style prefix reuse; the delta is repair after interior prompt mutation caused by tool-call injection.
- Closest OSS competitor + why it doesn't kill this: SGLang RadixCache is the strongest OSS baseline, but edmm reports the same hash-chain breakage class and demonstrates VMM repair instead of rebuilding the chain.
- Fatal flaw (the strongest reason a reviewer kills it): Correctness may be narrow if real models, tokenizers, RoPE positions, batching, and attention masks make KV splicing valid only for a small mutation class.
- Anti-FlashInfer test (would it matter if FlashInfer were infinitely fast? YES + why): YES, because infinitely fast decode kernels do not remove the recompute caused by invalidated prefix-cache hashes.
- No-code test (if the impl vanished, what knowledge remains?): A formal taxonomy of prompt edits that invalidate prefix caches but admit KV-preserving repair.
- Highest-EV experiment (1 sentence: what to run, min success criterion): Replay real tool-calling traces through vLLM/SGLang with interior injection frequencies measured, and require median TTFT within 1.3x of no-injection baseline on at least two models.
- Realistic venue: MLSys
- Your self-vote: GREEN (+ The measured 8.21x live penalty and 1.17x recovery are a clean systems result if the edit class is broad enough.)

THESIS 3:
- One-sentence thesis: Contiguous virtual KV sharing is an attention-kernel compatibility contract, not a capacity trick: it lets shared physical prefixes run through unmodified dense-attention kernels.
- Repo it builds on: forkedkv
- Contribution type: abstraction
- Supporting evidence (cite specific repo metric/result): forkedkv M3 shows `-0.1%` to `+1.1%` attention overhead with unmodified FlashAttention/SDPA; M5b shows Qwen2.5-7B 28-layer decode is bit-identical CoW vs clone with `44%` lower peak HBM.
- Closest prior work + the delta: PagedAttention/vLLM and RadixAttention expose non-contiguous software pages and need paged kernels; the delta is preserving the dense contiguous-VA ABI while physically sharing KV.
- Closest OSS competitor + why it doesn't kill this: FlashInfer kills many performance claims, but it does not make arbitrary dense kernels, vendor kernels, or research kernels page-aware.
- Fatal flaw (the strongest reason a reviewer kills it): Reviewers may say this is an implementation convenience, not a research contribution, unless shown across enough kernels and serving stacks.
- Anti-FlashInfer test (would it matter if FlashInfer were infinitely fast? YES + why): YES, because the claim is compatibility with unmodified dense kernels, not beating a specific paged-attention kernel.
- No-code test (if the impl vanished, what knowledge remains?): A compatibility boundary showing which KV-sharing designs preserve dense-kernel semantics and which require kernel rewrites.
- Highest-EV experiment (1 sentence: what to run, min success criterion): Test contiguous-VA sharing across SDPA, FlashAttention, FlashInfer dense mode, and one vendor/library kernel; success requires bit-identical output and under 2% overhead for shared prefixes.
- Realistic venue: EuroSys/ATC
- Your self-vote: YELLOW (+ Defensible as an abstraction paper, but too close to forkedkv unless generalized beyond its prototype.)

THESIS 4:
- One-sentence thesis: Agentic LLM serving should schedule speculative prefill against tool-call idle windows, treating external tool latency as reclaimable GPU time rather than dead time.
- Repo it builds on: edmm
- Contribution type: optimization
- Supporting evidence (cite specific repo metric/result): edmm combines tool-call speculative prefill with VMM pointer-swap repair and reports vLLM live recovery from `8.21x` TTFT penalty to `1.17x`.
- Closest prior work + the delta: Speculative decoding predicts future tokens during model execution; this predicts and prepares future KV state during non-model tool latency.
- Closest OSS competitor + why it doesn't kill this: vLLM/SGLang schedulers batch model work, but they do not expose tool-idle-aware prefill scheduling with KV repair for mid-prompt injection.
- Fatal flaw (the strongest reason a reviewer kills it): If real tool latency is short, highly variable, or dominated by dependency uncertainty, speculation wastes GPU cycles and harms tail latency.
- Anti-FlashInfer test (would it matter if FlashInfer were infinitely fast? YES + why): YES, because the idle-window scheduling opportunity exists outside attention-kernel speed.
- No-code test (if the impl vanished, what knowledge remains?): A workload model for tool-call idle windows and a decision rule for when speculative prefill is profitable.
- Highest-EV experiment (1 sentence: what to run, min success criterion): Measure tool-call idle distributions on real agent traces and run a scheduler replay; success requires at least 30% TTFT reduction at under 10% wasted GPU prefill work.
- Realistic venue: MLSys
- Your self-vote: YELLOW (+ Needs real trace evidence; current repo proves mechanism recovery, not workload prevalence.)

THESIS 5:
- One-sentence thesis: CUDA VMM remap latency, not memory bandwidth, is the limiting cost center for dynamic KV memory systems, so serving runtimes need map-operation batching and layout selection.
- Repo it builds on: forkedkv
- Contribution type: optimization
- Supporting evidence (cite specific repo metric/result): forkedkv reports CoW cost around `178us/page`, with D2D copy only `7%`; fork latency is linear with a real map-op floor near `50us/page`.
- Closest prior work + the delta: vAttention and forkedkv use VMM, but the delta is optimizing the map-operation path as the dominant primitive rather than optimizing KV copy volume or attention kernels.
- Closest OSS competitor + why it doesn't kill this: vLLM/SGLang avoid driver remaps by software paging, but any VMM-based KV system still pays this remap cost when it edits mappings.
- Fatal flaw (the strongest reason a reviewer kills it): NVIDIA driver internals may make batching impossible or non-portable, reducing this to layout heuristics with modest gains.
- Anti-FlashInfer test (would it matter if FlashInfer were infinitely fast? YES + why): YES, because the measured bottleneck is `cuMemSetAccess`/unmap/remap latency, not attention execution.
- No-code test (if the impl vanished, what knowledge remains?): A cost model decomposing VMM KV updates into map latency, access-setting latency, and copy bandwidth, with layout rules for minimizing remaps.
- Highest-EV experiment (1 sentence: what to run, min success criterion): Implement batched remap/layout grouping for fork and CoW paths; success requires at least 2x lower remap-dominated latency without increasing peak HBM by more than 10%.
- Realistic venue: EuroSys/ATC
- Your self-vote: RED (+ Too dependent on opaque driver behavior unless it discovers a portable batching primitive or a strong cross-GPU law.)
