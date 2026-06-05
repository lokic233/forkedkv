THESIS 1:
- One-sentence thesis: Mid-prompt tool-call injections in agentic workloads break exact-match prefix caches, but logical sequence continuity can be restored physically via GPU VMM pointer-swapping to avoid catastrophic KV recomputation.
- Repo it builds on: edmm
- Contribution type: mechanism
- Supporting evidence (cite specific repo metric/result): vLLM live TTFT penalty of 8.21x recovered to 1.17x via 58us cuMemMap pointer-swap (vs 274ms memcpy+recompute).
- Closest prior work + the delta: RadixAttention/SGLang rely on exact prefix hash matching from token 0; this mechanism survives mid-prompt mutations by stitching physical memory underneath the virtual address space.
- Closest OSS competitor + why it doesn't kill this: SGLang. Even if they implement fuzzy chunk-matching, EDMM's 58us pointer-swap avoids memory movement entirely, operating below the attention kernel layer.
- Fatal flaw (the strongest reason a reviewer kills it): If the tool output size alters the sequence length dynamically, rotary positional embeddings (RoPE) for all subsequent tokens must be recalculated anyway, negating the zero-copy swap benefit.
- Anti-FlashInfer test (would it matter if FlashInfer were infinitely fast? YES/NO + why): YES. FlashInfer speeds up attention during decode, not the TTFT KV prefill recomputation (the 274ms penalty) caused by prefix invalidation.
- No-code test (if the impl vanished, what knowledge remains?): The vulnerability of exact-match hash chains to mid-prompt injection in agent workloads, and the timing physics of VMM pointer-swap vs recompute.
- Highest-EV experiment (1 sentence: what to run, min success criterion): Profile end-to-end multi-turn agent execution with varying tool-output lengths to measure RoPE recalculation overhead versus VMM swap savings.
- Realistic venue: MLSys
- Your self-vote: GREEN (Solves a real, measured pain point in vLLM with an elegant, quantified OS-level mechanism).

THESIS 2:
- One-sentence thesis: Fine-grained GPU virtual memory management is structurally unfit for highly-branched agentic KV-cache copy-on-write due to a hard, OS-independent driver-level mapping table ceiling that causes metadata exhaustion long before physical memory fills.
- Repo it builds on: forkedkv
- Contribution type: characterization
- Supporting evidence (cite specific repo metric/result): Capacity limit (M4/M4b) shows OOMs at 6 branches due to cuMemSetAccess failures constrained by an independent ~520K mapping ceiling, while live HBM usage remains flat (Lab1).
- Closest prior work + the delta: vAttention uses VMM successfully for linear read-only serving; we characterize its catastrophic architectural failure for dense, branching copy-on-write patterns.
- Closest OSS competitor + why it doesn't kill this: vLLM / RadixAttention. Doesn't kill it because this thesis explicitly proves *why* software Radix (measured in R4 as 700x faster) fundamentally defeats hardware CoW.
- Fatal flaw (the strongest reason a reviewer kills it): NVIDIA could release a driver update (post-580.82) that expands or dynamically resizes the context mapping table, invalidating the 520K constant.
- Anti-FlashInfer test (would it matter if FlashInfer were infinitely fast? YES/NO + why): YES. Kernel speeds are irrelevant when the bottleneck is the CUDA driver `cuMemSetAccess` metadata limitation.
- No-code test (if the impl vanished, what knowledge remains?): The existence, empirical bounds, and root cause (mapping metadata vs physical memory) of the 520K GPU driver mapping ceiling on H100s.
- Highest-EV experiment (1 sentence: what to run, min success criterion): Test the mapping ceiling across multiple GPU architectures (A100, L40S, MI300X) to determine if the 520K limit is a hardware TLB constraint or an arbitrary driver configuration.
- Realistic venue: ASPLOS
- Your self-vote: GREEN (Reviewers love rigorous forensic teardowns of assumed hardware capabilities; the R4 retraction makes this bulletproof).

THESIS 3:
- One-sentence thesis: Current LLM serving engines optimize for linear sequence growth, but real agentic workloads exhibit destructive non-linear prompt mutations and extreme branching that degrade TTFT and HBM efficiency without physical-layer memory stitching.
- Repo it builds on: cross-repo
- Contribution type: workload-model
- Supporting evidence (cite specific repo metric/result): `forkedkv` M5 (90% fewer KV bytes / 80% lower HBM for SWE-bench traces) and `edmm` (8.21x TTFT penalty on vLLM from mid-prompt injection).
- Closest prior work + the delta: ServerlessLLM models prefix sharing across independent requests; we model intra-request destructive mutations (tool injection, branch rollbacks) and their specific physical memory penalties.
- Closest OSS competitor + why it doesn't kill this: SGLang (Radix tree). Handles branching efficiently but fails structurally on mid-prompt mutations (causing the 8.21x penalty) unless entirely recomputed.
- Fatal flaw (the strongest reason a reviewer kills it): It characterizes a problem across two axes (branching and injection) without a single unified solution, as the hardware CoW solution for branching was retracted.
- Anti-FlashInfer test (would it matter if FlashInfer were infinitely fast? YES/NO + why): YES. Infinite kernel speed does not solve physical memory capacity exhaustion (80% higher HBM in clones).
- No-code test (if the impl vanished, what knowledge remains?): The empirical divergence of agentic workloads from linear serving assumptions, specifically the frequency and hardware cost of mid-prompt mutations.
- Highest-EV experiment (1 sentence: what to run, min success criterion): Instrument 1,000 SWE-bench trajectories to quantify exactly how many tokens are recomputed due to prefix-cache invalidations vs simple appends.
- Realistic venue: MLSys
- Your self-vote: YELLOW (Strong motivation, but lacks a single unified architectural mechanism since we retracted the hardware CoW solution).

THESIS 4:
- One-sentence thesis: While VMM-backed contiguous virtual addressing allows zero-overhead unmodified fast-path attention (SDPA) for branched KV caches, the setup latency of driver map operations completely destroys the end-to-end performance gains.
- Repo it builds on: forkedkv
- Contribution type: optimization
- Supporting evidence (cite specific repo metric/result): Attn overhead (M3) is zero (-0.1% to +1.1%) vs unmodified SDPA, but map-op floor (~50us/page) makes software forks ~700x faster (R4).
- Closest prior work + the delta: PagedAttention requires custom kernels to handle non-contiguous memory; this restores contiguous VA for SDPA but exposes the driver tax.
- Closest OSS competitor + why it doesn't kill this: FlashInfer. It completely kills this because FlashInfer's paged attention kernels are so fast they render the "unmodified SDPA" requirement obsolete.
- Fatal flaw (the strongest reason a reviewer kills it): State-of-the-art paged kernels already minimize the PagedAttention tax, leaving only the VMM setup latency which we measured to be disastrous.
- Anti-FlashInfer test (would it matter if FlashInfer were infinitely fast? YES/NO + why): NO. If FlashInfer is infinitely fast, the need for contiguous VA to run vanilla SDPA completely evaporates.
- No-code test (if the impl vanished, what knowledge remains?): The measurement that driver-level VMM mapping latency eclipses the kernel-level overhead of handling non-contiguous pages.
- Highest-EV experiment (1 sentence: what to run, min success criterion): End-to-end latency bake-off: SDPA on VMM-contiguous pages vs FlashInfer on software-paged memory for highly branched SWE-bench traces.
- Realistic venue: KILL
- Your self-vote: RED (The thesis is trivially destroyed by the Anti-FlashInfer test and our own R4 measurements. It is a dead end).
