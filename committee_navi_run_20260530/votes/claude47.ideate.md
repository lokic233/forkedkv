THESIS 1:
- One-sentence thesis: The per-context GPU VMM mapping table (~520K entries on H100, K·P=const) is a first-class, exhaustible accelerator resource that must be scheduled, accounted, and exposed by the runtime — not the HBM byte budget.
- Repo it builds on: forkedkv
- Contribution type: characterization
- Supporting evidence (cite specific repo metric/result): M4b K·P≈520K constant across configs; Lab1 confirms ceiling is independent of vm.max_map_count (392 VMAs vs 67M limit); M4 forensic OOM site = cuMemSetAccess, not data allocation; live HBM flat at 12 GiB while branches scale 14×.
- Closest prior work + the delta: vAttention (ASPLOS'25) uses VMM for serving but never enumerates the mapping-table ceiling or models it; our delta is the K·P invariant, the OOM call-site forensics, and the proof it is orthogonal to HBM bytes and to Linux VMA limits.
- Closest OSS competitor + why it doesn't kill this: vLLM/SGLang sidestep VMM entirely (PagedAttention software refcount) so they never hit the ceiling, but every system that does use VMM (vAttention, future driver-level designs) inherits it; this is a portable architectural fact, not a vLLM design choice.
- Fatal flaw (the strongest reason a reviewer kills it): "NVIDIA will raise the limit in CUDA 13 / Blackwell." Rebuttal must show K·P is an MMU/page-table cardinality law, not a tunable knob.
- Anti-FlashInfer test (would it matter if FlashInfer were infinitely fast? YES/NO + why): YES. Kernel throughput is irrelevant to a driver-level mapping-metadata exhaustion bound.
- No-code test (if the impl vanished, what knowledge remains?): The K·P=const law, the cuMemSetAccess call-site evidence, the vm.max_map_count independence — usable by any future GPU memory-manager design.
- Highest-EV experiment (1 sentence: what to run, min success criterion): Reproduce K·P on A100, L40S, MI300X and a B200 if available; success = same invariant within ±15% on ≥2 vendor stacks → claim becomes a cross-architecture law, not an H100 quirk.
- Realistic venue: ASPLOS
- Your self-vote: GREEN — characterization with a measured invariant, a forensic call-site, and a cross-vendor reproducibility plan is exactly the shape ASPLOS will publish.

THESIS 2:
- One-sentence thesis: Contiguous-VA physical-page sharing is a *kernel-compatibility substrate* — the unique value of driver-level KV CoW is not fork speed but the ability to run unmodified attention kernels (SDPA, FlashAttention-3, FlashInfer) on shared prefixes without a PagedAttention rewrite tax.
- Repo it builds on: forkedkv
- Contribution type: abstraction
- Supporting evidence (cite specific repo metric/result): M3 attention overhead –0.1% to +1.1% with unmodified SDPA/FlashAttention on a CoW-forked branch; R4 retraction concedes software is 700× faster at fork but cannot offer contiguous-VA sharing without PagedAttention; M5b confirms full 28-layer Qwen2.5-7B decode is bit-identical on CoW pages.
- Closest prior work + the delta: PagedAttention/RadixAttention require every new kernel to be re-implemented against a paged layout; our delta is quantifying the ecosystem kernel-rewrite cost they impose and showing a primitive that side-steps it.
- Closest OSS competitor + why it doesn't kill this: vLLM/SGLang win on raw fork latency and capacity (R4) but force every kernel author to maintain a paged variant; this thesis reframes the competition on integration cost across the kernel ecosystem, not on fork µs.
- Fatal flaw (the strongest reason a reviewer kills it): "FlashInfer already ships paged kernels for everything that matters, so the compatibility tax is zero in practice." Must enumerate kernels without paged variants (custom research kernels, new attention variants, third-party fused ops) with measured porting cost.
- Anti-FlashInfer test (would it matter if FlashInfer were infinitely fast? YES/NO + why): YES — the thesis is precisely that not every kernel is FlashInfer, and the substrate's value scales with the count of kernels that aren't.
- No-code test (if the impl vanished, what knowledge remains?): A design principle — "branch the virtual address space, not the data layout" — and a quantified taxonomy of which kernels pay the paged-rewrite tax.
- Highest-EV experiment (1 sentence: what to run, min success criterion): Survey + port-effort measurement on 8–10 recent attention/MoE kernels (e.g., MLA, FlashMLA, GQA-fused, custom RL reward kernels); success = ≥3 kernels with measurable porting cost (LoC + perf delta) that CoW substrate avoids.
- Realistic venue: MLSys
- Your self-vote: YELLOW — strong framing but lives or dies on the kernel-survey numbers; if FlashInfer's paged coverage is near-complete, the thesis collapses to a niche.

THESIS 3:
- One-sentence thesis: Tool-call idle windows are a *schedulable resource* — speculative VMM remap of likely post-tool KV states during the 100ms–1s tool wait converts measured cache-invalidation penalty into hidden latency.
- Repo it builds on: edmm
- Contribution type: mechanism
- Supporting evidence (cite specific repo metric/result): Live vLLM penalty B/A = 8.21× TTFT from tool-call mid-prompt invalidation; EDMM cuMemMap pointer-swap = 58µs vs 202µs memcpy+recompute (µbench); EDMM recovery C/A = 1.17× shows the remap path works under live engine load.
- Closest prior work + the delta: EDMM itself remaps *after* the tool returns; the delta is a small predictor over historical tool outputs that drives *speculative* remaps during the wait, paying 58µs per candidate to hide the 274ms recompute on hits.
- Closest OSS competitor + why it doesn't kill this: SGLang's RadixAttention reuses common prefixes but cannot speculate post-tool branches because it has no per-page remap primitive; vLLM has no tool-aware scheduler at all.
- Fatal flaw (the strongest reason a reviewer kills it): "Tool outputs are too unpredictable for speculation to amortize." Must show even a 20–30% hit rate on a realistic tool-call distribution (search, code-exec, file-read) clears the break-even.
- Anti-FlashInfer test (would it matter if FlashInfer were infinitely fast? YES/NO + why): YES — kernel speed does not shorten the tool I/O window; the win is scheduling work into time the GPU would otherwise idle.
- No-code test (if the impl vanished, what knowledge remains?): The principle that agentic serving has an exploitable idle-time distribution distinct from chat workloads, plus a break-even model relating remap cost, recompute cost, and predictor accuracy.
- Highest-EV experiment (1 sentence: what to run, min success criterion): On a SWE-bench-Verified trace replay with real tool latencies, measure end-to-end TTFT with and without speculative remap; success = ≥1.5× TTFT reduction vs EDMM baseline at ≥25% predictor hit rate.
- Realistic venue: EuroSys/ATC
- Your self-vote: GREEN — builds on a measured 8.21× pain point, has a clear break-even model, and a venue that values scheduling mechanisms over peak throughput.

THESIS 4:
- One-sentence thesis: Agentic LLM serving has a workload-defining property — *cache-chain invalidation from mid-prompt tool insertion* — that no current benchmark captures, and a reusable trace+harness exposing it is a prerequisite for the next decade of KV-cache research.
- Repo it builds on: cross-repo (edmm + forkedkv SWE-bench traces)
- Contribution type: workload-model
- Supporting evidence (cite specific repo metric/result): edmm 8.21× live vLLM TTFT penalty is undocumented in any public benchmark (MLPerf-Inference, LMSYS-Chat-1M, ShareGPT); forkedkv M5 captures 24 real SWE-bench-Verified branching traces showing divergence patterns absent from chat datasets.
- Closest prior work + the delta: MLPerf-Inference and LMSYS workloads are chat-shaped (single linear context); delta is a public trace+replay harness that reproduces the 8.21× invalidation event on any serving stack and scores systems on agentic invalidation recovery, not raw throughput.
- Closest OSS competitor + why it doesn't kill this: AgentBench, SWE-bench, τ-bench measure *task success*, not serving-system behavior; no public artifact lets you run vLLM/SGLang/TRT-LLM head-to-head on agentic KV invalidation.
- Fatal flaw (the strongest reason a reviewer kills it): "It's just a benchmark — papers/datasets track only, no mechanism." Must position as a workload-model paper that *also* ships the benchmark, like how SPEC and TPC-DS papers were accepted.
- Anti-FlashInfer test (would it matter if FlashInfer were infinitely fast? YES/NO + why): YES — kernel speed is orthogonal to whether the cache hash chain is valid; FlashInfer at ∞ throughput still eats the 274ms recompute.
- No-code test (if the impl vanished, what knowledge remains?): A precise definition of the agentic-invalidation workload class, a tool-call insertion-rate distribution from real SWE-bench traces, and the empirical penalty curve as a function of insertion depth.
- Highest-EV experiment (1 sentence: what to run, min success criterion): Release traces + replay harness, run on vLLM, SGLang, TRT-LLM, and report invalidation-recovery TTFT distributions; success = ≥3 systems show >5× penalty under the workload, validating it as a real and unaddressed gap.
- Realistic venue: MLSys (Datasets & Benchmarks track) or Workshop
- Your self-vote: YELLOW — the 8.21× datum is real and the gap is real, but pure benchmark papers struggle at top venues unless paired with a mechanism contribution; risk of being absorbed as Section 2 of Thesis 3.
