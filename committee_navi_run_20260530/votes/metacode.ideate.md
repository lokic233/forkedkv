67 skills discovered
THESIS 1:
- One-sentence thesis: GPU VMM per-context mapping tables exhaust at ~520K entries on H100, independent of OS limits, imposing a predictable K/P capacity ceiling for VMM-based KV sharing.
- Repo it builds on: forkedkv
- Contribution type: characterization
- Supporting evidence (cite specific repo metric/result): M4b shows branches × prefix_pages = CONSTANT ~520K; Lab1 shows ceiling unchanged across vm.max_map_count 392 vs 67M; OOM site is cuMemSetAccess per M4.
- Closest prior work + the delta: vAttention (ASPLOS'25) uses CUDA VMM for KV but treats it as unbounded; delta is forensic identification of mapping-table limit, K/P model, and cuMemSetAccess as failure point.
- Closest OSS competitor + why it doesn't kill this: RadixAttention/SGLang uses software prefix tree, avoids VMM entirely, so never hits mapping limit; but it cannot run unmodified SDPA (pays PagedAttention tax), which is orthogonal to capacity characterization.
- Fatal flaw (the strongest reason a reviewer kills it): Limit is driver version (580.82) and GPU (H100) specific; NVIDIA may raise or remove limit in future, making characterization ephemeral.
- Anti-FlashInfer test (would it matter if FlashInfer were infinitely fast? YES/NO + why): YES, because mapping exhaustion is a metadata resource limit, not a compute bottleneck; infinitely fast kernels do not reduce VA mapping count.
- No-code test (if the impl vanished, what knowledge remains?): GPU VMM has ~520K per-context mapping entry ceiling; capacity for VMM-based sharing follows branches × pages ≈ 520K; OOM manifests at cuMemSetAccess, not data allocation.
- Highest-EV experiment (1 sentence: what to run, min success criterion): Sweep prefix_pages 16–4096 and branches 1–200 on H100/A100, measure OOM point and cuMemSetAccess latency; success if branches×pages ≈ 520K±5% across GPUs and latency correlates with mapping count.
- Realistic venue: ASPLOS
- Your self-vote: GREEN (+ 1 line why): Forensic measurement is solid, reproducible, and guides system design regardless of VMM deployment viability.

THESIS 2:
- One-sentence thesis: VMM-based copy-on-write can expose branched KV cache as a contiguous virtual address, allowing unmodified FlashAttention/SDPA to run on forked branches without PagedAttention kernel changes.
- Repo it builds on: forkedkv
- Contribution type: runtime-primitive
- Supporting evidence (cite specific repo metric/result): M3 shows -0.1% to +1.1% attention overhead running SDPA unmodified on forked branch; M5b shows 8-branch Qwen2.5-7B full 28-layer decode bit-identical to clone with -44% peak HBM.
- Closest prior work + the delta: vAttention (ASPLOS'25) provides contiguous VA for read-only KV; PagedAttention (SOSP'23) requires kernel rewrite for sharing; delta is CoW with contiguous VA preserves stock kernel ABI across writes.
- Closest OSS competitor + why it doesn't kill this: vLLM-APC and FlashInfer use PagedAttention, incurring 5–22% kernel tax (Lab3b corrected); they cannot execute vanilla SDPA on shared prefix without indirection, whereas contiguous VA primitive does.
- Fatal flaw (the strongest reason a reviewer kills it): Fork latency is 700× slower than software RadixAttention and capacity 6× smaller at 32-block prefix (R4 retraction); primitive is architecturally clean but practically dominated by software baseline.
- Anti-FlashInfer test (would it matter if FlashInfer were infinitely fast? YES/NO + why): NO, because if FlashInfer were infinitely fast, the PagedAttention tax vanishes; the sole advantage of avoiding kernel changes becomes moot.
- No-code test (if the impl vanished, what knowledge remains?): VMM remapping can present CoW-shared KV as a single contiguous VA range, enabling stock attention kernels to operate on branched cache without page-table indirection.
- Highest-EV experiment (1 sentence: what to run, min success criterion): Run FlashAttention-2 on 8-way forked KV (2K prefix) vs PagedAttention baseline, measure kernel time and output equality; success if overhead <2% and bit-identical without kernel mods.
- Realistic venue: MLSys
- Your self-vote: YELLOW (+ 1 line why): Mechanism is elegant and validated, but R4 retraction shows it loses to software on both speed and capacity, limiting practical impact.

THESIS 3:
- One-sentence thesis: Agentic tool-call mid-prompt injection breaks prefix-cache hash chains, causing 8.21× TTFT penalty, which VMM pointer-swap repairs to 1.17× in 58 µs.
- Repo it builds on: edmm
- Contribution type: workload-model
- Supporting evidence (cite specific repo metric/result): edmm live vLLM 0.6.6: Radix penalty B/A=8.21× TTFT on Qwen2.5-7B/H100; EDMM recovery C/A=1.17× via cuMemMap 58 µs vs recompute 274 ms; standalone 4.18×→0.96×.
- Closest prior work + the delta: RadixAttention/SGLang models prefix sharing for static prompts; no prior work characterizes tool-call as a cache-invalidation event or measures its live-engine TTFT impact; delta is workload pathology + pointer-swap repair.
- Closest OSS competitor + why it doesn't kill this: LangGraph/CrewAI checkpoint agent state to CPU/disk but do not repair GPU KV cache; they still incur 274 ms recompute on resume, whereas EDMM avoids compute via 58 µs remap.
- Fatal flaw (the strongest reason a reviewer kills it): Speculative prefill during tool idle assumes tool latency dominates and control flow is predictable; misprediction wastes compute, and measurement is tied to vLLM 0.6.6 internals.
- Anti-FlashInfer test (would it matter if FlashInfer were infinitely fast? YES/NO + why): YES, because penalty is memory-bound prefill recompute (274 ms), not kernel speed; infinitely fast FlashInfer still requires reading weights and writing KV, so pointer-swap remains 4,700× faster.
- No-code test (if the impl vanished, what knowledge remains?): Tool-call injection invalidates prefix hash chains, causing 8× TTFT in production engines; VMM pointer remapping can stitch KV cache in microseconds, avoiding recompute.
- Highest-EV experiment (1 sentence: what to run, min success criterion): Instrument vLLM 0.6.6 to inject tool-call at 5 prompt positions, measure TTFT for Radix vs EDMM; success if EDMM ≤1.2× baseline and ≥5× speedup over Radix penalty.
- Realistic venue: NSDI
- Your self-vote: GREEN (+ 1 line why): Live-engine 8.21× measurement is stark and reproducible; 58 µs repair is 4,700× faster than recompute, making the pathology and fix undeniable.

THESIS 4:
- One-sentence thesis: A hybrid VMM-software policy using the K/P mapping ceiling (~520K) to fall back from CoW to RadixAttention avoids VMM OOM while preserving contiguous-VA fast path for small forks.
- Repo it builds on: cross-repo
- Contribution type: optimization
- Supporting evidence (cite specific repo metric/result): forkedkv M4b gives K/P constant ~520K; M4 shows CoW OOM at 84 branches (12 GiB prefix) vs Radix 6× larger capacity (R4); edmm shows 58 µs VMM swap is viable for low branch counts.
- Closest prior work + the delta: vAttention uses VMM only; Radix uses software only; no prior work uses forensic mapping limit to guide runtime fallback; delta is K/P-driven hybrid policy.
- Closest OSS competitor + why it doesn't kill this: vLLM-APC (PagedAttention) never OOMs on mappings but pays 5–22% kernel tax (Lab3b); hybrid uses VMM only when mapping budget allows, otherwise falls back, avoiding both OOM and unnecessary tax.
- Fatal flaw (the strongest reason a reviewer kills it): R4 retraction shows software Radix is 700× faster fork and 6× higher capacity; the VMM fast path is so narrow that policy complexity outweighs benefit, and dynamic agentic workloads make K/P prediction unreliable.
- Anti-FlashInfer test (would it matter if FlashInfer were infinitely fast? YES/NO + why): YES, because OOM is caused by mapping-table exhaustion, not attention speed; infinitely fast FlashInfer does not increase the ~520K mapping budget.
- No-code test (if the impl vanished, what knowledge remains?): GPU VMM mapping limit follows K/P model; a runtime can estimate remaining mappings and switch to software prefix tree before OOM.
- Highest-EV experiment (1 sentence: what to run, min success criterion): Simulate branching workload with varying prefix size, implement policy that switches to Radix when predicted mappings >500K; success if zero OOM and p99 latency within 10% of oracle optimal.
- Realistic venue: EuroSys/ATC
- Your self-vote: YELLOW (+ 1 line why): Policy is sensible and builds on solid forensic data, but R4 practically kills VMM for high fanout, leaving only niche low-branch scenarios.
THESIS 1:
- One-sentence thesis: GPU VMM per-context mapping tables exhaust at ~520K entries on H100, independent of OS limits, imposing a predictable K/P capacity ceiling for VMM-based KV sharing.
- Repo it builds on: forkedkv
- Contribution type: characterization
- Supporting evidence (cite specific repo metric/result): M4b shows branches × prefix_pages = CONSTANT ~520K; Lab1 shows ceiling unchanged across vm.max_map_count 392 vs 67M; OOM site is cuMemSetAccess per M4.
- Closest prior work + the delta: vAttention (ASPLOS'25) uses CUDA VMM for KV but treats it as unbounded; delta is forensic identification of mapping-table limit, K/P model, and cuMemSetAccess as failure point.
- Closest OSS competitor + why it doesn't kill this: RadixAttention/SGLang uses software prefix tree, avoids VMM entirely, so never hits mapping limit; but it cannot run unmodified SDPA (pays PagedAttention tax), which is orthogonal to capacity characterization.
- Fatal flaw (the strongest reason a reviewer kills it): Limit is driver version (580.82) and GPU (H100) specific; NVIDIA may raise or remove limit in future, making characterization ephemeral.
- Anti-FlashInfer test (would it matter if FlashInfer were infinitely fast? YES/NO + why): YES, because mapping exhaustion is a metadata resource limit, not a compute bottleneck; infinitely fast kernels do not reduce VA mapping count.
- No-code test (if the impl vanished, what knowledge remains?): GPU VMM has ~520K per-context mapping entry ceiling; capacity for VMM-based sharing follows branches × pages ≈ 520K; OOM manifests at cuMemSetAccess, not data allocation.
- Highest-EV experiment (1 sentence: what to run, min success criterion): Sweep prefix_pages 16–4096 and branches 1–200 on H100/A100, measure OOM point and cuMemSetAccess latency; success if branches×pages ≈ 520K±5% across GPUs and latency correlates with mapping count.
- Realistic venue: ASPLOS
- Your self-vote: GREEN (+ 1 line why): Forensic measurement is solid, reproducible, and guides system design regardless of VMM deployment viability.

THESIS 2:
- One-sentence thesis: VMM-based copy-on-write can expose branched KV cache as a contiguous virtual address, allowing unmodified FlashAttention/SDPA to run on forked branches without PagedAttention kernel changes.
- Repo it builds on: forkedkv
- Contribution type: runtime-primitive
- Supporting evidence (cite specific repo metric/result): M3 shows -0.1% to +1.1% attention overhead running SDPA unmodified on forked branch; M5b shows 8-branch Qwen2.5-7B full 28-layer decode bit-identical to clone with -44% peak HBM.
- Closest prior work + the delta: vAttention (ASPLOS'25) provides contiguous VA for read-only KV; PagedAttention (SOSP'23) requires kernel rewrite for sharing; delta is CoW with contiguous VA preserves stock kernel ABI across writes.
- Closest OSS competitor + why it doesn't kill this: vLLM-APC and FlashInfer use PagedAttention, incurring 5–22% kernel tax (Lab3b corrected); they cannot execute vanilla SDPA on shared prefix without indirection, whereas contiguous VA primitive does.
- Fatal flaw (the strongest reason a reviewer kills it): Fork latency is 700× slower than software RadixAttention and capacity 6× smaller at 32-block prefix (R4 retraction); primitive is architecturally clean but practically dominated by software baseline.
- Anti-FlashInfer test (would it matter if FlashInfer were infinitely fast? YES/NO + why): NO, because if FlashInfer were infinitely fast, the PagedAttention tax vanishes; the sole advantage of avoiding kernel changes becomes moot.
- No-code test (if the impl vanished, what knowledge remains?): VMM remapping can present CoW-shared KV as a single contiguous VA range, enabling stock attention kernels to operate on branched cache without page-table indirection.
- Highest-EV experiment (1 sentence: what to run, min success criterion): Run FlashAttention-2 on 8-way forked KV (2K prefix) vs PagedAttention baseline, measure kernel time and output equality; success if overhead <2% and bit-identical without kernel mods.
- Realistic venue: MLSys
- Your self-vote: YELLOW (+ 1 line why): Mechanism is elegant and validated, but R4 retraction shows it loses to software on both speed and capacity, limiting practical impact.

THESIS 3:
- One-sentence thesis: Agentic tool-call mid-prompt injection breaks prefix-cache hash chains, causing 8.21× TTFT penalty, which VMM pointer-swap repairs to 1.17× in 58 µs.
- Repo it builds on: edmm
- Contribution type: workload-model
- Supporting evidence (cite specific repo metric/result): edmm live vLLM 0.6.6: Radix penalty B/A=8.21× TTFT on Qwen2.5-7B/H100; EDMM recovery C/A=1.17× via cuMemMap 58 µs vs recompute 274 ms; standalone 4.18×→0.96×.
- Closest prior work + the delta: RadixAttention/SGLang models prefix sharing for static prompts; no prior work characterizes tool-call as a cache-invalidation event or measures its live-engine TTFT impact; delta is workload pathology + pointer-swap repair.
- Closest OSS competitor + why it doesn't kill this: LangGraph/CrewAI checkpoint agent state to CPU/disk but do not repair GPU KV cache; they still incur 274 ms recompute on resume, whereas EDMM avoids compute via 58 µs remap.
- Fatal flaw (the strongest reason a reviewer kills it): Speculative prefill during tool idle assumes tool latency dominates and control flow is predictable; misprediction wastes compute, and measurement is tied to vLLM 0.6.6 internals.
- Anti-FlashInfer test (would it matter if FlashInfer were infinitely fast? YES/NO + why): YES, because penalty is memory-bound prefill recompute (274 ms), not kernel speed; infinitely fast FlashInfer still requires reading weights and writing KV, so pointer-swap remains 4,700× faster.
- No-code test (if the impl vanished, what knowledge remains?): Tool-call injection invalidates prefix hash chains, causing 8× TTFT in production engines; VMM pointer remapping can stitch KV cache in microseconds, avoiding recompute.
- Highest-EV experiment (1 sentence: what to run, min success criterion): Instrument vLLM 0.6.6 to inject tool-call at 5 prompt positions, measure TTFT for Radix vs EDMM; success if EDMM ≤1.2× baseline and ≥5× speedup over Radix penalty.
- Realistic venue: NSDI
- Your self-vote: GREEN (+ 1 line why): Live-engine 8.21× measurement is stark and reproducible; 58 µs repair is 4,700× faster than recompute, making the pathology and fix undeniable.

THESIS 4:
- One-sentence thesis: A hybrid VMM-software policy using the K/P mapping ceiling (~520K) to fall back from CoW to RadixAttention avoids VMM OOM while preserving contiguous-VA fast path for small forks.
- Repo it builds on: cross-repo
- Contribution type: optimization
- Supporting evidence (cite specific repo metric/result): forkedkv M4b gives K/P constant ~520K; M4 shows CoW OOM at 84 branches (12 GiB prefix) vs Radix 6× larger capacity (R4); edmm shows 58 µs VMM swap is viable for low branch counts.
- Closest prior work + the delta: vAttention uses VMM only; Radix uses software only; no prior work uses forensic mapping limit to guide runtime fallback; delta is K/P-driven hybrid policy.
- Closest OSS competitor + why it doesn't kill this: vLLM-APC (PagedAttention) never OOMs on mappings but pays 5–22% kernel tax (Lab3b); hybrid uses VMM only when mapping budget allows, otherwise falls back, avoiding both OOM and unnecessary tax.
- Fatal flaw (the strongest reason a reviewer kills it): R4 retraction shows software Radix is 700× faster fork and 6× higher capacity; the VMM fast path is so narrow that policy complexity outweighs benefit, and dynamic agentic workloads make K/P prediction unreliable.
- Anti-FlashInfer test (would it matter if FlashInfer were infinitely fast? YES/NO + why): YES, because OOM is caused by mapping-table exhaustion, not attention speed; infinitely fast FlashInfer does not increase the ~520K mapping budget.
- No-code test (if the impl vanished, what knowledge remains?): GPU VMM mapping limit follows K/P model; a runtime can estimate remaining mappings and switch to software prefix tree before OOM.
- Highest-EV experiment (1 sentence: what to run, min success criterion): Simulate branching workload with varying prefix size, implement policy that switches to Radix when predicted mappings >500K; success if zero OOM and p99 latency within 10% of oracle optimal.
- Realistic venue: EuroSys/ATC
- Your self-vote: YELLOW (+ 1 line why): Policy is sensible and builds on solid forensic data, but R4 practically kills VMM for high fanout, leaving only niche low-branch scenarios.
