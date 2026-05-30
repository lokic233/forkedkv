# Briefing: Forkable GPU Memory for Replayable Agent Execution
## For external hostile review — requesting brutal guidance on lab experiment design

---

## Target Conference: ASPLOS 2027 (primary), OSDI 2026 (backup)

ASPLOS bar: hardware-software co-design, memory/cache architecture, GPU systems, scheduling at the architecture/runtime boundary. Rejects pure-systems (→ OSDI) and pure-ML-systems (→ MLSys).

---

## The Idea (1 paragraph)

When an autonomous coding agent (e.g. SWE-agent, OpenHands, Devin) explores multiple solution strategies in parallel, it forks its execution into branches that share a long common prefix of KV-cache state. Today, each branch copies the entire KV prefix — at 12 GiB for a coding agent context, this limits a single H100 to ~6 concurrent branches before OOM. We propose **branch-aware copy-on-write of KV-cache pages at the GPU MMU level** using the CUDA Virtual Memory Management (VMM) API. Forking aliases the parent's physical HBM pages (zero-copy, refcounted); only pages that a branch *actually writes to* trigger a per-page CoW remap. The architectural insight: this is OS-style demand-paging applied to GPU attention state, enabling 14× more concurrent branches on the same hardware — and the ceiling is mapping-metadata, not data memory.

---

## What the MVP Prototype Has Proved (real H100 measurements, Qwen2.5-7B)

### Mechanism (all real CUDA VMM driver calls, not os.fork or tensor copy):
- `cuMemCreate` / `cuMemMap` / `cuMemUnmap` / `cuMemRetainAllocationHandle`
- Physical page aliasing verified via driver handle identity
- Software-detected CoW: refcount check → cuMemUnmap + cuMemMap to new physical handle + D2D copy
- Dynamic VA growth via lazy page mapping (append_page)
- VA free-list for serial branch recycling

### Measured results (v0.3, commit 9f2ecb7):

| Metric | Result | Comparison |
|--------|--------|------------|
| **Capacity** | Full-clone OOMs at **6 branches** (94.5 GiB); CoW reaches **84 branches** (live HBM flat at 12.0 GiB). OOM forensically pinned to `cuMemSetAccess` (VA-mapping metadata, NOT data). | **14× capacity gain** |
| **Bytes written (5% tail divergence)** | CoW writes **95% fewer** KV bytes than full-clone | Exact integers, tail model |
| **Attention overhead** | VMM-paged KV adds **+0.05%** vs contiguous (was +7.7% before methodology fix) | Essentially zero |
| **Multi-layer real decode** | 4 full Qwen2.5-7B transformer blocks (attn+MLP+residuals), 16 branches × 128 tokens, **bit-identical** CoW vs clone (hard assert, all branches) | Non-degenerate tokens, unaligned prefix |
| **Peak HBM (decode)** | CoW **288 MiB** vs clone **544 MiB** (−47%) | Real attention computation |
| **Throughput** | CoW **221 tok/s** vs clone **187 tok/s** (CoW slightly faster — clone does upfront copy work) | Parity to slight win |
| **CoW cost breakdown** | Full CoW: 178 µs. D2D copy: 13 µs (7%). Scratch-VA overhead: 83 µs (47% — **hypothesis "removable" was FALSIFIED**, retracted with evidence). VA-swap alternative: 72 µs (59% faster, breaks contiguity). | Honest null result |
| **VA recycling** | 120 fork/destroy cycles, reserved=10, reused=119, HBM flat | Serial throughput unbounded |
| **SWE-bench coverage** | 24 real instances (143–24,770 chars, 7 repos, median matches full-set) | 90% bytes / 80% HBM reduction stable |

### Key scientific findings:
1. **The 84-branch ceiling is NOT data memory** — it's cuMemSetAccess mapping-metadata exhaustion (515K live mappings). Only 12 of 97 GiB HBM used at OOM.
2. **Scratch-VA pooling does NOT help** (only 3% gain, falsifying the R1 "47% removable" hypothesis). The cost is cuMemSetAccess + cuMemUnmap, irreducible without NVIDIA driver changes.
3. **Fork latency is NOT flat** — scales linearly with prefix pages (per-page cuMemMap cost). CoW is ~1.3× faster than full-clone but does not eliminate the per-page overhead.
4. **Zero attention kernel overhead** from VMM page indirection (the TLB handles it transparently at seqlen ≤8192).

### Honest null results / limitations:
- Fork latency target "flat" was MISSED — linear, not flat
- Wall-time speedup is parity (mechanism wins on memory, not latency)
- Software-detected CoW (no hardware page fault on GPU)
- Still 4 of 28 layers in headline decode (full-model depth being validated in R3)
- No live vLLM integration (analytic comparison + design sketch provided)
- 2 MiB CoW granularity is coarse (1-token write copies full 2 MiB page)

---

## Committee Review History

| Round | Verdicts | ASPLOS acceptance estimate |
|-------|----------|---------------------------|
| R0 (v0.1) | 4× YELLOW | 15-30% |
| R1 (v0.2) | 1× GREEN + 3× borderline-YELLOW | 35-45% |
| R2 (v0.3) | **3× GREEN + 1× borderline-YELLOW** | **55-70%** |
| R3 (in progress) | target: 4× GREEN | — |

Internal committee: codex (gpt-5.5), claude (Opus 4.8), gemini, metacode (Avocado/Muse Spark).
Each reviewer has full code+data access and verified mechanism integrity via line-by-line code inspection.

---

## What R3 Is Addressing (metacode's 3 remaining asks):
1. Full 28-layer depth validation (prove mechanism scales beyond 4 of 28)
2. Concurrency ceiling model (quantify max_branches ≈ 515K / prefix_pages)
3. Partial-page CoW waste quantification (honest about 2 MiB granularity overhead)

---

## Prior Art We Differentiate From (verified):
- **vAttention (ASPLOS'25)**: CUDA VMM for serving KV — but read-only sharing, no fork/CoW
- **ChunkAttention (ASPLOS'24)**: chunked attention for shared prefixes — no write semantics after fork
- **vLLM PagedAttention (SOSP'23)**: KV virtual memory — per-request lifecycle, no branching
- **SGLang RadixAttention (NeurIPS'24)**: radix-tree prefix sharing — software refcounts, no GPU MMU aliasing
- **POD-Attention (ASPLOS'25)**: partition-of-decode — orthogonal scheduling problem
- **CXLfork (ASPLOS'25)**: CXL-mediated fork — CPU-only, ours is GPU
- **ServerlessLLM (OSDI'24)**: model checkpointing — model weights, not request-level KV state
- **rr (ATC'17) / CRIU**: process replay/checkpoint — no GPU KV awareness

---

## What We Need Guidance On (for the external reviewer):

1. **Is the 4-layer → 28-layer scaling convincing, or does a hostile ASPLOS reviewer need full-model end-to-end throughput numbers (which require vLLM-level serving integration)?**

2. **Should the paper lead with "Forkable GPU Memory" (architecture angle) or "Branchable Replay for Agents" (systems angle)?** Our internal committee unanimously says architecture-first, agent-as-application. Is that right for ASPLOS specifically?

3. **The 84-branch ceiling is a driver metadata limit. Is this a strength (honest, novel characterization of CUDA VMM behavior) or a weakness (shows the system doesn't actually scale)?**

4. **CoW granularity is 2 MiB (CUDA VMM minimum). For agents that diverge by 1 token (~1 KB of KV), this wastes 99.9% of the copied page. Is sub-page CoW (with a custom CUDA kernel) needed for ASPLOS, or is characterizing the waste sufficient?**

5. **What specific experiments would make this a clear ASPLOS oral?** We have H100 hardware access and can run anything.

---

## Researcher Profile:
- Strong LLM serving / infra / agent systems background (vLLM, LMCache, KV-cache)
- Production agent systems experience (queues, routing, retries, multi-agent flows)
- Access to H100/H200, TPU v6, AMD MI300/MI350, MTIA
- At Meta (FAIR lab quality target)

---

## Artifact:
- ~/branchable_replay/ on H100 devgpu014 (13 git commits, ~2500 LOC Python+CUDA-Python)
- Every number reproducible from committed scripts with default args
- Every claim cites a specific data/CSV file
- Null results surfaced proactively (not hidden)
- B8 "47% removable" hypothesis was built, measured, FALSIFIED, and retracted with evidence

