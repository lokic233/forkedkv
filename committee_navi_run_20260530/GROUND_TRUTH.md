# GROUND TRUTH DIGEST — Research Committee (Navi run)
Source of truth = GitHub repo artifacts (read 2026-05-30). NOT chat memory.

## REPO 1: forkedkv (29 commits, May 29 — main push of the week; 7.6MB)
Title: "Forkable GPU Memory for Replayable Agent Execution"
Mechanism (IMPLEMENTED + HW-TESTED on H100, CUDA 12.8, driver 580.82):
- Branch-aware copy-on-write of attention KV-cache pages via CUDA VMM driver API
  (cuMemCreate/cuMemMap/cuMemUnmap/cuMemRetainAllocationHandle). Fork aliases parent's
  physical HBM pages (refcounted, zero-copy); write to shared page detected in SOFTWARE
  (refcount>1 check) -> driver-level per-page remap. CoW unit = 2 MiB VMM page.
- src/: vmm_pool.py, kv_branch_manager.py, baseline_fullclone.py, replay.py, decode_layer.py
KEY MEASURED RESULTS (all cited to data/*.csv, reproducible):
- Capacity (M4): 12GiB prefix -> full-clone OOMs at 6 branches; CoW reaches 84 (14x), live
  HBM flat at 12GiB. OOM call forensically = cuMemSetAccess (VA-mapping-metadata exhaustion,
  NOT data memory).
- Predictable ceiling (M4b): branches x prefix_pages = CONSTANT ~520K (K/P model). Driver
  per-context mapping-table capacity ~520K entries on H100.
- Lab1: ceiling INDEPENDENT of vm.max_map_count (392 VMAs vs 67M limit). Pure driver limit.
- Attn overhead (M3): -0.1% to +1.1% (essentially zero) — FlashAttention/SDPA runs UNMODIFIED
  on forked branch because VA range is contiguous. vLLM-APC needs PagedAttention; we don't.
- Bytes written (M2b): 95%/90% fewer at 5%/10% tail divergence.
- Fork latency (M1): linear NOT flat; 1.3x faster than clone; map-op floor real (~50us/page).
- E2E (M5): 24 real SWE-bench-Verified instances -> 90% fewer KV bytes, 80% lower peak HBM.
- M5b: REAL Qwen2.5-7B FULL 28-layer autoregressive decode on CoW pages, 8 branches
  bit-identical CoW vs clone, peak HBM -44%.
- CoW cost: ~178us/page; D2D copy only 7%; map-op bound (cuMemSetAccess+Unmap).
HONEST RETRACTIONS (this repo is rigorous):
- R4: NOT a practical capacity/speed win vs strong software baseline (vLLM-APC/RadixAttention):
  software ~700x faster fork, ~6x larger capacity at 32-block prefix. RETRACTED the headline.
- Lab3b: corrected an earlier "2.3x kernel speedup" claim down to 5-22% vs FlashInfer.
- "page-fault-on-write" framing RETRACTED: detection is software, not HW GPU #PF.
- B8 NULL RESULT: R1 "47% removable bookkeeping" was WRONG; pooling removes only 3%.
WHAT SURVIVES (per repo's own R4 repositioning):
1. Forensic architectural CHARACTERIZATION of GPU VMM mapping ceiling (~520K, K/P model,
   cuMemSetAccess call site, vm.max_map_count independence) — useful regardless of deployment.
2. Physical KV sharing exposed as a CONTIGUOUS virtual address: unmodified SDPA on forked
   branch (the one thing software prefix-sharing cannot do without PagedAttention tax).

## REPO 2: edmm (10 commits, May 26-27; 648KB)
Title: "EDMM: Execution-Driven Memory Management for Agentic LLM Workloads"
Problem: tool-call mid-prompt injection breaks prefix-cache hash chain -> full KV recompute.
Measured 8.21x TTFT penalty through vLLM 0.6.6 LIVE engine (Qwen2.5-7B, H100).
Solution: CUDA VMM cuMemMap pointer-swap (~58us) instead of recompute (~274ms); speculative
prefill during tool-call idle window.
RESULTS: vLLM live: Radix penalty B/A=8.21x; EDMM recovery C/A=1.17x. Standalone: 4.18x->0.96x.
CUDA VMM micro: cuMemMap 58us vs memcpy+recompute 202us.
Integration: vLLM fork +14 lines/3 files +2 modules (~470 lines), 31 tests pass. SGLang baseline.
Has .cu correctness/perf proofs (mini_edmm_test.cu, mini_edmm_bench.cu).

## REPO 3: agent-failure-attribution-research (2 commits; meta-framework)
This is the COMMITTEE ORCHESTRATOR itself + its prior outputs. Heterogeneous 5-model committee
(claude/codex/metacode/gemini + spawn_agent), public-source citation verification (S2/OpenAlex/
arXiv/GitHub, >=2 sources to count a kill), hard "Pause" failsafe.
PRIOR COMMITTEE OUTPUTS (these ideas ALREADY EXIST — DO NOT re-propose as new):
- ASPLOS framing debate: consensus = "Forkable GPU Memory" angle, 35-45% ASPLOS accept prob.
- Phase 10 GREEN: IDEA_DET_8 "Determinism Bisect" (root-cause localization in trace divergence);
  IDEA_DET_6 "Long-Horizon Memory as Storage System (LSM-for-Agents)".
- Phase 11 GREEN: IDEA_RF_4 "AgentTraceStore" (trace-specialized storage format).
- YELLOW pool: Bisimulation Replay, Prompt-Git, CAP-for-Agents, Provenance-for-Reasoning,
  Agent-JIT, Decoder Telemetry Sidechannel, OpinionShift, PromptDelta, Trajectory Specialization.
- DEAD/RED: "agent partial-progress checkpoint protocol" (anticipated by Anthropic 2025 harness +
  LangGraph/CrewAI checkpointing — schema packaging, not a mechanism); AgentMQ.

## KNOWN PRIOR ART BOUNDARY (reviewers will weaponize these):
vAttention (ASPLOS'25, CUDA VMM for KV serving, read-only), ChunkAttention (ASPLOS'24, shared
prefix), PagedAttention/vLLM (SOSP'23), RadixAttention/SGLang (software refcount prefix tree),
POD-Attention (ASPLOS'25), CXLfork (ASPLOS'25, CPU-only fork), ServerlessLLM (OSDI'24),
LMCache, FlashInfer (production paged kernels). LangGraph/CrewAI (agent checkpointing).
