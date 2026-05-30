# DEAD THESIS REGISTRY
- T3-orig "Speculative prefill into tool-idle window": 3 RED (R2). Fatal: (1) needs an exact-token
  predictor for arbitrary tool returns (infeasible — SQL rows/API payloads/web scrape unpredictable);
  (2) under continuous batching, a request being idle does NOT make the GPU idle, so there is no
  reclaimable window. Killed by claude48/codex/metacode/gemini.
- "Contiguous-VA as kernel-compat substrate" (R1 cluster C): gemini RED on anti-FlashInfer —
  if FlashInfer's paged kernels are fast+complete, the "run unmodified SDPA" value evaporates.
  Survives only as a minor abstraction; not pursued.
- "VA-map concurrency/serialization wall" (claude48 T5, codex T5): RED — core claim UNMEASURED in
  either repo; hypothesis not thesis until a concurrent-mapper experiment runs.
- (Pre-existing, from prior repo committee — NOT re-proposed): Determinism Bisect, AgentTraceStore,
  LSM-for-Agents, agent partial-progress checkpoint protocol (RED: anticipated by Anthropic 2025
  harness + LangGraph/CrewAI), AgentMQ.
