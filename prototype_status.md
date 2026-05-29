# Prototype Status — Forkable GPU Memory for Replayable Agent Execution

Last updated: (in progress)

## Engineering decisions (committed early)

### D1: Standalone KV manager, NOT vLLM block-manager patch
**Decision:** Build a standalone paged KV manager backed by CUDA VMM, integrated with
HuggingFace `transformers` for real model attention.
**Why:**
- vLLM's PagedAttention allocates KV blocks from a single pre-reserved tensor pool
  (`torch.zeros`) addressed by integer block tables. Forking at the *virtual address*
  level (the ASPLOS angle: GPU MMU CoW) requires owning the allocation via the CUDA
  VMM driver API (cuMemCreate/cuMemMap). Patching vLLM to swap its pool for VMM-mapped
  memory is a multi-week surgery against a fast-moving codebase and obscures the
  mechanism we are evaluating.
- A standalone manager lets the *page* be the unit of CoW and lets us alias physical
  handles across branches via cuMemMap — exactly what the paper claims.
- We still measure against a "full-clone" baseline that mimics vLLM's behavior
  (allocate fresh pool + copy), so the comparison is fair.
- vAttention (ASPLOS'25) showed VMM-backed KV is viable for production attention;
  we reuse that insight but add *branch-aware CoW*, which vAttention does not do.

### D2: Model = Qwen2.5-7B (fallback Llama-3.1-8B). 1 H100 (device 0).
### D3: CoW unit = one KV "page" = one cuMemCreate physical handle (granularity = 2MiB,
   the CUDA VMM min granularity on H100). KV blocks pack into pages.
### D4: Page-fault-on-write is implemented *explicitly* (software MMU): a write to an
   aliased page is intercepted by the manager, which allocates a private physical
   handle, remaps the page's VA to it, copies contents, decrements refcount on the old
   handle. True hardware #PF-on-write to read-only GPU mappings is NOT exposed by the
   CUDA VMM API at user level (no SIGSEGV handler path for device mappings), so we
   document this as "software-enforced CoW over hardware VA remapping." See LIMITATIONS.md.
