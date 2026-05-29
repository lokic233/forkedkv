# Related Work & Mechanism Deltas

Each entry states what the prior work does and the *specific* mechanism we add.

## vAttention — ASPLOS '25 (Prabhu et al.)
Backs vLLM-style KV cache with CUDA VMM (cuMemCreate/cuMemMap) so KV is *virtually*
contiguous (no PagedAttention block table needed) while physically paged on demand.
**Our delta:** vAttention uses VMM for on-demand growth of ONE sequence's KV. We use
the same VMM substrate but add *branch-aware copy-on-write*: multiple branches' VA
ranges alias the SAME physical handles (refcounted), and writes trigger per-page CoW
remap. vAttention has no notion of forking/sharing pages across sequences.

## ChunkAttention — ASPLOS '24 (Ye et al.)
Prefix-sharing for KV via a trie of KV chunks; shared system-prompt prefixes reuse KV.
**Our delta:** ChunkAttention shares *read-only* prefix KV across requests using a
software trie + a fused kernel. It does not provide write isolation: once a sequence
diverges it allocates fresh KV and the kernel must know the chunk boundaries. We share
at the *hardware page* level and provide transparent CoW on write — the attention
kernel sees an ordinary contiguous tensor (Metric 3 confirms ~0 kernel overhead),
no kernel modification required.

## vLLM / PagedAttention — SOSP '23 (Kwon et al.)
Paged KV with an integer block table; copy-on-write at *block* granularity for parallel
sampling (beam search) within one request, implemented by ref-counting logical blocks
in a pre-reserved tensor pool.
**Our delta:** vLLM's CoW is a software block table over a single torch tensor pool;
"forking" a whole sequence to a new branch still requires copying blocks into the new
sequence's table or sharing only within one request's sampler. We move CoW to the GPU
MMU (cuMemMap remap), so branches are independent sequences that alias physical HBM and
diverge per-page. Our baseline (`baseline_fullclone.py`) mimics naive vLLM full-sequence
cloning; Metric 4 shows it OOMs at 6 branches where ours reaches 64.

## SGLang / RadixAttention — NeurIPS '24 (Zheng et al.)
RadixAttention keeps a radix tree of KV prefixes for automatic cross-request reuse.
**Our delta:** RadixAttention is read-sharing of cached prefixes keyed by token content;
it is an eviction/cache-reuse policy. We provide *mutable* branch state with isolation:
each branch can write and only its written pages diverge. Complementary — a radix tree
could choose WHICH snapshot to fork; we provide the forking mechanism.

## CXLfork — ASPLOS '25
Fast process forking over CXL-attached memory using hardware-assisted CoW across the
CXL fabric.
**Our delta:** Same CoW-fork philosophy but on the GPU memory hierarchy (HBM via the
GPU MMU through CUDA VMM), not CXL host memory. The forked entity is an agent's KV
state on-accelerator, not an OS process.

## ServerlessLLM — OSDI '24 (Fu et al.)
Fast checkpoint/restore of LLM state (incl. KV) for serverless cold starts via tiered
storage and locality-aware loading.
**Our delta:** ServerlessLLM snapshots to/from storage to migrate/restart a model. We
snapshot *in place* on the GPU at a causal boundary and fork many live branches that
share HBM; no serialization to storage. Our Snapshot is O(#pages) handle-refs, not a
byte dump.

## rr — USENIX ATC '17 (O'Callahan et al.)
Record-and-replay debugger: records nondeterministic inputs (syscalls, signals, RNG)
and deterministically replays a process from the log.
**Our delta:** We borrow rr's record/replay-with-controlled-nondeterminism idea
(`src/replay.py` records KV/RNG/TOOL domain events and replays with modifiers) but the
replay *shares state* with the original run via GPU-page CoW instead of re-executing
from scratch; only the divergent suffix recomputes. rr is CPU-process granularity and
copies nothing on the GPU.
