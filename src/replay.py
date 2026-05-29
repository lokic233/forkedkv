"""
replay.py — Replay a branch with controlled nondeterminism (Priority 2).

A *trajectory* is a recorded sequence of steps. Each step appends KV (advances pages)
and records a domain-tagged event:
  - KV   : attention KV-cache bytes (the forkable GPU pages)
  - RNG  : sampler RNG state (seed/counter) — replayed deterministically unless modified
  - TOOL : external tool-call log (stdout/result) — replayed from log unless modified

Replay(branch_id, modifiers): fork the branch from its snapshot, then re-execute its
recorded steps. `modifiers` injects controlled nondeterminism at a chosen step
(e.g. change RNG seed, or change a tool result). Pages before the divergence point are
shared via CoW (no copy); only steps at/after the modified step write new KV (CoW
copy), which the divergence detector then reports.

This is the rr-style record/replay idea (rr ATC'17) but the replay shares state with
the original via GPU-page CoW instead of re-running from scratch.
"""
import sys, os, hashlib, json
sys.path.insert(0, os.path.dirname(__file__))
from kv_branch_manager import KVBranchManager


class Step:
    __slots__ = ("idx", "domain", "page_index", "rng_seed", "tool_result", "payload_byte")
    def __init__(self, idx, domain, page_index, rng_seed=None, tool_result=None, payload_byte=0):
        self.idx = idx
        self.domain = domain          # "KV" | "RNG" | "TOOL"
        self.page_index = page_index  # which KV page this step writes
        self.rng_seed = rng_seed
        self.tool_result = tool_result
        self.payload_byte = payload_byte  # the byte we deterministically write to KV


class Trajectory:
    def __init__(self, name, num_pages):
        self.name = name
        self.num_pages = num_pages
        self.steps = []

    def add_kv_step(self, page_index, payload_byte, rng_seed=None, tool_result=None):
        dom = "KV"
        if rng_seed is not None: dom = "RNG"
        if tool_result is not None: dom = "TOOL"
        self.steps.append(Step(len(self.steps), dom, page_index,
                               rng_seed=rng_seed, tool_result=tool_result,
                               payload_byte=payload_byte))


class ReplayEngine:
    def __init__(self, mgr: KVBranchManager):
        self.mgr = mgr

    def record(self, traj: Trajectory, branch_id):
        """Materialize a trajectory's KV into a fresh branch (the 'original' run)."""
        self.mgr.create_branch(branch_id, traj.num_pages)
        for i in range(traj.num_pages):
            self.mgr.alloc_page(branch_id, i, fill_value=0)
        for s in traj.steps:
            b = self._effective_byte(s, modifiers=None)
            self.mgr.write_page(branch_id, s.page_index, fill_value=b)
        return branch_id

    def _effective_byte(self, step, modifiers):
        """Compute the KV byte a step writes, applying any modifier for this step."""
        if modifiers and step.idx in modifiers:
            mod = modifiers[step.idx]
            if "rng_seed" in mod:
                return (mod["rng_seed"] * 31 + step.idx) % 251 + 1
            if "tool_result" in mod:
                return (hash(mod["tool_result"]) % 251) + 1
            if "payload_byte" in mod:
                return mod["payload_byte"]
        # deterministic default
        if step.domain == "RNG" and step.rng_seed is not None:
            return (step.rng_seed * 31 + step.idx) % 251 + 1
        if step.domain == "TOOL" and step.tool_result is not None:
            return (hash(step.tool_result) % 251) + 1
        return step.payload_byte % 251

    def replay(self, traj: Trajectory, original_branch, snapshot, new_branch_id, modifiers=None):
        """Fork from snapshot and re-execute steps; modifiers inject nondeterminism.
        Returns (fork_handle, diverged_pages, first_divergent_step)."""
        fh = self.mgr.fork(snapshot, new_branch_id)
        first_div_step = None
        # Content-addressed CoW: only WRITE pages whose replayed value differs from the
        # original. Identical replays touch nothing -> zero CoW copies, alias preserved.
        # This makes the divergence detector report true value divergence, not "any write".
        for s in traj.steps:
            b = self._effective_byte(s, modifiers)
            orig_b = self._effective_byte(s, None)
            if b != orig_b:
                self.mgr.write_page(new_branch_id, s.page_index, fill_value=b)
                if first_div_step is None:
                    first_div_step = s.idx
        diverged = self.mgr.diverged_pages(original_branch, new_branch_id)
        return fh, diverged, first_div_step
