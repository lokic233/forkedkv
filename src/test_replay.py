"""Test Priority-2 Replay + divergence detector with controlled nondeterminism."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from kv_branch_manager import KVBranchManager
from replay import Trajectory, ReplayEngine

def main():
    m = KVBranchManager(device_id=0)
    eng = ReplayEngine(m)
    # build a 6-page trajectory: KV steps, one RNG step, one TOOL step
    tr = Trajectory("agent_run", num_pages=6)
    tr.add_kv_step(0, payload_byte=10)
    tr.add_kv_step(1, payload_byte=20)
    tr.add_kv_step(2, payload_byte=30, rng_seed=12345)     # RNG-domain step
    tr.add_kv_step(3, payload_byte=40, tool_result="ls -> [a.py]")  # TOOL-domain step
    tr.add_kv_step(4, payload_byte=50)
    tr.add_kv_step(5, payload_byte=60)

    eng.record(tr, "orig")
    snap = m.snapshot("orig")
    print("[record] original trajectory materialized into 'orig'")

    # Replay 1: faithful (no modifiers) -> should fully match original (0 diverged pages)
    _, div0, fds0 = eng.replay(tr, "orig", snap, "replay_faithful", modifiers=None)
    print(f"[replay-faithful] diverged_pages={div0} first_div_step={fds0} (expect [] None)")
    assert div0 == [] and fds0 is None

    # Replay 2: modify RNG seed at step 2 -> KV diverges at page 2 onward (only page 2 written differently)
    _, div1, fds1 = eng.replay(tr, "orig", snap, "replay_rng", modifiers={2: {"rng_seed": 999}})
    print(f"[replay-rng] modified step 2 seed; diverged_pages={div1} first_div_step={fds1}")
    assert 2 in div1 and fds1 == 2

    # Replay 3: modify TOOL result at step 3 -> page 3 diverges
    _, div2, fds2 = eng.replay(tr, "orig", snap, "replay_tool", modifiers={3: {"tool_result": "ls -> [a.py, b.py]"}})
    print(f"[replay-tool] modified step 3 tool result; diverged_pages={div2} first_div_step={fds2}")
    assert 3 in div2 and fds2 == 3

    print("\nREPLAY TESTS PASSED")
    print("stats:", m.stats())

if __name__ == "__main__":
    main()
