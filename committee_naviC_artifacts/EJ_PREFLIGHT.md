# E-J PRE-FLIGHT — the forge-resistance fatal test (decides Candidate J)
Probe: /tmp/ej_probe.py on H100, cuda-python. Bounded (3 pages).

## MEASURED:
- handle(vA)=28901248, handle(vA2 aliasing vA)=28901248  → EQUAL (alias detection WORKS)
- handle(vB distinct)=28902352 → DIFFERS (correct)
- re-retain stable: True.
- **handle == int(original cuMemCreate handle pA): TRUE (28901248 == int(pA)).**
- freed-handle slot recycle: new handle did NOT collide (28918128≠28901248) — refs are live-scoped.

## VERDICT: CANDIDATE J FAILS THE FATAL TEST → KILL (same death as E).
cuMemRetainAllocationHandle returns a PROCESS-LOCAL OPAQUE DRIVER REFERENCE whose value is the
SAME integer as the cuMemCreate handle the runtime already holds. It is NOT a hardware-rooted,
cryptographically-bound, GPU-attested token. Consequences for J's threat model:
- The handle is a userspace-meaningful integer the allocator-owning runtime ALREADY possesses.
  A malicious runtime in the same context can read/report/fabricate these values freely.
- It is NOT bound to a CC/SPDM attestation report; an EXTERNAL verifier in a separate trust
  domain gets no forge-resistant evidence — exactly CC48's fatal condition ("if the handle turns
  out to be a local userspace integer the untrusted runtime can fabricate → J dies exactly like E").
- The aliasing proof is genuine as an INTERNAL self-check (equal-for-alias, differ-for-distinct),
  but that is the SAME bit-identical-isolation property E-E already showed software matches.

## So J collapses for the SAME structural reason as E: no capability software/runtime lacks once
## you don't have a hardware root of trust. The driver handle is a verifiability NICETY, not an
## attestation. Confirmed empirically, not assumed. Candidate J: KILLED.

## IMPLICATION: of the 3rd-GREEN candidates, I (decision procedure) and Gemini's throughput-
## collapse are the survivors. Both are HONEST extensions of A*+C* rather than new capabilities.
## Next: pursue the strongest GREEN-able framing of I/throughput, OR accept 2 GREEN + honest
## report that the 3rd does not exist on this evidence (the committee's own anti-coping standard).
