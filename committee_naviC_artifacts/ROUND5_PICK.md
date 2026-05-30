# ROUND 5 — 3rd-GREEN candidate selection

| Candidate | CC48 | CC47 | CC46 | Muse | Codex | Gemini | PICK count |
|---|---|---|---|---|---|---|---|
| I (unified A*+C* decision procedure) | RED→Y | YELLOW | YELLOW | YELLOW(pick) | YELLOW | YELLOW | Muse |
| **J (driver-handle physical provenance/attestation)** | YELLOW(pick) | YELLOW(pick) | YELLOW(pick) | RED | Y/G(pick) | RED | **CC48,CC47,CC46,Codex = 4/6** |
| own (VA-tax / throughput collapse) | — | — | — | — | — | Gemini | Gemini |

## CONSENSUS PICK: CANDIDATE J — "Driver-level physical provenance: cuMemRetainAllocationHandle
## provides a VERIFIABLE attestation that two tenants' GPU KV pages never physically aliased —
## a capability software refcounting structurally CANNOT emit."

## THE GATING EXPERIMENT (unanimous among J-pickers): External-verifier / untrusted-runtime test.
- Setup: serving runtime is UNTRUSTED, GPU driver is TRUSTED, an external verifier (separate
  trust domain) must prove cross-tenant KV-page non-aliasing WITHOUT trusting runtime bookkeeping.
- Measure: (1) HW: contaminated/tenant-A handle ≠ tenant-B handle at driver level, all reps;
  (2) adversarial BOOKKEEPING-LIE case: a malicious runtime reports "isolated" but actually
  aliased — does the driver-handle proof CATCH it while software refcount report PASSES the lie?
  (3) verification overhead.
- THE FATAL TEST (CC48/CC46): is cuMemRetainAllocationHandle a HARDWARE-ROOTED, forge-resistant
  value bound to the GPU (ideally CC/SPDM attestation), or a userspace integer the untrusted
  runtime can FABRICATE? If forge-resistant → 6/6 GREEN. If fabricable → J dies like E.
- Verifier-detectable lie rate target: HW 100%, SW 0%. Overhead <5%.

## DECISION: build + run E-J (the untrusted-runtime verifier experiment) on H100. Honest:
## this either turns E-E's residue into a real capability (3rd GREEN) or kills J cleanly.
