# E1 result (AMD MI350X gfx950, ROCm 6.4/7.0, devgpu499) — captured before box went offline
HIP_VMM_GRANULARITY=4096 bytes   (vs 2 MiB = 2097152 on H100/CUDA)
RESULT P=16 max_branches=400000 (OUR CAP) B_times_P=6400000 failing_call=none reserved_vas=400000

CONCLUSION: forkedkv's CUDA ~520K mapping ceiling does NOT reproduce on ROCm. AMD sustained
>=6.4M live VMM mappings with zero failure at our cap (12x past the CUDA wall). The 520K
invariant is NVIDIA-CUDA-driver-specific, NOT a cross-vendor hardware law.
=> Confirms Muse Park's T1 RED. T1 demoted to NVIDIA-scoped characterization.
TODO when devgpu499 back: raise cap, binary-search AMD true ceiling, sweep P=64/256/1024.
