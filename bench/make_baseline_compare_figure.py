"""Generate figures for the software-baseline head-to-head (R4 P0-1).
Reads data/baseline_compare_m1_fork_latency.csv (the most informative one).
"""
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

D = os.path.join(os.path.dirname(__file__), "..", "data")
F = os.path.join(os.path.dirname(__file__), "..", "figures")

df = pd.read_csv(os.path.join(D, "baseline_compare_m1_fork_latency.csv"))

fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
for meth, label, color, marker in [
    ("software", "Software (vLLM-APC-style)", "C0", "o"),
    ("hardware_vmm", "Hardware (ForkedKV / CUDA VMM)", "C3", "s"),
]:
    s = df[df.method == meth].sort_values("prefix_blocks")
    ax.errorbar(
        s.prefix_blocks, s.mean_us, yerr=s.stddev_us,
        marker=marker, label=label, color=color, capsize=3, lw=1.5
    )

ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Prefix length (2 MiB blocks)")
ax.set_ylabel("Fork latency (us)")
ax.set_title(
    "Fork latency: software prefix sharing vs ForkedKV\n"
    "(H100, n=16/point, log-log; software is ~700x faster at 128 blocks)"
)
ax.grid(alpha=0.3, which="both")
ax.legend(loc="upper left")
plt.tight_layout()
out = os.path.join(F, "baseline_compare_m1_fork_latency.png")
plt.savefig(out, dpi=130)
print("wrote", out)

# Bar plot: capacity at 32-block prefix
fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
labels = ["Software\n(host RAM bound)", "Hardware\n(driver mapping ceiling K/P)"]
caps = [100000, 16250]
colors = ["C0", "C3"]
bars = ax.bar(labels, caps, color=colors)
ax.set_yscale("log")
ax.set_ylabel("Max concurrent branches at 32-block (64 MiB) prefix")
ax.set_title("Capacity comparison at 32-block prefix")
for b, c in zip(bars, caps):
    ax.text(b.get_x() + b.get_width()/2, b.get_height()*1.1,
            f"{c:,}", ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
out = os.path.join(F, "baseline_compare_m3_capacity.png")
plt.savefig(out, dpi=130)
print("wrote", out)
