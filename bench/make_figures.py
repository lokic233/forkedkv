"""Generate all figures from committed CSVs. Reproduce: python bench/make_figures.py"""
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

D = os.path.join(os.path.dirname(__file__), "..", "data")
F = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(F, exist_ok=True)

# Fig 1: fork latency vs prefix length (cow vs clone), with error bars
df = pd.read_csv(os.path.join(D, "metric1_fork_latency.csv"))
g = df.groupby(["prefix_mib","method"])["latency_us"].agg(["mean","std"]).reset_index()
plt.figure(figsize=(6,4))
for meth, lbl, c in [("cow_fork","CoW fork","C0"),("full_clone","Full clone","C3")]:
    s = g[g.method==meth].sort_values("prefix_mib")
    plt.errorbar(s.prefix_mib, s["mean"], yerr=s["std"], marker="o", label=lbl, color=c, capsize=3)
plt.xlabel("Prefix size (MiB)"); plt.ylabel("Fork latency (us)")
plt.title("Metric 1: Fork latency vs prefix length\n(H100, n=10, mean+/-sd)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(F,"metric1_fork_latency.png"), dpi=130); plt.close()

# Fig 2: bytes written + live HBM vs fanout
df = pd.read_csv(os.path.join(D, "metric2_bytes_written.csv"))
fig, ax = plt.subplots(1,2,figsize=(10,4))
for meth,lbl,c in [("cow_fork","CoW","C0"),("full_clone","Full clone","C3")]:
    s=df[df.method==meth].sort_values("fanout")
    ax[0].plot(s.fanout, s.bytes_written/2**20, "o-", label=lbl, color=c)
    ax[1].plot(s.fanout, s.live_phys_mib, "o-", label=lbl, color=c)
ax[0].set_title("KV bytes written per fork batch"); ax[0].set_xlabel("branch fanout"); ax[0].set_ylabel("MiB written")
ax[1].set_title("Live HBM footprint"); ax[1].set_xlabel("branch fanout"); ax[1].set_ylabel("MiB live")
for a in ax: a.legend(); a.grid(alpha=0.3)
plt.suptitle("Metric 2: 64 MiB prefix, no divergence (H100)"); plt.tight_layout()
plt.savefig(os.path.join(F,"metric2_bytes_written.png"), dpi=130); plt.close()

# Fig 2b: reduction vs divergence
df = pd.read_csv(os.path.join(D, "metric2b_divergence.csv"))
s = df[df.method=="cow_fork"].sort_values("divergence_frac")
plt.figure(figsize=(6,4))
plt.plot(s.divergence_frac*100, s.reduction_pct, "o-", color="C0")
plt.axhline(90, ls="--", color="gray", label="90% target")
plt.xlabel("% of prefix pages written per branch (divergence)")
plt.ylabel("KV bytes-written reduction vs full clone (%)")
plt.title("Metric 2b: Reduction vs divergence\n(32-page prefix, fanout 16, H100)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(F,"metric2b_divergence.png"), dpi=130); plt.close()

# Fig 3: attn overhead
df = pd.read_csv(os.path.join(D, "metric3_attn_overhead.csv"))
g = df.groupby(["seqlen","method"])["ms"].agg(["mean","std"]).reset_index()
plt.figure(figsize=(6,4))
for meth,lbl,c in [("contiguous","Contiguous KV","C1"),("vmm_paged","VMM-paged KV","C0")]:
    s=g[g.method==meth].sort_values("seqlen")
    plt.errorbar(s.seqlen, s["mean"], yerr=s["std"], marker="o", label=lbl, color=c, capsize=3)
plt.xlabel("KV sequence length (tokens)"); plt.ylabel("SDPA time (ms)")
plt.title("Metric 3: Attention kernel overhead (fp16, 32 heads, n=30)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(F,"metric3_attn_overhead.png"), dpi=130); plt.close()

# Fig 4: capacity bar
df = pd.read_csv(os.path.join(D, "metric4_capacity.csv"))
plt.figure(figsize=(5,4))
bars = plt.bar(df.method, df.branches_succeeded, color=["C3","C0"])
for b,oom in zip(bars, df.oom):
    plt.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
             "OOM" if oom else "cap", ha="center")
plt.ylabel("Branches before OOM/cap")
plt.title("Metric 4: Branch capacity on one H100\n(12 GiB shared prefix)")
plt.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(F,"metric4_capacity.png"), dpi=130); plt.close()

# Fig 5: e2e
df = pd.read_csv(os.path.join(D, "metric5_e2e.csv"))
fig, ax = plt.subplots(1,2,figsize=(11,4))
x = range(len(df)); labels=[i.split("__")[-1] for i in df.instance_id]
ax[0].bar([i-0.2 for i in x], df.cow_bytes_written/2**20, 0.4, label="CoW", color="C0")
ax[0].bar([i+0.2 for i in x], df.clone_bytes_written/2**20, 0.4, label="Full clone", color="C3")
ax[0].set_yscale("log"); ax[0].set_ylabel("KV bytes written (MiB, log)")
ax[0].set_xticks(list(x)); ax[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
ax[0].set_title("KV bytes written (fanout 8, 10% divergence)"); ax[0].legend()
ax[1].bar([i-0.2 for i in x], df.cow_s*1e3, 0.4, label="CoW", color="C0")
ax[1].bar([i+0.2 for i in x], df.clone_s*1e3, 0.4, label="Full clone", color="C3")
ax[1].set_ylabel("Wall time (ms)"); ax[1].set_xticks(list(x))
ax[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
ax[1].set_title("Branch wall time (~equal: CoW wins memory, not latency)"); ax[1].legend()
plt.suptitle("Metric 5: End-to-end on 7 real SWE-bench-Verified instances (H100)")
plt.tight_layout(); plt.savefig(os.path.join(F,"metric5_e2e.png"), dpi=130); plt.close()

print("wrote figures to", F)
for fn in sorted(os.listdir(F)): print(" ", fn)
