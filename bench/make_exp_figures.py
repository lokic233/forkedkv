import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/home/dengcchi/branchable_replay"
DATA = os.path.join(ROOT, "data"); FIG = os.path.join(ROOT, "figures")

# ---------------- EXP 1: TLB / cache overhead ----------------
rows = list(csv.DictReader(open(os.path.join(DATA, "exp1_tlb_pressure.csv"))))
seqlens = sorted({int(r["seqlen"]) for r in rows})
branches = sorted({int(r["branches"]) for r in rows})

def med(sl, nb, method):
    xs = [float(r["ms"]) for r in rows if int(r["seqlen"])==sl and int(r["branches"])==nb and r["method"]==method]
    return float(np.median(xs)) if xs else np.nan

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
# left: overhead % vs seqlen, one line per branch count
for nb in branches:
    ys = []
    for sl in seqlens:
        c, v = med(sl, nb, "contiguous"), med(sl, nb, "vmm_paged")
        ys.append(100*(v/c-1) if c==c and v==v else np.nan)
    ax1.plot(seqlens, ys, marker="o", label=f"{nb} branch")
ax1.axhline(0, color="k", lw=0.8, ls="--")
ax1.set_xscale("log", base=2)
ax1.set_xlabel("context length (tokens)"); ax1.set_ylabel("VMM-paged overhead vs contiguous (%)")
ax1.set_title("Exp 1: VMM-paged KV attention overhead\n(28-layer Qwen2.5-7B SDPA, fp16, H100)")
ax1.legend(fontsize=8, title="concurrent branches"); ax1.grid(alpha=0.3)
ax1.set_ylim(-3, 3)
# right: heatmap of overhead % over (seqlen x branches)
H = np.full((len(branches), len(seqlens)), np.nan)
for i, nb in enumerate(branches):
    for j, sl in enumerate(seqlens):
        c, v = med(sl, nb, "contiguous"), med(sl, nb, "vmm_paged")
        if c==c and v==v: H[i, j] = 100*(v/c-1)
im = ax2.imshow(H, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2, origin="lower")
ax2.set_xticks(range(len(seqlens))); ax2.set_xticklabels(seqlens)
ax2.set_yticks(range(len(branches))); ax2.set_yticklabels(branches)
ax2.set_xlabel("context length (tokens)"); ax2.set_ylabel("concurrent branches")
ax2.set_title("Overhead % heatmap\n(blue=faster, red=slower; ncu unavailable -> Event timing)")
for i in range(len(branches)):
    for j in range(len(seqlens)):
        if H[i,j]==H[i,j]:
            ax2.text(j, i, f"{H[i,j]:+.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, ax=ax2, label="overhead %")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "exp1_tlb_overhead.png"), dpi=130)
print("wrote exp1_tlb_overhead.png")
plt.close()

# ---------------- EXP 2: driver contention scaling ----------------
rows2 = list(csv.DictReader(open(os.path.join(DATA, "exp2_driver_contention.csv"))))
ths = sorted({int(r["threads"]) for r in rows2})
def get(t, op, col):
    return float([r for r in rows2 if int(r["threads"])==t and r["op"]==op][0][col])

fig, (a, b, c) = plt.subplots(1, 3, figsize=(17, 5))
# (a) aggregate throughput + ideal-linear reference
cps = [get(t, "full_cycle", "cycles_per_s") for t in ths]
ideal = [cps[0]*t/ths[0] for t in ths]
a.plot(ths, cps, marker="o", color="crimson", label="observed throughput")
a.plot(ths, ideal, ls="--", color="gray", label="ideal linear scaling")
a.set_xscale("log", base=2); a.set_yscale("log")
a.set_xlabel("concurrent threads"); a.set_ylabel("VMM cycles / sec")
a.set_title("Exp 2a: throughput does NOT scale\n(flat == driver serialization)")
a.legend(fontsize=9); a.grid(alpha=0.3, which="both")
# (b) full-cycle latency percentiles
for col, lab in [("p50_us","p50"),("p95_us","p95"),("p99_us","p99")]:
    a_y = [get(t, "full_cycle", col)/1e3 for t in ths]
    b.plot(ths, a_y, marker="o", label=lab)
b.set_xscale("log", base=2); b.set_yscale("log")
b.set_xlabel("concurrent threads"); b.set_ylabel("full map/unmap cycle latency (ms)")
b.set_title("Exp 2b: per-op latency cliff\n(latency grows ~linearly with threads)")
b.legend(fontsize=9); b.grid(alpha=0.3, which="both")
# (c) per-op p50 breakdown: which call serializes?
ops = ["map","unmap","setaccess","release","reserve","free","create"]
for op in ops:
    ys = [get(t, op, "p50_us") for t in ths]
    style = "-" if op in ("map","unmap","release") else "--"
    c.plot(ths, ys, marker=".", ls=style, label=op)
c.set_xscale("log", base=2); c.set_yscale("log")
c.set_xlabel("concurrent threads"); c.set_ylabel("per-op p50 latency (us)")
c.set_title("Exp 2c: page-table-mutating ops serialize\n(solid=map/unmap/release; dashed=VA-only stay flat)")
c.legend(fontsize=8, ncol=2); c.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "exp2_contention_scaling.png"), dpi=130)
print("wrote exp2_contention_scaling.png")
plt.close()
