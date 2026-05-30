"""
analyze_lab2_ncu.py — read data/lab2_ncu_counters.csv, summarize per-metric
medians for contig vs vmm, print a table that can drop directly into the writeup.
"""
import os, csv, statistics
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN  = os.path.join(ROOT, "data", "lab2_ncu_counters.csv")
OUT = os.path.join(ROOT, "data", "lab2_ncu_summary.csv")
TXT = os.path.join(ROOT, "data", "lab2_ncu_summary.txt")

KEY_METRICS = [
    ("gpu__time_duration.sum",                                 "kernel_ns"),
    ("sm__throughput.avg.pct_of_peak_sustained_elapsed",       "sm_throughput_pct"),
    ("dram__throughput.avg.pct_of_peak_sustained_elapsed",     "dram_throughput_pct"),
    ("lts__t_sector_hit_rate.pct",                             "l2_hit_rate_pct"),
    ("lts__t_sectors_aperture_device.sum",                     "l2_sectors_device"),
    ("lts__t_sectors_aperture_peer.sum",                       "l2_sectors_peer"),
    ("lts__t_sectors_aperture_sysmem.sum",                     "l2_sectors_sysmem"),
    ("smsp__inst_executed.sum",                                "instructions_executed"),
    ("l1tex__t_sector_hit_rate.pct",                           "l1_hit_rate_pct"),
]

def fnum(s):
    return float(str(s).replace(",", ""))

def main():
    if not os.path.exists(IN):
        print(f"missing {IN}"); return
    rows = list(csv.DictReader(open(IN)))
    # only SDPA flash kernel(s)
    rows = [r for r in rows if "sdpa" in r["kernel"].lower() or "flash" in r["kernel"].lower()]
    by_cell = defaultdict(list)  # (method, seqlen, metric) -> [values]
    for r in rows:
        try: v = fnum(r["value"])
        except ValueError: continue
        by_cell[(r["method"], int(r["seqlen"]), r["metric"])].append(v)

    seqlens = sorted({k[1] for k in by_cell})
    out_rows = []
    lines = []
    lines.append("Lab 2 — Hardware counter evidence: contiguous vs VMM-paged SDPA")
    lines.append("="*108)
    for sl in seqlens:
        lines.append(f"\nseqlen = {sl}")
        lines.append(f"  {'metric':<30} {'contig (median)':>18} {'vmm (median)':>16} {'rel_diff':>10}  {'verdict':<20}")
        lines.append("  " + "-"*100)
        for mt, label in KEY_METRICS:
            vc = by_cell.get(("contig", sl, mt))
            vv = by_cell.get(("vmm", sl, mt))
            if not vc or not vv:
                continue
            mc = statistics.median(vc); mv = statistics.median(vv)
            rel = (mv - mc) / mc * 100 if mc else float('nan')
            verdict = ""
            if abs(rel) < 1.0:    verdict = "identical (<1%)"
            elif abs(rel) < 3.0:  verdict = "near-identical (<3%)"
            elif abs(rel) < 10.0: verdict = "mild diff (<10%)"
            else:                 verdict = "DIFFERS"
            lines.append(f"  {label:<30} {mc:>18.4g} {mv:>16.4g} {rel:>+9.2f}%  {verdict:<20}")
            out_rows.append(dict(seqlen=sl, metric=mt, contig_median=mc, vmm_median=mv, rel_diff_pct=rel,
                                 contig_n=len(vc), vmm_n=len(vv)))
    txt = "\n".join(lines)
    print(txt)
    with open(TXT, "w") as f: f.write(txt + "\n")
    if out_rows:
        with open(OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
        print(f"\nwrote {TXT} and {OUT}")

if __name__ == "__main__":
    main()
