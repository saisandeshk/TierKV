"""TierKV publication figures (vector PDF, IEEE single-column 3.5in).

Regenerates all figures from results/figures/*.csv (no raw-run access):
  mem-bars.pdf         Delta MemAvailable per arm x length, 95% CI whiskers
  ttft-crossover.pdf   resume flat vs recompute linear + host-restore points
                       + ShareGPT overlay markers at 6011
  tpot-concurrency.pdf bg TPOT base vs loaded at bg 2/4/8
  emc-timeline-sample.pdf one representative spill->evict->reload window
                       (host_h2d_direct L6011 b0 meas reload mc_all series)

Style: 3.5in width, all fonts >= 8pt, vector-only, tight bbox.
Deps: stdlib + numpy + matplotlib only.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = "/home/saisandeshk/Study/ISP/TierKV/results/figures"
W = 3.5  # IEEE single-column width, inches

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,  # embedded TrueType, stays vector
})


def read_csv(name):
    with open(os.path.join(FIG, name)) as f:
        return list(csv.DictReader(f))


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s (%d bytes)" % (path, os.path.getsize(path)))


ARMS = ["retain", "host_d2h", "recompute", "host_h2d_direct"]
PHASE = {"retain": "reload", "host_d2h": "reload",
         "recompute": "recompute", "host_h2d_direct": "reload"}


def fig_mem_bars():
    rows = read_csv("mem-bars.csv")
    get = {(r["arm"], int(r["length"]), r["phase"]): r for r in rows}
    Ls = [2048, 4096, 6011]
    fig, ax = plt.subplots(figsize=(W, 2.6))
    x = np.arange(len(Ls))
    wd = 0.18
    for i, arm in enumerate(ARMS):
        ms, lo, hi = [], [], []
        for L in Ls:
            r = get[(arm, L, PHASE[arm])]
            ms.append(float(r["mean_dip_mb"]))
            lo.append(float(r["lo_mb"]))
            hi.append(float(r["hi_mb"]))
        ms = np.array(ms)
        ax.bar(x + (i - 1.5) * wd, ms, wd, label=arm)
        ax.errorbar(x + (i - 1.5) * wd, ms,
                    yerr=[ms - np.array(lo), np.array(hi) - ms],
                    fmt="none", ecolor="black", capsize=2, elinewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in Ls])
    ax.set_xlabel("prefix length (tokens)")
    ax.set_ylabel("MemAvailable dip (MB)")
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    save(fig, "mem-bars.pdf")


def fig_ttft_crossover():
    rows = read_csv("ttft-crossover.csv")
    sh = read_csv("ttft-crossover-sharegpt.csv")
    mat = [r for r in rows if r["source"] == "matrix"]
    sweep = [r for r in rows if r["source"].startswith("sweepA")]
    fig, ax = plt.subplots(figsize=(W, 2.8))
    for arm, marker, ls in [("retain", "o", "-"),
                            ("host_d2h", "s", "--"),
                            ("recompute", "^", "-"),
                            ("host_h2d_direct", "v", "-")]:
        pts = sorted([r for r in mat if r["arm"] == arm],
                     key=lambda r: int(r["tokens"]))
        xs = np.array([int(r["tokens"]) for r in pts])
        ys = np.array([float(r["mean_ttft_s"]) for r in pts])
        lo = np.array([num(r["lo_s"]) for r in pts])
        hi = np.array([num(r["hi_s"]) for r in pts])
        ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt=marker + ls,
                    markersize=4, capsize=2, elinewidth=0.8, label=arm)
    # sweepA noload reps (light): resume flat + recompute linear
    for arm, mk in [("retain-resume", "."), ("recompute", "x")]:
        pts = [r for r in sweep if r["arm"] == arm]
        ax.scatter([int(r["tokens"]) for r in pts],
                   [float(r["mean_ttft_s"]) for r in pts],
                   s=9, marker=mk, c="gray", alpha=0.6)
    # ShareGPT overlay markers at 6011 (distinct, black-edged)
    for r in sh:
        if r["source"] != "sharegpt":
            continue
        m, lo, hi = float(r["mean_ttft_s"]), num(r["lo_s"]), num(r["hi_s"])
        ax.errorbar([int(r["tokens"]) + 120], [m], yerr=[[m - lo], [hi - m]],
                    fmt="D", markersize=5, color="black",
                    markeredgecolor="black",
                    capsize=2, elinewidth=0.8)
    # single proxy entry for the overlay set
    ax.scatter([], [], s=25, marker="D", c="black",
               label="sharegpt @6011")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("prefix tokens")
    ax.set_ylabel("arrival TTFT (s)")
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    fig.tight_layout()
    save(fig, "ttft-crossover.pdf")


def fig_tpot_concurrency():
    rows = read_csv("tpot-concurrency.csv")
    fig, ax = plt.subplots(figsize=(W, 2.6))
    base, hit, cold = {}, {}, {}
    for r in rows:
        bgn, y = int(r["bg_n"]), float(r["mean_tpot_s"])
        src, arr = r["source"], r["arrival"]
        if arr == "none":
            base.setdefault(bgn, []).append(y)
        elif arr in ("hit-6K",):
            hit.setdefault(bgn, []).append(y)
        elif arr in ("cold-6K",):
            cold.setdefault(bgn, []).append(y)
        # host-restore-6K handled as off-scale annotation below
    for d, mk, ls, lab in [(base, "o", "-", "base (no arrival)"),
                           (hit, "s", "--", "loaded device-hit"),
                           (cold, "^", "--", "loaded cold prefill")]:
        xs = sorted(d)
        ax.errorbar(xs, [np.mean(d[x]) for x in xs], fmt=mk + ls,
                    markersize=4, capsize=2, elinewidth=0.8, label=lab)
    ax.set_xticks([2, 4, 8])
    ax.set_xlabel("background streams (bg_n)")
    ax.set_ylabel("bg TPOT (s)")
    ax.set_ylim(0.028, 0.046)
    ax.annotate("host-restore 0.145 s (off scale; gapmax 14.6 s)",
                xy=(4, 0.0445), xytext=(0.98, 0.96),
                textcoords="axes fraction", fontsize=8, ha="right",
                va="top", bbox=dict(fc="white", ec="none", pad=1),
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.legend(loc="center right")
    fig.tight_layout()
    save(fig, "tpot-concurrency.pdf")


def fig_emc_timeline():
    rows = read_csv("emc-timeline-sample.csv")
    h2d = [(float(r["t_rel_s"]), float(r["mc_all"])) for r in rows
           if r["arm"] == "host_h2d_direct"]
    h2d.sort()
    fig, ax = plt.subplots(figsize=(W, 2.6))
    t = np.array([p[0] for p in h2d])
    v = np.array([p[1] for p in h2d])
    ax.plot(t, v / 1e6, lw=1.0, label="host_h2d_direct L6011 b0 reload")
    ax.axvline(0.0, color="black", lw=0.8, ls="--")
    ax.text(0.6, 1.9, "t0 (arrival)", fontsize=8, va="top")
    ax.text(7.0, 0.35, "evict window", fontsize=8, ha="center")
    ax.text(16.2, 1.55, "host restore", fontsize=8, ha="center")
    ax.set_xlabel("t - t0 (s)")
    ax.set_ylabel("mc_all (Mcounts)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "emc-timeline-sample.pdf")


def main():
    fig_mem_bars()
    fig_ttft_crossover()
    fig_tpot_concurrency()
    fig_emc_timeline()


if __name__ == "__main__":
    main()
