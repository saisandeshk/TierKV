"""TierKV ShareGPT-slice analysis (additive, rerunnable).

Reads: runs/imported/sharegpt/ (sealed+verified slice: ShareGPT_Vicuna_unfiltered
       rev 192ab218, 6011-tok prefix, 2 blocks x 4 arms, nvme skipped by design).
Writes (additive only):
  results/summary.json                        appends "sharegpt" + bumps "rev"
  results/coverage.json                       appends "sharegpt" key
  results/figures/ttft-crossover-sharegpt.csv synthetic L6011 + sharegpt @6011

Rules: warmups never pooled; failed/crashed cells listed, never pooled;
every number records file+block provenance; no invented values.
Contamination: recompute_b1 non-clean files are superseded by the clean rerun
(see b1-contamination-note.txt); only *_clean.json enters stats. The dirty
device-hit artifact (ttft 0.139s, cached 6010) is recorded as excluded evidence.
Host_h2d_direct b0 meas spill failed (arrival comp None) + CRASH; the r1 retry is
an invalid race artifact (arrival http None, orch teardown mid-cell) and is
excluded; b1 meas reload/spill/fill0 PASS supply the arm's only valid point.
"""
import csv
import json
import os

from tierkv.analysis_cells import (
    bg_stats, mem_stats, emc_stats, arrival_stats, load_json,
)
from tierkv.analysis_stats import boot_ci

REPO = "/home/saisandeshk/Study/ISP/TierKV"
SH = os.path.join(REPO, "runs/imported/sharegpt")
RESULTS = os.path.join(REPO, "results")
FIGURES = os.path.join(RESULTS, "figures")
TOKENS = 6011
SEED0 = 20260922

# Valid meas cells only. (phase, block, file, warm-flag)
MEAS = {
    "retain": [("reload", 0, "retain_b0_meas_reload.json"),
               ("reload", 1, "retain_b1_meas_reload.json")],
    "host_d2h": [("reload", 0, "host_d2h_b0_meas_reload.json"),
                 ("reload", 1, "host_d2h_b1_meas_reload.json")],
    "recompute": [("recompute", 0, "recompute_b0_meas_recompute.json"),
                  ("recompute", 1, "recompute_b1_meas_recompute_clean.json")],
    "host_h2d_direct": [("reload", 1,
                         "host_h2d_direct_b1_meas_reload.json"),
                        ("spill", 1,
                         "host_h2d_direct_b1_meas_spill.json"),
                        ("fill0", 1,
                         "host_h2d_direct_b1_meas_fill0.json")],
}

EXCLUDED = {
    "recompute_b1_dirty": {
        "files": ["recompute_b1_base_bgonly.json",
                  "recompute_b1_warm_recompute.json",
                  "recompute_b1_meas_recompute.json"],
        "why": "timing contamination: duplicate driver ran concurrent cells "
               "on port 31503; server log shows 0 flush_cache successes, "
               "meas arrival is a device-hit artifact (ttft 0.1388s, cached "
               "6010). Superseded by *_clean.json single-server rerun.",
        "note_file": "b1-contamination-note.txt",
    },
    "host_h2d_direct_b0_meas": {
        "files": ["host_h2d_direct_b0_meas_spill.json"],
        "why": "meas spill arrival comp None + CRASH (abrupt server death, "
               "empty server log, no CUDA-illegal signature). Listed, not "
               "pooled.",
        "note_file": "b0-crash-note.txt",
    },
    "host_h2d_direct_b0_retry": {
        "files": ["host_h2d_direct_b0_meas_spill_r1.json",
                  "host_h2d_direct_b0_base_bgonly_retry.json"],
        "why": "invalid race artifact: reused orch server torn down mid-cell "
               "(arrival http None). Ignored per b0-crash-note addendum.",
        "note_file": "b0-crash-note.txt",
    },
}

METRICS = ["arrival_ttft_s", "bg_tpot_mean", "bg_gap_p95_mean", "bg_gap_max",
           "mem_dip_mb", "mem_baseline_mb", "emc_rate_mean", "emc_rate_max"]


def cell_record(phase, block, fname):
    d = load_json(os.path.join(SH, fname))
    rec = {"file": os.path.join(SH, fname), "phase": phase, "block": block,
           "warm": False}
    rec.update(bg_stats(d))
    rec.update(mem_stats(d))
    rec.update(emc_stats(d))
    rec.update(arrival_stats(d))
    return rec


def agg(recs, key, label, seed):
    vals = {r["block"]: r[key] for r in recs
            if r[key] is not None and r["block"] is not None}
    ci = boot_ci(list(vals.values()), seed=seed)
    ci["label"] = label
    ci["values_each"] = [vals[b] for b in sorted(vals)]
    ci["files"] = [r["file"] for r in recs if r[key] is not None]
    ci["blocks"] = sorted(vals)
    return ci


def main():
    os.makedirs(FIGURES, exist_ok=True)
    summ = load_json(os.path.join(SH, "sh_summary.json"))
    prev = load_json(os.path.join(RESULTS, "summary.json"))
    synth = prev["aggregates"]

    recs = {}
    for arm, cells in MEAS.items():
        recs[arm] = [cell_record(ph, b, f) for ph, b, f in cells]

    per_arm, comparison = {}, {}
    seed = SEED0
    for arm, rows in recs.items():
        per_arm[arm] = {"blocks": sorted(r["block"] for r in rows),
                        "files": [r["file"] for r in rows]}
        for m in METRICS:
            seed += 1
            # h2d spill/fill0 rows only feed their own phase keys below
            if arm == "host_h2d_direct":
                rs = [r for r in rows if r["phase"] == "reload"]
            else:
                rs = rows
            per_arm[arm][m] = agg(rs, m, "sharegpt %s %s" % (arm, m), seed)
        per_arm[arm]["cached"] = {
            "dev": [r["cached_device"] for r in rows
                    if r["phase"] == ("reload" if arm != "recompute"
                                      else "recompute")],
            "host": [r["cached_host"] for r in rows
                     if r["phase"] == ("reload" if arm != "recompute"
                                       else "recompute")],
        }
    # h2d spill + fill0 single-block characterization (n=1 each)
    for ph in ["spill", "fill0"]:
        rs = [r for r in recs["host_h2d_direct"] if r["phase"] == ph]
        for m in ["arrival_ttft_s", "bg_tpot_mean", "mem_dip_mb",
                  "mem_baseline_mb"]:
            seed += 1
            per_arm["host_h2d_direct_%s" % ph] = per_arm.get(
                "host_h2d_direct_%s" % ph, {})
            per_arm["host_h2d_direct_%s" % ph][m] = agg(
                rs, m, "sharegpt host_h2d_direct %s %s" % (ph, m), seed)

    # direction-vs-synthetic at matched 6011, same arm+phase+metric
    phase_of = {"retain": "reload", "host_d2h": "reload",
                "recompute": "recompute", "host_h2d_direct": "reload"}
    for arm in ["retain", "host_d2h", "recompute", "host_h2d_direct"]:
        ph = phase_of[arm]
        comparison[arm] = {}
        for m in METRICS:
            s = synth.get("%s_L6011_%s_%s" % (arm, ph, m), {})
            g = per_arm[arm][m]
            sm, gm = s.get("mean"), g.get("mean")
            if sm is None or gm is None:
                comparison[arm][m] = {"verdict": "not_comparable",
                                      "synthetic_mean": sm,
                                      "sharegpt_mean": gm}
                continue
            d = gm - sm
            pct = d / abs(sm) * 100.0 if sm != 0 else None
            # arm-ordering direction check uses TTFT/cache semantics;
            # here: does sharegpt preserve the synthetic sign of
            # (arm minus retain)? computed for ttft only at the end.
            comparison[arm][m] = {
                "synthetic_mean": sm,
                "synthetic_ci": [s.get("lo"), s.get("hi")],
                "sharegpt_mean": gm,
                "sharegpt_ci": [g.get("lo"), g.get("hi")],
                "sharegpt_n": g.get("n"),
                "delta_sharegpt_minus_synthetic": d,
                "delta_pct": pct,
                "synthetic_mean_in_sharegpt_ci":
                    (g.get("lo") is not None and g.get("lo") <= sm
                     <= g.get("hi")),
            }
    # ordering preserved? synthetic @6K: retain~=host_d2h (device-hit ~0.15s)
    # < recompute (cold ~1.33s) < h2d host-restore (~14.7s). Same in sharegpt?
    order = {}
    for src, get in [("synthetic",
                      lambda a: synth.get(
                          "%s_L6011_%s_arrival_ttft_s"
                          % (a, phase_of[a]), {}).get("mean")),
                     ("sharegpt", lambda a: per_arm[a]["arrival_ttft_s"].get(
                         "mean"))]:
        vals = {a: get(a) for a in phase_of}
        order[src] = vals
    ordering_same = (
        order["sharegpt"]["retain"] < 1.0
        and order["sharegpt"]["host_d2h"] < 1.0
        and 1.0 < order["sharegpt"]["recompute"] < 5.0
        and order["sharegpt"]["host_h2d_direct"] > 5.0)

    ci_note = ("n=2 per arm (n=1 host_h2d_direct reload: b0 crashed, valid "
               "retest is orch b1). Bootstrap 95% CIs over blocks are "
               "honestly wide by construction; no 10%-margin equivalence or "
               "superiority claims are made from ShareGPT alone.")
    caveats = [
        "recompute b0 mem_dip_mb is negative (-56.4MB: load-min above "
        "lead-mean) with lead baseline 13726MB vs clean-b1 17810MB; b0 ran "
        "under concurrent-server memory pressure with a mid-cell peer "
        "teardown freeing memory. The recompute mem_dip ShareGPT aggregate "
        "averages incommensurate baselines and is not interpretable; use "
        "per-block values.",
        "host_d2h b1 lead baseline 13685MB vs b0 17760MB (same concurrent-"
        "server cause); dip values stay small-positive (2.45/1.60MB).",
        "recompute b1 dirty files excluded (device-hit artifact); clean "
        "rerun used for all timing comparisons.",
    ]
    sharegpt = {
        "provenance": {
            "dataset": summ["provenance"]["dataset"],
            "dataset_rev": summ["provenance"]["dataset_rev_main"],
            "slice_sha256": summ["provenance"]["slice_sha256"],
            "slice_convos": summ["provenance"]["slice_convos"],
            "prefix_sha256": summ["victim_prefix"]["sha256"],
            "prefix_tokens": summ["victim_prefix"]["prompt_tokens"],
            "prefix_file": "prefix_sharegpt_6011.txt",
            "blocks": summ["blocks"],
            "arms": sorted(MEAS),
            "nvme": "skipped by design",
            "bg_note": summ.get("bg_note"),
            "corrections": summ.get("corrections"),
        },
        "coverage": {"meas_cells_complete": [7, 8],
                     "note": "7/8 arm x block reload/recompute meas cells "
                             "PASS (h2d b0 crashed); h2d b1 spill+fill0+"
                             "reload sub-phases PASS; nvme not planned",
                     "failed_listed_not_pooled": [
                         "host_h2d_direct_b0 meas spill (arrival comp None) "
                         "+ CRASH",
                     ]},
        "per_arm": per_arm,
        "vs_synthetic_L6011": comparison,
        "arm_ordering_TTFT": {"synthetic": order["synthetic"],
                              "sharegpt": order["sharegpt"],
                              "same_direction": ordering_same},
        "ci_honesty": ci_note,
        "caveats": caveats,
        "excluded": EXCLUDED,
        "ci_method": "block-mean 95pct percentile bootstrap, B=10000, "
                     "seed=%d+n" % SEED0,
    }

    # ---- additive summary.json update ----
    new = dict(prev)
    new["rev"] = prev.get("rev", 1) + 1 if "rev" in prev else 2
    new["sharegpt"] = sharegpt
    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(new, f, indent=1)

    # ---- additive coverage.json update ----
    cov = load_json(os.path.join(RESULTS, "coverage.json"))
    cov_new = dict(cov)
    cov_new["sharegpt"] = {
        "meas_cells_complete": [7, 8],
        "per_arm": {a: {"meas_pass": len(MEAS[a]) if a != "host_h2d_direct"
                        else 1,
                        "blocks": sorted({b for _, b, _ in MEAS[a]}),
                        "files": ["runs/imported/sharegpt/%s" % f
                                  for _, _, f in MEAS[a]]}
                    for a in MEAS},
        "failed": [{"cell": "host_h2d_direct_b0",
                    "arm": "host_h2d_direct", "block": 0, "phase": "meas/spill",
                    "note": "arrival comp None",
                    "file": "runs/imported/sharegpt/"
                            "host_h2d_direct_b0_meas_spill.json",
                    "server_log": "runs/imported/sharegpt/"
                                  "server-host_h2d_direct-blk0.log"},
                   {"cell": "host_h2d_direct_b0", "arm": "host_h2d_direct",
                    "block": 0, "phase": "CRASH", "note": "health fail",
                    "file": None,
                    "server_log": "runs/imported/sharegpt/"
                                  "server-host_h2d_direct-blk0.log"}],
        "superseded_invalid": EXCLUDED,
        "provenance": sharegpt["provenance"],
    }
    with open(os.path.join(RESULTS, "coverage.json"), "w") as f:
        json.dump(cov_new, f, indent=1)

    # ---- ttft-crossover-sharegpt.csv ----
    def _r(v):
        return round(v, 6) if isinstance(v, float) else v
    with open(os.path.join(FIGURES, "ttft-crossover-sharegpt.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "arm", "length", "tokens", "mean_ttft_s",
                    "lo_s", "hi_s", "n", "cached_note"])
        for arm in ["retain", "host_d2h", "recompute", "host_h2d_direct"]:
            ph = phase_of[arm]
            a = synth["%s_L6011_%s_arrival_ttft_s" % (arm, ph)]
            note = {"retain": "device-hit", "host_d2h": "device-hit",
                    "recompute": "cold",
                    "host_h2d_direct": "host-hit-direct"}[arm]
            w.writerow(["synthetic", arm, TOKENS, TOKENS, _r(a.get("mean")),
                        _r(a.get("lo")), _r(a.get("hi")), a.get("n"), note])
        for arm in ["retain", "host_d2h", "recompute", "host_h2d_direct"]:
            g = per_arm[arm]["arrival_ttft_s"]
            note = {"retain": "device-hit", "host_d2h": "device-hit",
                    "recompute": "cold",
                    "host_h2d_direct": "host-hit-direct"}[arm]
            w.writerow(["sharegpt", arm, TOKENS, TOKENS, _r(g.get("mean")),
                        _r(g.get("lo")), _r(g.get("hi")), g.get("n"), note])

    # ---- verdict lines ----
    for arm in ["retain", "host_d2h", "recompute", "host_h2d_direct"]:
        t = comparison[arm]["arrival_ttft_s"]
        print("%s TTFT: synth %.4f vs sharegpt %.4f (%+.1f%%) synth-in-ci=%s" %
              (arm, t["synthetic_mean"], t["sharegpt_mean"], t["delta_pct"],
               t["synthetic_mean_in_sharegpt_ci"]))
    print("ordering same direction (retain~=d2h<h2d slowest): %s"
          % ordering_same)
    print(ci_note)
    print("wrote sharegpt section (rev=%s) + coverage sharegpt + "
          "ttft-crossover-sharegpt.csv" % new["rev"])


if __name__ == "__main__":
    main()
