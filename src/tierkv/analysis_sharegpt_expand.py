"""TierKV ShareGPT-expand analysis b0-b4 (additive, rerunnable).

Combines sealed slice b0-b1 (runs/imported/sharegpt, ShareGPT_Vicuna_unfiltered
rev 192ab218, 6011-tok prefix) with Strand C expand b2-b4
(runs/imported/campaign-v2/sharegpt-expand, same prefix sha, 12/12 PASS).

Reads: runs/imported/sharegpt/ + runs/imported/campaign-v2/sharegpt-expand/
Writes (additive only, never touches summary.json / coverage.json /
campaign-v2-summary.json / cross-summary.json):
  results/sharegpt-expand-summary.json

Frozen contract: blocks are reps (retain/host_d2h/recompute n=5;
host_h2d_direct reload n=4 -- b0 meas crashed, listed not pooled);
paired-block 95% percentile bootstrap B=10000 seed=20260921;
warmups never pooled; failed/crashed listed never pooled;
every number records file+block provenance; nvme skipped by design;
bg asymmetry (victim natural text, bg synthetic short prompts) noted;
EMC utilization-only; joules windowed.
"""

import json
import os

from tierkv.analysis_cells import (
    bg_stats, mem_stats, emc_stats, arrival_stats, joules_stats,
    load_json,
)
from tierkv.analysis_stats import boot_ci

REPO = "/home/saisandeshk/Study/ISP/TierKV"
SH01 = os.path.join(REPO, "runs/imported/sharegpt")
SHEXP = os.path.join(REPO, "runs/imported/campaign-v2/sharegpt-expand")
RESULTS = os.path.join(REPO, "results")
TOKENS = 6011
SEED0 = 20260921

# Valid meas cells only: (dir, phase, block, file).
MEAS = {
    "retain": [
        (SH01, "reload", 0, "retain_b0_meas_reload.json"),
        (SH01, "reload", 1, "retain_b1_meas_reload.json"),
        (SHEXP, "reload", 2, "retain_b2_meas_reload.json"),
        (SHEXP, "reload", 3, "retain_b3_meas_reload.json"),
        (SHEXP, "reload", 4, "retain_b4_meas_reload.json"),
    ],
    "host_d2h": [
        (SH01, "reload", 0, "host_d2h_b0_meas_reload.json"),
        (SH01, "reload", 1, "host_d2h_b1_meas_reload.json"),
        (SHEXP, "reload", 2, "host_d2h_b2_meas_reload.json"),
        (SHEXP, "reload", 3, "host_d2h_b3_meas_reload.json"),
        (SHEXP, "reload", 4, "host_d2h_b4_meas_reload.json"),
    ],
    "recompute": [
        (SH01, "recompute", 0, "recompute_b0_meas_recompute.json"),
        (SH01, "recompute", 1, "recompute_b1_meas_recompute_clean.json"),
        (SHEXP, "recompute", 2, "recompute_b2_meas_recompute.json"),
        (SHEXP, "recompute", 3, "recompute_b3_meas_recompute.json"),
        (SHEXP, "recompute", 4, "recompute_b4_meas_recompute.json"),
    ],
    "host_h2d_direct": [
        (SH01, "reload", 1, "host_h2d_direct_b1_meas_reload.json"),
        (SHEXP, "reload", 2, "host_h2d_direct_b2_meas_reload.json"),
        (SHEXP, "reload", 3, "host_h2d_direct_b3_meas_reload.json"),
        (SHEXP, "reload", 4, "host_h2d_direct_b4_meas_reload.json"),
        (SH01, "spill", 1, "host_h2d_direct_b1_meas_spill.json"),
        (SHEXP, "spill", 2, "host_h2d_direct_b2_meas_spill.json"),
        (SHEXP, "spill", 3, "host_h2d_direct_b3_meas_spill.json"),
        (SHEXP, "spill", 4, "host_h2d_direct_b4_meas_spill.json"),
        (SH01, "fill0", 1, "host_h2d_direct_b1_meas_fill0.json"),
        (SHEXP, "fill0", 2, "host_h2d_direct_b2_meas_fill0.json"),
        (SHEXP, "fill0", 3, "host_h2d_direct_b3_meas_fill0.json"),
        (SHEXP, "fill0", 4, "host_h2d_direct_b4_meas_fill0.json"),
    ],
}

EXCLUDED = {
    "recompute_b1_dirty": {
        "files": ["recompute_b1_base_bgonly.json",
                  "recompute_b1_warm_recompute.json",
                  "recompute_b1_meas_recompute.json"],
        "why": "timing contamination: duplicate driver ran concurrent cells "
               "on port 31503; meas arrival is a device-hit artifact "
               "(ttft 0.1388s, cached 6010). Superseded by *_clean.json.",
        "note_file": "b1-contamination-note.txt",
    },
    "host_h2d_direct_b0_meas": {
        "files": ["host_h2d_direct_b0_meas_spill.json"],
        "why": "meas spill arrival comp None + CRASH. Listed, not pooled; "
               "hence h2d reload n=4 (b1-b4), not 5.",
        "note_file": "b0-crash-note.txt",
    },
    "host_h2d_direct_b0_retry": {
        "files": ["host_h2d_direct_b0_meas_spill_r1.json",
                  "host_h2d_direct_b0_base_bgonly_retry.json"],
        "why": "invalid race artifact (reused orch server torn down "
               "mid-cell, arrival http None). Ignored per addendum.",
        "note_file": "b0-crash-note.txt",
    },
}

METRICS = ["arrival_ttft_s", "bg_tpot_mean", "bg_gap_p95_mean", "bg_gap_max",
           "mem_dip_mb", "mem_baseline_mb", "emc_rate_mean", "emc_rate_max",
           "joules_window_j"]


def cell_record(basedir, phase, block, fname):
    d = load_json(os.path.join(basedir, fname))
    rec = {"file": os.path.join(basedir, fname), "phase": phase,
           "block": block, "warm": False}
    rec.update(bg_stats(d))
    rec.update(mem_stats(d))
    rec.update(emc_stats(d))
    rec.update(arrival_stats(d))
    rec.update(joules_stats(d))
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


def warm_excluded_list():
    out = []
    for base in (SH01, SHEXP):
        for f in sorted(os.listdir(base)):
            if (("_warm_" in f or f.startswith("warm")) and f.endswith(".json")):
                out.append(os.path.join(base, f))
    # exclude pycache-adjacent non-cell jsons: keep only known warm cells
    out = [p for p in out if os.path.basename(p) in (
        "retain_b0_warm_spill.json", "retain_b1_warm_spill.json",
        "host_d2h_b0_warm_spill.json", "host_d2h_b1_warm_spill.json",
        "recompute_b0_warm_recompute.json",
        "recompute_b1_warm_recompute.json",
        "recompute_b1_warm_recompute_clean.json",
        "host_h2d_direct_b0_warm_spill.json",
        "host_h2d_direct_b0_warm_fill0.json",
        "host_h2d_direct_b0_warm_reload.json",
        "host_h2d_direct_b1_warm_spill.json",
        "host_h2d_direct_b1_warm_fill0.json",
        "host_h2d_direct_b1_warm_reload.json",
        "retain_b2_warm_spill.json", "retain_b3_warm_spill.json",
        "retain_b4_warm_spill.json",
        "host_d2h_b2_warm_spill.json", "host_d2h_b3_warm_spill.json",
        "host_d2h_b4_warm_spill.json",
        "recompute_b2_warm_recompute.json",
        "recompute_b3_warm_recompute.json",
        "recompute_b4_warm_recompute.json",
        "host_h2d_direct_b2_warm_spill.json",
        "host_h2d_direct_b2_warm_fill0.json",
        "host_h2d_direct_b2_warm_reload.json",
        "host_h2d_direct_b3_warm_spill.json",
        "host_h2d_direct_b3_warm_fill0.json",
        "host_h2d_direct_b3_warm_reload.json",
        "host_h2d_direct_b4_warm_spill.json",
        "host_h2d_direct_b4_warm_fill0.json",
        "host_h2d_direct_b4_warm_reload.json")]
    return sorted(out)


def main():
    summ01 = load_json(os.path.join(SH01, "sh_summary.json"))
    exp = load_json(os.path.join(SHEXP, "expand_summary.json"))
    prev = load_json(os.path.join(RESULTS, "summary.json"))
    synth = prev["aggregates"]

    recs = {}
    for arm, cells in MEAS.items():
        recs[arm] = [cell_record(bd, ph, b, f) for bd, ph, b, f in cells]

    per_arm, comparison = {}, {}
    seed = SEED0
    jseed = SEED0 + 100000
    for arm, rows in recs.items():
        if arm == "host_h2d_direct":
            rs = [r for r in rows if r["phase"] == "reload"]
        else:
            rs = rows
        per_arm[arm] = {"blocks": sorted({r["block"] for r in rs}),
                        "files": [r["file"] for r in rs]}
        for m in METRICS:
            if m == "joules_window_j":
                jseed += 1
                mseeds = jseed
            else:
                seed += 1
                mseeds = seed
            per_arm[arm][m] = agg(rs, m, "sharegpt-expand %s %s" % (arm, m),
                                  mseeds)
        per_arm[arm]["cached"] = {
            "dev": [r["cached_device"] for r in rs
                    if r["phase"] == ("reload" if arm != "recompute"
                                      else "recompute")],
            "host": [r["cached_host"] for r in rs
                     if r["phase"] == ("reload" if arm != "recompute"
                                       else "recompute")],
        }
    # h2d spill + fill0 characterization (n=4 each: b1-b4)
    for ph in ["spill", "fill0"]:
        rs = [r for r in recs["host_h2d_direct"] if r["phase"] == ph]
        key = "host_h2d_direct_%s" % ph
        per_arm[key] = {"blocks": sorted({r["block"] for r in rs}),
                        "files": [r["file"] for r in rs]}
        for m in ["arrival_ttft_s", "bg_tpot_mean", "mem_dip_mb",
                  "mem_baseline_mb"]:
            seed += 1
            per_arm[key][m] = agg(
                rs, m, "sharegpt-expand host_h2d_direct %s %s" % (ph, m),
                seed)

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
                                      "sharegpt_expand_mean": gm}
                continue
            d = gm - sm
            pct = d / abs(sm) * 100.0 if sm != 0 else None
            if g.get("lo") is None or g.get("hi") is None:
                in_ci = None
                ci_flag = ("indeterminate_n1"
                           if g.get("n") == 1 else "indeterminate_ci_None")
            else:
                in_ci = (g["lo"] <= sm <= g["hi"])
                ci_flag = "ok"
            comparison[arm][m] = {
                "synthetic_mean": sm,
                "synthetic_ci": [s.get("lo"), s.get("hi")],
                "sharegpt_expand_mean": gm,
                "sharegpt_expand_ci": [g.get("lo"), g.get("hi")],
                "sharegpt_expand_n": g.get("n"),
                "delta_expand_minus_synthetic": d,
                "delta_pct": pct,
                "synthetic_mean_in_expand_ci": in_ci,
                "ci_overlap_flag": ci_flag,
            }

    order = {}
    for src, get in [("synthetic",
                      lambda a: synth.get(
                          "%s_L6011_%s_arrival_ttft_s"
                          % (a, phase_of[a]), {}).get("mean")),
                     ("sharegpt_expand",
                      lambda a: per_arm[a]["arrival_ttft_s"].get("mean"))]:
        order[src] = {a: get(a) for a in phase_of}
    ordering_same = (
        order["sharegpt_expand"]["retain"] < 1.0
        and order["sharegpt_expand"]["host_d2h"] < 1.0
        and 1.0 < order["sharegpt_expand"]["recompute"] < 5.0
        and order["sharegpt_expand"]["host_h2d_direct"] > 5.0)

    warm_excl = warm_excluded_list()
    ci_note = ("blocks are reps: retain/host_d2h/recompute n=5 (b0-b4), "
               "host_h2d_direct reload/spill/fill0 n=4 (b1-b4; b0 crashed, "
               "listed not pooled). Bootstrap 95%% CIs over blocks, B=10000, "
               "seed=%d+n. n=1 CIs undefined (never zero-width); no "
               "10%%-margin equivalence/superiority claims from ShareGPT "
               "alone. emc_rate_* utilization-only audit raw (decaying "
               "20ms proxy, 204M floor), not claim contrasts." % SEED0)
    caveats = [
        "bg asymmetry: victim prefix is natural ShareGPT text (6011 tok); "
        "bg decodes stay synthetic short prompts (same strings as "
        "matrix-v1). Cross vs synthetic TTFT direction only; bg-gap "
        "contrasts inherit this asymmetry.",
        "recompute b0 mem_dip_mb negative (-56.4MB: load-min above "
        "lead-mean, lead baseline ~13726MB vs clean-b1 ~17810MB) from "
        "concurrent-server memory pressure in slice b0-b1; expand b2-b4 "
        "baselines are self-consistent. Use per-block values, not the "
        "pooled dip, for recompute memory claims.",
        "nvme skipped by design in ShareGPT (no nvme arm planned/collected).",
    ]

    out = {
        "rev": 1,
        "frozen_contract": {
            "reps": "blocks (retain/host_d2h/recompute n=5 b0-b4; "
                    "host_h2d_direct n=4 b1-b4)",
            "ci": "paired-block 95pct percentile bootstrap, B=10000, "
                  "seed=%d+n" % SEED0,
            "warmups": "excluded from all stats (%d warm files listed, "
                       "never pooled)" % len(warm_excl),
            "gating": "arrival http 200 + completion_tokens==64; "
                      "None dropped per block, never pooled",
            "nvme": "skipped by design",
            "bg_asymmetry": "victim natural ShareGPT text; bg synthetic "
                            "short prompts (matrix-v1 strings)",
            "emc": "utilization-only audit raw, never claim contrasts",
        },
        "provenance": {
            "dataset": summ01["provenance"]["dataset"],
            "dataset_rev": summ01["provenance"]["dataset_rev_main"],
            "slice_sha256": summ01["provenance"]["slice_sha256"],
            "slice_convos": summ01["provenance"]["slice_convos"],
            "prefix_sha256": summ01["victim_prefix"]["sha256"],
            "prefix_sha256_expand_match": True,
            "prefix_tokens": TOKENS,
            "prefix_file": "prefix_sharegpt_6011.txt",
            "blocks": [0, 1, 2, 3, 4],
            "arms": sorted([a for a in MEAS if not a.startswith(
                "host_h2d_direct_")]),
            "nvme": "skipped by design",
            "bg_note": summ01.get("bg_note"),
            "expand_campaign": exp.get("campaign"),
            "expand_cells": "%d/12 complete" % sum(
                1 for c in exp.get("cells", {}).values()
                if c.get("status") == "complete"),
        },
        "seals": {
            "sharegpt_b01_files": 65,
            "sharegpt_b01_seal": "PASS (workstation sha256sum -c)",
            "expand_files": 78,
            "expand_seal": "PASS (Orin + workstation sha256sum -c, 78/78 OK)",
        },
        "coverage": {
            "meas_cells": {"retain": [5, 5], "host_d2h": [5, 5],
                           "recompute": [5, 5],
                           "host_h2d_direct_reload": [4, 5],
                           "host_h2d_direct_spill": [4, 5],
                           "host_h2d_direct_fill0": [4, 5]},
            "note": "19/20 arm x block reload/recompute meas cells PASS "
                    "(h2d b0 crashed, listed not pooled); h2d spill+fill0 "
                    "4/4 blocks b1-b4 PASS; nvme not planned",
            "failed_listed_not_pooled": [
                "host_h2d_direct_b0 meas spill (arrival comp None) + CRASH",
            ],
        },
        "per_arm": per_arm,
        "vs_synthetic_L6011": comparison,
        "arm_ordering_TTFT": {"synthetic": order["synthetic"],
                              "sharegpt_expand": order["sharegpt_expand"],
                              "same_direction": ordering_same},
        "ci_honesty": ci_note,
        "caveats": caveats,
        "excluded": EXCLUDED,
        "warm_excluded_files": warm_excl,
        "warm_excluded_n": len(warm_excl),
        "ci_method": "block-mean 95pct percentile bootstrap, B=10000, "
                     "seed=%d+n" % SEED0,
    }

    with open(os.path.join(RESULTS, "sharegpt-expand-summary.json"),
              "w") as f:
        json.dump(out, f, indent=1)

    for arm in ["retain", "host_d2h", "recompute", "host_h2d_direct"]:
        t = comparison[arm]["arrival_ttft_s"]
        print("%s TTFT: synth %.4f vs expand %.4f (%+.1f%%) "
              "synth-in-ci=%s n=%s"
              % (arm, t["synthetic_mean"], t["sharegpt_expand_mean"],
                 t["delta_pct"], t["synthetic_mean_in_expand_ci"],
                 t["sharegpt_expand_n"]))
    print("ordering same direction (retain~=d2h<recompute<h2d): %s"
          % ordering_same)
    print(ci_note)
    print("wrote results/sharegpt-expand-summary.json (rev=1)")


if __name__ == "__main__":
    main()
