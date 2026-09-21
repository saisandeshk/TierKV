"""TierKV frozen-matrix analysis driver.

Reads: runs/imported/{matrix-v1,extras,phase1,crash-probe,v4h-load} (+ configs).
Writes (results/ only):
  results/summary.json            all contrasts + CIs + n (+coverage, crossover,
                                  discrepancy, duplication, underpowered)
  results/coverage.json           cell-by-cell audit (failed/crashed stay listed)
  results/figures/mem-bars.csv
  results/figures/emc-timeline-sample.csv
  results/figures/ttft-crossover.csv
  results/figures/tpot-concurrency.csv

Rules enforced: warmups never pooled (warm=True excluded from every
aggregate); failed/crashed cells listed, never pooled; every aggregate
records its file+block provenance; no invented numbers (None where missing).
"""
import csv
import json
import os
import glob

from tierkv.analysis_cells import (
    MATRIX, EXTRAS, PHASE1, IMPORTED,
    load_json, list_matrix_cells, cell_metrics, coverage_audit,
)
from tierkv.analysis_stats import boot_ci, paired_contrast, margin_test, linfit

RESULTS = "/home/saisandeshk/Study/ISP/TierKV/results"
FIGURES = os.path.join(RESULTS, "figures")
LENGTHS = [2048, 4096, 6011]
BLOCKS = [0, 1, 2, 3, 4]
ARMS = ["retain", "host_d2h", "host_h2d_direct", "recompute", "nvme_spill"]
SEED0 = 20260921


def by_block(recs, key):
    return {r["block"]: r[key] for r in recs
            if r[key] is not None and r["block"] is not None}


def agg(recs, key, label, seed):
    vals = by_block(recs, key)
    ci = boot_ci(list(vals.values()), seed=seed)
    ci["label"] = label
    ci["files"] = sorted(r["file"] for r in recs if r[key] is not None)
    ci["blocks"] = sorted(vals)
    return ci


def main():
    os.makedirs(FIGURES, exist_ok=True)
    recs = [cell_metrics(c["file"]) for c in list_matrix_cells()]
    meas = [r for r in recs if not r["warm"]]
    warm = [r for r in recs if r["warm"]]

    # ---------------- 1. COVERAGE AUDIT ----------------
    audit = coverage_audit()
    # meas-phase pass accounting per arm x length (warm kept separate)
    cov = {}
    for arm in ARMS:
        for L in LENGTHS:
            mrows = [r for r in audit["rows"]
                     if r["arm"] == arm and r["length"] == L
                     and not r["warm"] and r["phase"] != "CRASH"]
            wants = {"retain": ["meas"], "host_d2h": ["meas"],
                     "recompute": ["meas"], "host_h2d_direct": ["meas"],
                     "nvme_spill": ["meas"]}[arm]
            # nvme has no meas rows at all -> 0 coverage
            npass = sum(1 for r in mrows if r["pass"])
            # expected meas result entries per block:
            exp_per_block = {"retain": 1, "host_d2h": 1, "recompute": 1,
                             "host_h2d_direct": 1, "nvme_spill": 0}[arm]
            cov["%s_L%d" % (arm, L)] = {
                "meas_pass": npass, "meas_entries": len(mrows),
                "expected_meas_per_block": exp_per_block,
                "blocks": BLOCKS,
            }
    n_cells_ok = sum(1 for k, v in audit["cells"].items()
                     if v["arm"] != "nvme_spill" and v["status"] == "complete")
    coverage_pct = {"meas_cells_complete_non_nvme": [n_cells_ok, 60],
                    "pct": n_cells_ok / 60 * 100.0,
                    "planned_cells": 75, "nvme_meas_cells": [0, 15]}
    with open(os.path.join(RESULTS, "coverage.json"), "w") as f:
        json.dump({"coverage_pct": coverage_pct, "per_arm_length": cov,
                   "failed": audit["failed"],
                   "infra_incidents": audit["infra_incidents"],
                   "all_rows": audit["rows"]}, f, indent=1)

    # ---------------- 2. PAIRED-BLOCK AGGREGATES (meas only) ----------------
    def sel(arm, L, phase):
        return sorted([r for r in meas if r["arm"] == arm and r["length"] == L
                       and r["phase"] == phase], key=lambda r: r["block"])

    aggregates = {}   # label -> ci dict
    seed = SEED0
    metrics = ["arrival_ttft_s", "bg_tpot_mean", "bg_gap_p95_mean",
               "bg_gap_max", "mem_dip_mb", "mem_baseline_mb",
               "emc_rate_mean", "emc_rate_max"]
    for arm in ["retain", "host_d2h", "recompute", "host_h2d_direct"]:
        for L in LENGTHS:
            phase = {"retain": "reload", "host_d2h": "reload",
                     "recompute": "recompute",
                     "host_h2d_direct": "reload"}[arm]
            rows = sel(arm, L, phase)
            for m in metrics:
                seed += 1
                aggregates["%s_L%d_%s_%s" % (arm, L, phase, m)] = agg(
                    rows, m, "%s L%d %s %s" % (arm, L, phase, m), seed)
            # cached split trace (constant per cell; record blocks+files)
            aggregates["%s_L%d_%s_cached" % (arm, L, phase)] = {
                "dev": [r["cached_device"] for r in rows],
                "host": [r["cached_host"] for r in rows],
                "blocks": [r["block"] for r in rows],
                "files": [r["file"] for r in rows],
            }
    # h2d_direct spill + fill0 mem dips (eviction-path characterization)
    for L in LENGTHS:
        for phase in ["spill", "fill0"]:
            rows = sel("host_h2d_direct", L, phase)
            for m in ["mem_dip_mb", "mem_baseline_mb", "arrival_ttft_s",
                      "bg_tpot_mean"]:
                seed += 1
                aggregates["host_h2d_direct_L%d_%s_%s" % (L, phase, m)] = agg(
                    rows, m, "host_h2d_direct L%d %s %s" % (L, phase, m),
                    seed)
    # nvme warm-spill reference (warm: excluded from stats, listed only)
    nvme_ref = {}
    for L in LENGTHS:
        rows = sorted([r for r in warm if r["arm"] == "nvme_spill"
                       and r["length"] == L and r["phase"] == "spill"],
                      key=lambda r: r["block"])
        nvme_ref["nvme_spill_L%d_warm_spill" % L] = {
            "mem_dip_mb_each": [r["mem_dip_mb"] for r in rows],
            "blocks": [r["block"] for r in rows],
            "files": [r["file"] for r in rows],
            "excluded_warm": True,
        }
    # warm-spill cold TTFT reference (excluded, never pooled)
    cold_ref = {}
    for arm in ["retain", "host_d2h"]:
        for L in LENGTHS:
            rows = sorted([r for r in warm if r["arm"] == arm
                           and r["length"] == L and r["phase"] == "spill"],
                          key=lambda r: r["block"])
            cold_ref["%s_L%d_warm_spill_cold_ttft" % (arm, L)] = {
                "ttft_each": [r["arrival_ttft_s"] for r in rows],
                "blocks": [r["block"] for r in rows],
                "files": [r["file"] for r in rows],
                "excluded_warm": True,
            }

    # ---------------- 2b. PAIRED CONTRASTS + 10% MARGIN ----------------
    contrasts = {}

    def contrast(name, arm_x, arm_y, L, metric, phase_x=None,
                 phase_y=None, ref_arm="y"):
        rx = sel(arm_x, L, phase_x or
                 {"retain": "reload", "host_d2h": "reload",
                  "recompute": "recompute",
                  "host_h2d_direct": "reload"}[arm_x])
        ry = sel(arm_y, L, phase_y or
                 {"retain": "reload", "host_d2h": "reload",
                  "recompute": "recompute",
                  "host_h2d_direct": "reload"}[arm_y])
        bx, by = by_block(rx, metric), by_block(ry, metric)
        pc = paired_contrast(bx, by, seed=SEED0 + hash(name) % 10000)
        ref = aggregates.get("%s_L%d_%s_%s" % (
            arm_y, L, (phase_y or {"retain": "reload",
                                   "host_d2h": "reload",
                                   "recompute": "recompute",
                                   "host_h2d_direct": "reload"}[arm_y]),
            metric), {}).get("mean")
        mt = margin_test(pc, ref)
        contrasts[name] = {
            "x": "%s L%d" % (arm_x, L), "y": "%s L%d" % (arm_y, L),
            "metric": metric, "paired_diff": pc,
            "reference_mean_y": ref, "margin_10pct": mt,
            "x_files": sorted(r["file"] for r in rx),
            "y_files": sorted(r["file"] for r in ry),
        }

    for L in LENGTHS:
        contrast("resume_ttft_host_d2h-retain_L%d" % L,
                 "host_d2h", "retain", L, "arrival_ttft_s")
        contrast("cold_recompute-resume_retain_L%d" % L,
                 "recompute", "retain", L, "arrival_ttft_s")
        contrast("restore_h2d-recompute_L%d" % L,
                 "host_h2d_direct", "recompute", L, "arrival_ttft_s")
        contrast("bg_tpot_host_d2h-retain_L%d" % L,
                 "host_d2h", "retain", L, "bg_tpot_mean")
        contrast("bg_tpot_recompute-retain_L%d" % L,
                 "recompute", "retain", L, "bg_tpot_mean")
        contrast("bg_tpot_h2d-retain_L%d" % L,
                 "host_h2d_direct", "retain", L, "bg_tpot_mean")
        contrast("memdip_host_d2h-retain_L%d" % L,
                 "host_d2h", "retain", L, "mem_dip_mb")
        contrast("memdip_h2dreload-retain_L%d" % L,
                 "host_h2d_direct", "retain", L, "mem_dip_mb")
        contrast("emcrate_host_d2h-retain_L%d" % L,
                 "host_d2h", "retain", L, "emc_rate_mean")

    # ---------------- 3. DISCREPANCY: cold-vs-hit arrival ----------------
    # phase1: bgonly base vs cold-arrival loaded (bg4, ~6K)
    p1 = {}
    for name in ["p1-retain-bgonly.json", "p1-retain-r1.json",
                 "p1-retain-r2.json", "p2-host-bgonly.json",
                 "p2-host-r1.json", "p2-host-r2.json",
                 "p4-nvme-loaded.json"]:
        d = load_json(os.path.join(PHASE1, name))
        bgs = d.get("background", [])
        tp = [b.get("tpot_s") for b in bgs if b.get("tpot_s") is not None]
        gm = max((b.get("gap_max_s") for b in bgs
                  if b.get("gap_max_s") is not None), default=None)
        arr = d.get("arrival") or {}
        p1[name] = {
            "bg_tpot_mean": sum(tp) / len(tp) if tp else None,
            "bg_gap_max": gm,
            "arrival_ttft": arr.get("ttft_s"),
            "arrival_cached": arr.get("cached_tokens"),
            "arrival_tpot": arr.get("tpot_s"),
        }
    # sweepB: bg-only baselines vs hit-arrival loaded (bg2/bg8, 6K)
    sweepB = load_json(os.path.join(EXTRAS, "sweepB_concurrency.json"))
    sweepB_rows = []
    for lvl in sweepB["levels"]:
        base = [b["tpot_s"] for b in lvl["baseline"]["bg"]]
        base_m = sum(base) / len(base)
        for i, c in enumerate(lvl["overlap_cells"]):
            loaded = [b["tpot_s"] for b in c["bg"]]
            lm = sum(loaded) / len(loaded)
            sweepB_rows.append({
                "bg_n": lvl["bg_n"], "rep": i,
                "base_tpot": base_m,
                "loaded_tpot": lm,
                "inflation_pct": (lm - base_m) / base_m * 100.0,
                "arrival_ttft": c["arrival"]["ttft_s"],
                "arrival_cached": c["arrival"]["cached_tokens"],
                "gap_max": max(b["gap_max_s"] for b in c["bg"]),
            })
    # matrix @6K bg4: hit-loaded (retain/host_d2h) vs cold-loaded (recompute)
    disc_matrix = {}
    for arm, ph in [("retain", "reload"), ("host_d2h", "reload"),
                    ("recompute", "recompute"),
                    ("host_h2d_direct", "reload")]:
        rows = sel(arm, 6011, ph)
        tp = [r["bg_tpot_mean"] for r in rows]
        disc_matrix[arm] = {
            "bg_tpot_each": tp,
            "bg_tpot_mean": sum(tp) / len(tp),
            "arr_ttft_each": [r["arrival_ttft_s"] for r in rows],
            "arr_cached_each": [r["arrival_cached"] for r in rows],
            "gap_max_each": [r["bg_gap_max"] for r in rows],
            "files": [r["file"] for r in rows],
        }
    discrepancy = {"phase1": p1, "sweepB": sweepB_rows,
                   "matrix_6K_bg4": disc_matrix}

    # ---------------- 4. DUPLICATION NARROWING ----------------
    dup = {"phase1_host_dips": {}, "matrix_plain_spill": {},
           "matrix_baselines": {}, "extras_plain_spill": {},
           "direct_eviction": {}}
    # phase1 host vs retain dips (lead-mean minus load-min, recomputed here)
    for name in ["p1-retain-bgonly.json", "p1-retain-r1.json",
                 "p1-retain-r2.json", "p2-host-bgonly.json",
                 "p2-host-r1.json", "p2-host-r2.json",
                 "p2-host-d2h-trace.json", "p4-nvme-loaded.json"]:
        p = os.path.join(PHASE1, name)
        try:
            d = load_json(p)
        except Exception:
            continue
        if name == "p2-host-d2h-trace.json":
            base = d.get("baseline_avail_kb")
            av = [m["avail_kb"] for m in d.get("mem", [])]
            dup["phase1_host_dips"][name] = {
                "dip_mb": (base - min(av)) / 1024.0 if av else None,
                "baseline_mb": base / 1024.0 if base else None,
                "method": "baseline_avail_kb minus trace min",
            }
            continue
        mem, t0, t1 = d.get("mem", []), d.get("t0"), d.get("t1")
        lead = [m["avail_kb"] for m in mem if m["t"] < t0]
        load = [m["avail_kb"] for m in mem if t0 <= m["t"] <= t1]
        dup["phase1_host_dips"][name] = {
            "dip_mb": ((sum(lead) / len(lead)) - min(load)) / 1024.0,
            "baseline_mb": (sum(lead) / len(lead)) / 1024.0,
        }
    # extras sweepC plain spill deltas (both backends)
    sweepC = load_json(os.path.join(EXTRAS, "sweepC_d2h_dup.json"))
    for b in sweepC["backends"]:
        dup["extras_plain_spill"][b["backend"]] = {
            "delta_spill_mb": b["delta_spill_mb"],
            "delta_reload_mb": b["delta_reload_mb"],
            "spill_ttft_s": b["spill"]["ttft_s"],
            "resume_ttft_s": b["resume"]["ttft_s"],
        }
    # matrix: plain-spill warm dips host_d2h vs retain + meas baselines
    for arm in ["retain", "host_d2h"]:
        for L in LENGTHS:
            rows = sorted([r for r in warm if r["arm"] == arm
                           and r["length"] == L and r["phase"] == "spill"],
                          key=lambda r: r["block"])
            dup["matrix_plain_spill"]["%s_L%d_warm_spill" % (arm, L)] = {
                "dip_each_mb": [r["mem_dip_mb"] for r in rows],
                "excluded_warm": True,
                "files": [r["file"] for r in rows],
            }
    for arm in ARMS[:4]:
        for L in LENGTHS:
            key = "%s_L%d_reload_mem_baseline_mb" % (arm, L) \
                if arm != "recompute" else "%s_L%d_recompute_mem_baseline_mb" % (arm, L)
            src = aggregates.get(key, {})
            dup["matrix_baselines"]["%s_L%d" % (arm, L)] = {
                "mean_mb": src.get("mean"), "lo": src.get("lo"),
                "hi": src.get("hi"), "n": src.get("n"),
            }
    # direct-backend eviction path: matrix h2d spill/fill/reload + v4h + V4h
    for L in LENGTHS:
        for phase in ["spill", "fill0", "reload"]:
            key = "host_h2d_direct_L%d_%s_mem_dip_mb" % (L, phase)
            src = aggregates.get(key, {})
            dup["direct_eviction"]["matrix_%s_L%d_%s" % (
                "host_h2d_direct", L, phase)] = {
                "mean_mb": src.get("mean"), "lo": src.get("lo"),
                "hi": src.get("hi"), "n": src.get("n"),
                "files": src.get("files"),
            }
    try:
        v4h = load_json(os.path.join(IMPORTED, "v4h-load", "SUMMARY.json"))
        dup["direct_eviction"]["v4h_load"] = v4h.get("cycles")
        dup["direct_eviction"]["v4h_control"] = v4h.get(
            "control_retain_style_hier_on")
    except Exception as e:
        dup["direct_eviction"]["v4h_load"] = "load_error %r" % e
    try:
        cp = load_json(os.path.join(IMPORTED, "crash-probe", "SUMMARY.json"))
        v4h_cp = [v for v in cp.get("variants", []) if v.get("id") == "V4h"]
        dup["direct_eviction"]["crash_probe_V4h"] = v4h_cp
    except Exception as e:
        dup["direct_eviction"]["crash_probe_V4h"] = "load_error %r" % e

    # ---------------- 5. CROSSOVER ----------------
    # OLS fits over meas block points (15 pts/arm): ttft vs prompt tokens.
    # prompt tokens: arrival.prompt_tokens not stored in recs; use nominal
    # mapping 2048->2053, 4096->4105, 6011->6011 observed cached+1.
    TOK = {2048: 2053, 4096: 4105, 6011: 6011}
    pts_resume, pts_recomp, pts_h2d = [], [], []
    for L in LENGTHS:
        for r in sel("retain", L, "reload"):
            pts_resume.append((TOK[L], r["arrival_ttft_s"]))
        for r in sel("recompute", L, "recompute"):
            pts_recomp.append((TOK[L], r["arrival_ttft_s"]))
        for r in sel("host_h2d_direct", L, "reload"):
            pts_h2d.append((TOK[L], r["arrival_ttft_s"]))
    fit_resume = linfit(pts_resume)
    fit_recomp = linfit(pts_recomp)
    fit_h2d = linfit(pts_h2d)
    # sweepA independent resume/recompute points (no bg load, hierarchical OFF)
    sweepA = load_json(os.path.join(EXTRAS, "sweepA_resume_crossover.json"))
    sweepA_pts = [(c["length"], c["rep"], c["spill"]["ttft_s"],
                   c["resume"]["ttft_s"], c["recompute"]["ttft_s"])
                  for c in sweepA["cells"]]

    def crossing(f1, f2):
        if f1["b"] is None or f2["b"] is None or f1["b"] == f2["b"]:
            return None
        return (f2["a"] - f1["a"]) / (f1["b"] - f2["b"])

    crossover = {
        "fit_resume_retain": fit_resume,
        "fit_recompute": fit_recomp,
        "fit_host_restore_direct": fit_h2d,
        "slope_units": "s_per_token",
        "cross_resume_vs_recompute_tok": crossing(fit_resume, fit_recomp),
        "cross_hostrestore_vs_recompute_tok": crossing(fit_h2d, fit_recomp),
        "sweepA_points": [{"length": l, "rep": rp, "spill_ttft": s,
                           "resume_ttft": r, "recompute_ttft": c}
                          for l, rp, s, r, c in sweepA_pts],
        "nvme": {
            "matrix_meas": [0, 15],
            "matrix_warm_reload_fail": 15,
            "matrix_crash_blocks": 15,
            "note": "no meas files exist; warm reload arrival comp None + "
                    "health fail per block (server logs: illegal/Traceback/"
                    "CUDA/AcceleratorError/Scheduler exception)",
            "phase1_spill_bytes_after_6K": 260542464,
            "phase1_expected_bytes": 257832960,
            "phase1_restore_ttft_s": 1.332,
            "phase1_restore_cached": 0,
        },
    }

    # ---------------- 6. UNDERPOWERED ----------------
    under = []
    for name, c in contrasts.items():
        pc, mt = c["paired_diff"], c["margin_10pct"]
        if mt["verdict"] in ("underpowered_wide_ci",
                             "indeterminate_overlaps_band",
                             "not_testable"):
            under.append({"contrast": name, "verdict": mt["verdict"],
                          "ci_width": pc.get("width"),
                          "band": mt.get("band"), "n": pc.get("n_blocks")})
    for label, a in aggregates.items():
        if a.get("n", 5) < 5 or a.get("width") is None:
            under.append({"aggregate": label, "issue": "n<5 or missing",
                          "n": a.get("n")})

    summary = {
        "frozen_contract": {
            "reps": "blocks (n=5 per arm x length)",
            "ci": "paired-block 95pct percentile bootstrap, B=10000, seed=%d"
                  % SEED0,
            "margin": "10% of reference mean",
            "warmups": "excluded from all stats",
        },
        "coverage": coverage_pct,
        "claim_scope": {
            "may_claim": "meas phases of retain/host_d2h/host_h2d_direct/"
                         "recompute (60/60 arm x length x block meas cells "
                         "PASS incl. h2d spill+fill+reload sub-phases)",
            "must_exclude": "nvme_spill meas (0/15, reload crash every "
                            "block); warmups as reps; kernel/page_first "
                            "H2D (excluded by design; crash-probe 3/3 fatal)",
        },
        "aggregates": aggregates,
        "contrasts": contrasts,
        "discrepancy": discrepancy,
        "duplication": dup,
        "crossover": crossover,
        "cold_reference_excluded": cold_ref,
        "nvme_warm_reference_excluded": nvme_ref,
        "underpowered": under,
    }
    with open(os.path.join(RESULTS, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # ---------------- FIGURE CSVs ----------------
    # mem-bars: meas dip + CI per arm x length (+ h2d spill/fill rows)
    with open(os.path.join(FIGURES, "mem-bars.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "length", "phase", "mean_dip_mb", "lo_mb",
                    "hi_mb", "n", "mean_baseline_mb", "files"])
        for arm in ["retain", "host_d2h", "recompute", "host_h2d_direct"]:
            for L in LENGTHS:
                ph = {"retain": "reload", "host_d2h": "reload",
                      "recompute": "recompute",
                      "host_h2d_direct": "reload"}[arm]
                a = aggregates["%s_L%d_%s_mem_dip_mb" % (arm, L, ph)]
                b = aggregates["%s_L%d_%s_mem_baseline_mb" % (arm, L, ph)]
                w.writerow([arm, L, ph, _r(a.get("mean")), _r(a.get("lo")),
                            _r(a.get("hi")), a.get("n"),
                            _r(b.get("mean")), len(a.get("files", []))])
        for L in LENGTHS:
            for ph in ["spill", "fill0"]:
                a = aggregates["host_h2d_direct_L%d_%s_mem_dip_mb" % (L, ph)]
                b = aggregates[
                    "host_h2d_direct_L%d_%s_mem_baseline_mb" % (L, ph)]
                w.writerow(["host_h2d_direct", L, ph, _r(a.get("mean")),
                            _r(a.get("lo")), _r(a.get("hi")), a.get("n"),
                            _r(b.get("mean")), len(a.get("files", []))])
    # emc-timeline-sample: raw mc_all series, 2 cells (retain + h2d, 6K b0)
    with open(os.path.join(FIGURES, "emc-timeline-sample.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "file", "t_rel_s", "mc_all", "emc_hz"])
        for path in [os.path.join(
                MATRIX, "retain_L6011_b0_meas_reload.json"),
                os.path.join(
                MATRIX,
                "host_h2d_direct_L6011_b0_meas_reload.json")]:
            d = load_json(path)
            t0 = d["t0"]
            arm = (d.get("_matrix") or {}).get("arm")
            for e in d.get("emc", []):
                w.writerow([arm, os.path.basename(path),
                            round(e["t"] - t0, 3), e["mc_all"],
                            e["emc_hz"]])
    # ttft-crossover: per-length means (matrix) + sweepA reps (flagged)
    with open(os.path.join(FIGURES, "ttft-crossover.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "arm", "length", "tokens", "mean_ttft_s",
                    "lo_s", "hi_s", "n", "cached_note"])
        for arm, ph, note in [
                ("retain", "reload", "device-hit"),
                ("host_d2h", "reload", "device-hit"),
                ("recompute", "recompute", "cold"),
                ("host_h2d_direct", "reload", "host-hit-direct")]:
            for L in LENGTHS:
                a = aggregates["%s_L%d_%s_arrival_ttft_s" % (arm, L, ph)]
                w.writerow(["matrix", arm, L, TOK[L], _r(a.get("mean")),
                            _r(a.get("lo")), _r(a.get("hi")), a.get("n"),
                            note])
        for l, rp, s, r, c in sweepA_pts:
            w.writerow(["sweepA_rep%d" % rp, "retain-spill-cold", l, l, s,
                        "", "", 1, "cold-noload"])
            w.writerow(["sweepA_rep%d" % rp, "retain-resume", l, l, r,
                        "", "", 1, "device-hit-noload"])
            w.writerow(["sweepA_rep%d" % rp, "recompute", l, l, c,
                        "", "", 1, "cold-noload"])
    # tpot-concurrency: sweepB base/loaded + matrix bg4 loaded@6K + phase1
    with open(os.path.join(FIGURES, "tpot-concurrency.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "bg_n", "arrival", "arm", "mean_tpot_s",
                    "extra"])
        for r in sweepB_rows:
            w.writerow(["sweepB-base", r["bg_n"], "none", "retain-noload",
                        round(r["base_tpot"], 6), "rep%d" % r["rep"]])
            w.writerow(["sweepB-loaded", r["bg_n"], "hit-6K", "retain",
                        round(r["loaded_tpot"], 6),
                        "rep%d infl=%.1f%% gapmax=%.3f" % (
                            r["rep"], r["inflation_pct"], r["gap_max"])])
        for arm, ph in [("retain", "reload"), ("host_d2h", "reload"),
                        ("recompute", "recompute"),
                        ("host_h2d_direct", "reload")]:
            a = aggregates["%s_L6011_%s_bg_tpot_mean" % (arm, ph)]
            arr = "hit-6K" if arm != "recompute" else "cold-6K"
            if arm == "host_h2d_direct":
                arr = "host-restore-6K"
            w.writerow(["matrix", 4, arr, arm, _r(a.get("mean")),
                        "n=%s" % a.get("n")])
        w.writerow(["phase1-base", 4, "none", "retain",
                    round(p1["p1-retain-bgonly.json"]["bg_tpot_mean"], 6),
                    "bgonly"])
        w.writerow(["phase1-loaded", 4, "cold-6K", "retain",
                    round(p1["p1-retain-r1.json"]["bg_tpot_mean"], 6),
                    "gapmax=%.3f" % p1["p1-retain-r1.json"]["bg_gap_max"]])
        w.writerow(["phase1-base", 4, "none", "host",
                    round(p1["p2-host-bgonly.json"]["bg_tpot_mean"], 6),
                    "bgonly"])
        w.writerow(["phase1-loaded", 4, "cold-6K", "host",
                    round(p1["p2-host-r1.json"]["bg_tpot_mean"], 6),
                    "gapmax=%.3f" % p1["p2-host-r1.json"]["bg_gap_max"]])
    print("wrote summary.json + coverage.json + 4 figure CSVs")
    print("aggregates=%d contrasts=%d underpowered=%d failed_cells=%d"
          % (len(aggregates), len(contrasts), len(under),
             len(audit["failed"])))


def _r(v):
    return round(v, 6) if isinstance(v, float) else v


if __name__ == "__main__":
    main()
