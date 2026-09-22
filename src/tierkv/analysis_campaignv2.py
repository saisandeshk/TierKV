"""TierKV campaign-v2 analysis driver (Strand A nvme-direct + Strand B file-overflow).

Reads: runs/imported/campaign-v2/{nvme-direct,file-overflow}/cells (+ summaries).
Writes (results/ only, additive -- never touches results/summary.json):
  results/campaign-v2-summary.json
  results/figures/campaign-v2-ttft.csv

Frozen-contract conventions reused from analysis_main.py (blocks are reps,
warmups excluded, paired-block 95% percentile bootstrap B=10000 seed=20260921,
10% margin, None values dropped per block with warn, never pooled):
  - Strand B quiet vs loaded: PAIRED per-block contrasts (same blocks 0-4)
    on arrival TTFT + cached storage tokens (and cached total).
  - Strand A meas device-hit TTFT vs Strand B file TTFT: DESCRIPTIVE ONLY
    (ratio of means, no pooling across strands, no cross-strand CI).
  - Gating: arrival_ok <=> arrival http 200 and completion_tokens == 64;
    loaded bg streams (where present) http 200 and completion_tokens == 128;
    quiet bg is null by design (no load). Warm spill files excluded.
  - Failures indexed, never pooled.
"""

import csv
import json
import os
import zlib

from tierkv.analysis_stats import boot_ci, paired_contrast

BASE = "/home/saisandeshk/Study/ISP/TierKV"
C2 = os.path.join(BASE, "runs/imported/campaign-v2/nvme-direct/cells")
FO = os.path.join(BASE, "runs/imported/campaign-v2/file-overflow/cells")
RESULTS = os.path.join(BASE, "results")
FIGURES = os.path.join(RESULTS, "figures")
SEED0 = 20260921
BLOCKS = [0, 1, 2, 3, 4]
LENGTHS = [2048, 4096, 6011]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def arrival_gate(arr, tag):
    """arrival_ok <=> http 200 and completion_tokens == 64."""
    ok = (arr.get("http") == 200
          and arr.get("completion_tokens") == 64)
    if not ok:
        print("warn gate FAIL %s: http=%r comp=%r"
              % (tag, arr.get("http"), arr.get("completion_tokens")))
    return ok


def bg_gate(bgs, tag):
    """Loaded bg streams: http 200 + completion_tokens == 128.

    Quiet bg is [null x4] by design (no load) -> returns (True, 'no-load').
    n_chunks shortfalls are indexed as notes, never silently pooled.
    """
    present = [g for g in bgs if g is not None]
    if not present:
        return True, "no-load-by-design"
    notes = []
    for i, g in enumerate(present):
        if g.get("http") != 200 or g.get("completion_tokens") != 128:
            print("warn bg gate FAIL %s bg%d: http=%r comp=%r"
                  % (tag, i, g.get("http"), g.get("completion_tokens")))
            return False, "bg-gate-fail-bg%d" % i
        if g.get("n_chunks") != 128:
            notes.append("bg%d n_chunks=%r (comp still 128)"
                         % (i, g.get("n_chunks")))
    return True, ";".join(notes) if notes else "ok"


def read_strand_b():
    """Per-block quiet/loaded reload records. Warmups: n/a (no warm files
    in Strand B; spill/flushmid are setup phases, not reps)."""
    recs = []
    failures = []
    for b in BLOCKS:
        for phase in ["quiet_reload", "loaded_reload"]:
            path = os.path.join(FO, "fo_b%d_%s.json" % (b, phase))
            d = load_json(path)
            arr = d.get("arrival") or {}
            m = d.get("_matrix") or {}
            a_ok = arrival_gate(arr, "fo_b%d_%s" % (b, phase))
            b_ok, b_note = bg_gate(d.get("background", []),
                                   "fo_b%d_%s" % (b, phase))
            ok = bool(m.get("pass")) and a_ok and b_ok
            if not ok:
                failures.append({"file": path, "block": b, "phase": phase,
                                 "matrix_pass": m.get("pass"),
                                 "arrival_ok": a_ok, "bg_ok": b_ok,
                                 "bg_note": b_note})
            recs.append({
                "block": b, "phase": phase, "file": path,
                "matrix_pass": bool(m.get("pass")),
                "arrival_ok": a_ok, "bg_ok": b_ok, "bg_note": b_note,
                "gated": ok,
                "ttft_s": arr.get("ttft_s"),
                "cached_tokens": arr.get("cached_tokens"),
                "cached_device": arr.get("cached_device"),
                "cached_host": arr.get("cached_host"),
                "cached_storage": arr.get("cached_storage"),
                "prompt_tokens": arr.get("prompt_tokens"),
                "completion_tokens": arr.get("completion_tokens"),
            })
    return recs, failures


def read_strand_a():
    """Meas reload records only; 15 warm_spill files excluded (listed)."""
    recs = []
    failures = []
    warm_excluded = []
    for L in LENGTHS:
        for b in BLOCKS:
            wpath = os.path.join(
                C2, "nvme_direct_L%d_b%d_warm_spill.json" % (L, b))
            if os.path.exists(wpath):
                warm_excluded.append(wpath)
            path = os.path.join(
                C2, "nvme_direct_L%d_b%d_meas_reload.json" % (L, b))
            d = load_json(path)
            arr = d.get("arrival") or {}
            m = d.get("_matrix") or {}
            a_ok = arrival_gate(
                arr, "nvme_direct_L%d_b%d_meas" % (L, b))
            ok = bool(m.get("pass")) and a_ok
            if not ok:
                failures.append({"file": path, "length": L, "block": b,
                                 "matrix_pass": m.get("pass"),
                                 "arrival_ok": a_ok})
            recs.append({
                "length": L, "block": b, "file": path,
                "matrix_pass": bool(m.get("pass")),
                "arrival_ok": a_ok, "gated": ok,
                "ttft_s": arr.get("ttft_s"),
                "cached_tokens": arr.get("cached_tokens"),
                "cached_device": arr.get("cached_device"),
                "cached_host": arr.get("cached_host"),
                "cached_storage": arr.get("cached_storage"),
                "prompt_tokens": arr.get("prompt_tokens"),
                "completion_tokens": arr.get("completion_tokens"),
            })
    return recs, failures, warm_excluded


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
    fo_recs, fo_fail = read_strand_b()
    nv_recs, nv_fail, warm_excl = read_strand_a()
    fo = [r for r in fo_recs if r["gated"]]
    nv = [r for r in nv_recs if r["gated"]]
    if len(fo) != len(fo_recs):
        print("warn strand B gated %d/%d" % (len(fo), len(fo_recs)))
    if len(nv) != len(nv_recs):
        print("warn strand A gated %d/%d" % (len(nv), len(nv_recs)))

    # Dedicated seed stream (SEED0+200000) -- never collides with the
    # matrix-v1 streams (SEED0+counter, SEED0+100000+jseed) or the
    # per-contrast crc32 seeds below.
    seed = SEED0 + 200000

    aggregates = {}
    for phase in ["quiet_reload", "loaded_reload"]:
        rows = [r for r in fo if r["phase"] == phase]
        for metric in ["ttft_s", "cached_storage", "cached_tokens",
                       "cached_host", "cached_device"]:
            seed += 1
            aggregates["file_overflow_%s_%s" % (phase, metric)] = agg(
                rows, metric, "file_overflow %s %s" % (phase, metric),
                seed)
    for L in LENGTHS:
        rows = [r for r in nv if r["length"] == L]
        for metric in ["ttft_s", "cached_tokens"]:
            seed += 1
            aggregates["nvme_direct_L%d_meas_%s" % (L, metric)] = agg(
                rows, metric, "nvme_direct L%d meas %s" % (L, metric),
                seed)

    # ---------------- paired Strand B contrasts (quiet - loaded) --------
    contrasts = {}

    def contrast(name, metric):
        q = by_block([r for r in fo if r["phase"] == "quiet_reload"],
                     metric)
        l = by_block([r for r in fo if r["phase"] == "loaded_reload"],
                     metric)
        shared = sorted(set(q) & set(l))
        if len(shared) != min(len(q), len(l)):
            print("warn contrast %s: paired %d of q=%d l=%d "
                  "(None dropped, never pooled)"
                  % (name, len(shared), len(q), len(l)))
        pc = paired_contrast(
            q, l, seed=SEED0 + (zlib.crc32(name.encode()) % 10000))
        contrasts[name] = {
            "x": "file_overflow quiet_reload",
            "y": "file_overflow loaded_reload",
            "metric": metric, "paired_diff_quiet_minus_loaded": pc,
            "x_files": sorted(r["file"] for r in fo
                              if r["phase"] == "quiet_reload"),
            "y_files": sorted(r["file"] for r in fo
                              if r["phase"] == "loaded_reload"),
        }

    for metric in ["ttft_s", "cached_storage", "cached_tokens"]:
        contrast("quiet_vs_loaded_%s" % metric, metric)

    # ---------------- descriptive cross-strand (NEVER pooled) -----------
    # Ratio of means only; no cross-strand CI, no shared bootstrap.
    descriptive = {}
    q_mean = aggregates["file_overflow_quiet_reload_ttft_s"]["mean"]
    l_mean = aggregates["file_overflow_loaded_reload_ttft_s"]["mean"]
    for L in LENGTHS:
        a_mean = aggregates["nvme_direct_L%d_meas_ttft_s" % L]["mean"]
        descriptive["nvme_vs_file_L%d" % L] = {
            "nvme_direct_meas_ttft_mean_s": a_mean,
            "file_quiet_ttft_mean_s": q_mean,
            "file_loaded_ttft_mean_s": l_mean,
            "ratio_quiet_over_nvme": (q_mean / a_mean
                                      if a_mean else None),
            "ratio_loaded_over_nvme": (l_mean / a_mean
                                       if a_mean else None),
            "method": "ratio of per-strand block means; strands never "
                      "pooled, no cross-strand CI (different prefixes, "
                      "different lengths/arms).",
        }

    summary = {
        "rev": 1,
        "frozen_contract": {
            "reps": "blocks (n=5 per strand arm x phase)",
            "ci": "paired-block 95pct percentile bootstrap, B=10000, "
                  "seed=%d" % SEED0,
            "margin": "10% of reference mean (not tested cross-strand)",
            "warmups": "excluded from all stats "
                       "(15 Strand A warm_spill files listed, never pooled)",
            "gating": "aggregates+contrasts require matrix_pass and "
                      "arrival_ok (arrival http 200, completion_tokens==64); "
                      "loaded bg streams http 200 + completion_tokens==128; "
                      "quiet bg null by design; None metric values dropped "
                      "per block with warn, never pooled",
            "strand_rule": "Strand A vs Strand B is descriptive only "
                           "(different arms/prefixes); no pooled statistic "
                           "crosses strands",
        },
        "import": {
            "nvme_direct_files": 53,
            "nvme_direct_seal": "PASS (Orin + workstation sha256sum -c)",
            "file_overflow_files": 49,
            "file_overflow_seal": "PASS (Orin + workstation sha256sum -c)",
            "file_overflow_storage_shards": "via storage_manifest.json: "
                "5 blocks x 20012 files x 860676096 bytes/blk (~863MB du); "
                "shards not in seal, not copied (copy-out of sealed files "
                "only, no raw mutation)",
            "nvme_direct_storage": "empty dirs (0 files): device-hit path "
                                   "leaves nothing on disk, as expected",
        },
        "gating": {
            "strand_b_cells": len(fo_recs),
            "strand_b_gated": len(fo),
            "strand_a_meas_cells": len(nv_recs),
            "strand_a_gated": len(nv),
            "strand_a_warm_excluded": len(warm_excl),
            "bg_notes": sorted(set(r["bg_note"] for r in fo_recs
                                   if r["bg_note"] not in ("ok",))),
        },
        "aggregates": aggregates,
        "contrasts": contrasts,
        "descriptive_cross_strand": descriptive,
        "failures_indexed": fo_fail + nv_fail,
        "warm_excluded_files": warm_excl,
    }
    with open(os.path.join(RESULTS, "campaign-v2-summary.json"),
              "w") as f:
        json.dump(summary, f, indent=1)

    with open(os.path.join(FIGURES, "campaign-v2-ttft.csv"), "w",
              newline="") as f:
        f.write("# campaign-v2 TTFT + cached split per block. "
                "Strand B: fo prefixes C/D/E (~6K tok); "
                "Strand A: synthetic prefixes (2048/4096/6011). "
                "Rows are block reps; strands never pooled.\n")
        w = csv.writer(f)
        w.writerow(["strand", "arm_phase", "length", "block",
                    "ttft_s", "cached_total", "cached_device",
                    "cached_host", "cached_storage", "prompt_tokens",
                    "completion_tokens", "gated", "file"])
        for r in sorted(fo_recs, key=lambda r: (r["phase"], r["block"])):
            w.writerow(["B", "file_overflow_%s" % r["phase"], "",
                        r["block"], _r(r["ttft_s"]), r["cached_tokens"],
                        r["cached_device"], r["cached_host"],
                        r["cached_storage"], r["prompt_tokens"],
                        r["completion_tokens"], r["gated"], r["file"]])
        for r in sorted(nv_recs, key=lambda r: (r["length"], r["block"])):
            w.writerow(["A", "nvme_direct_meas_reload", r["length"],
                        r["block"], _r(r["ttft_s"]), r["cached_tokens"],
                        r["cached_device"], r["cached_host"],
                        r["cached_storage"], r["prompt_tokens"],
                        r["completion_tokens"], r["gated"], r["file"]])
    print("wrote campaign-v2-summary.json + campaign-v2-ttft.csv")
    print("aggregates=%d contrasts=%d failures=%d warm_excluded=%d"
          % (len(aggregates), len(contrasts),
             len(fo_fail + nv_fail), len(warm_excl)))


def _r(v):
    return round(v, 6) if isinstance(v, float) else v


if __name__ == "__main__":
    main()
