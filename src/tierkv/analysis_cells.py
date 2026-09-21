"""TierKV matrix-v1 analysis: cell loading + per-cell metric extraction.

Read-only wrt raw runs. All metrics derive from imported cell JSON files.
Warmup-cycle files are loadable but ALWAYS flagged warm=True so callers can
exclude them from stats (frozen contract: blocks are the reps, never warmups).

Metric definitions (frozen for this analysis):
- resume_TTFT_s: arrival.ttft_s in a meas RELOAD cell (hit resume).
- cold_TTFT_s: arrival.ttft_s in a meas RECOMPUTE cell (flush then full
  prefill; cold by construction). Warm-spill cold values are exposed only as
  excluded reference (warm=True), never pooled.
- bg_TPOT_loaded_s: mean of background[].tpot_s within the cell (4x bg128).
- bg_gap_max_s / bg_gap_p95_s: max / mean-of-p95 across background streams.
- mem_dip_MB: lead-mean MemAvailable minus load-min MemAvailable, MB, where
  lead = mem rows with t<t0, load = rows with t0<=t<=t1.
- mem_baseline_MB: lead-mean MemAvailable (server-pool baseline for the cell).
- emc_rate_mean / emc_rate_max: mean/max of d(mc_all)/dt over load window.
- cached_dev / cached_host: arrival cached split.
"""
import json
import glob
import os
import re

IMPORTED = "/home/saisandeshk/Study/ISP/TierKV/runs/imported"
MATRIX = os.path.join(IMPORTED, "matrix-v1")
EXTRAS = os.path.join(IMPORTED, "extras")
PHASE1 = os.path.join(IMPORTED, "phase1")

CELL_RE = re.compile(
    r"^(retain|host_d2h|host_h2d_direct|recompute|nvme_spill)"
    r"_L(\d+)_b(\d+)_(meas|warm)_(.+)\.json$"
)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def parse_cell_name(basename):
    m = CELL_RE.match(basename)
    if not m:
        return None
    return {
        "arm": m.group(1),
        "length": int(m.group(2)),
        "block": int(m.group(3)),
        "cycle": m.group(4),  # meas | warm
        "phase": m.group(5),  # reload | spill | recompute | fillN
    }


def list_matrix_cells():
    """All matrix-v1 per-cell JSON files with parsed identity (excl. summary)."""
    out = []
    for path in sorted(glob.glob(os.path.join(MATRIX, "*.json"))):
        base = os.path.basename(path)
        if base == "matrix-summary.json":
            continue
        meta = parse_cell_name(base)
        if meta is None:
            continue
        meta["file"] = path
        meta["warm"] = (meta["cycle"] == "warm")
        out.append(meta)
    return out


def bg_stats(cell):
    tpot = [b.get("tpot_s") for b in cell.get("background", [])]
    tpot = [v for v in tpot if isinstance(v, (int, float))]
    p95 = [b.get("gap_p95_s") for b in cell.get("background", [])]
    p95 = [v for v in p95 if isinstance(v, (int, float))]
    gmax = [b.get("gap_max_s") for b in cell.get("background", [])]
    gmax = [v for v in gmax if isinstance(v, (int, float))]
    n = len(cell.get("background", []))
    mean = sum(tpot) / len(tpot) if tpot else None
    return {
        "bg_n": n,
        "bg_tpot_each": tpot,
        "bg_tpot_mean": mean,
        "bg_gap_p95_each": p95,
        "bg_gap_p95_mean": (sum(p95) / len(p95)) if p95 else None,
        "bg_gap_max_each": gmax,
        "bg_gap_max": max(gmax) if gmax else None,
    }


def mem_stats(cell):
    mem = cell.get("mem", []) or []
    t0, t1 = cell.get("t0"), cell.get("t1")
    lead = [m["avail_kb"] for m in mem
            if m.get("avail_kb") is not None and t0 is not None and m["t"] < t0]
    load = [m["avail_kb"] for m in mem
            if m.get("avail_kb") is not None and t0 is not None
            and t1 is not None and t0 <= m["t"] <= t1]
    if not lead or not load:
        return {"mem_n": len(mem), "mem_baseline_mb": None,
                "mem_dip_mb": None, "mem_load_min_mb": None,
                "mem_load_mean_mb": None}
    baseline = sum(lead) / len(lead)
    lmin = min(load)
    lmean = sum(load) / len(load)
    return {
        "mem_n": len(mem),
        "mem_baseline_mb": baseline / 1024.0,
        "mem_dip_mb": (baseline - lmin) / 1024.0,
        "mem_load_min_mb": lmin / 1024.0,
        "mem_load_mean_mb": lmean / 1024.0,
    }


def emc_stats(cell):
    emc = cell.get("emc", []) or []
    t0, t1 = cell.get("t0"), cell.get("t1")
    hz = sorted({e["emc_hz"] for e in emc if e.get("emc_hz") is not None})
    load = [e for e in emc if e.get("mc_all") is not None
            and t0 is not None and t1 is not None and t0 <= e["t"] <= t1]
    rates = []
    for a, b in zip(load, load[1:]):
        dt = b["t"] - a["t"]
        if dt > 0:
            rates.append((b["mc_all"] - a["mc_all"]) / dt)
    nulls = sum(1 for e in emc if e.get("mc_all") is None)
    return {
        "emc_n": len(emc),
        "emc_nulls": nulls,
        "emc_hz_values": hz,
        "emc_rate_mean": (sum(rates) / len(rates)) if rates else None,
        "emc_rate_max": max(rates) if rates else None,
        "emc_rate_min": min(rates) if rates else None,
        "emc_rate_n": len(rates),
    }


def arrival_stats(cell):
    arr = cell.get("arrival")
    if not isinstance(arr, dict) or arr.get("http") != 200:
        return {"arrival_http": (arr or {}).get("http"),
                "arrival_ttft_s": None,
                "arrival_tpot_s": arr.get("tpot_s") if isinstance(arr, dict) else None,
                "arrival_cached": None, "cached_device": None,
                "cached_host": None, "completion_tokens": None,
                "arrival_ok": False}
    return {
        "arrival_http": 200,
        "arrival_ttft_s": arr.get("ttft_s"),
        "arrival_tpot_s": arr.get("tpot_s"),
        "arrival_cached": arr.get("cached_tokens"),
        "cached_device": arr.get("cached_device"),
        "cached_host": arr.get("cached_host"),
        "completion_tokens": arr.get("completion_tokens"),
        "arrival_ok": arr.get("completion_tokens") == 64,
    }


def cell_metrics(path):
    """Full per-cell metric record (identity + all metrics + pass flag)."""
    cell = load_json(path)
    meta = parse_cell_name(os.path.basename(path)) or {}
    mx = cell.get("_matrix", {}) or {}
    rec = {
        "file": path,
        "arm": meta.get("arm"),
        "length": meta.get("length"),
        "block": meta.get("block"),
        "cycle": meta.get("cycle"),
        "phase": meta.get("phase"),
        "warm": meta.get("cycle") == "warm",
        "matrix_pass": mx.get("pass"),
        "matrix_note": mx.get("note"),
        "overlap": cell.get("overlap"),
    }
    rec.update(bg_stats(cell))
    rec.update(mem_stats(cell))
    rec.update(emc_stats(cell))
    rec.update(arrival_stats(cell))
    return rec


def load_all_matrix_metrics():
    return [cell_metrics(c["file"]) for c in list_matrix_cells()]


def coverage_audit():
    """Coverage from matrix-summary.json + on-disk file check.

    Returns dict with per-cell status, failed/crashed list with log pointers,
    and warmup accounting kept separate.
    """
    summary = load_json(os.path.join(MATRIX, "matrix-summary.json"))
    cells = summary.get("cells", {})
    rows = []
    failed = []  # every non-passing result phase + CRASH pseudo-phases
    for key in sorted(cells):
        c = cells[key]
        arm, length, block = c["arm"], c["length"], c["block"]
        for r in c.get("results", []):
            f = r.get("file")
            on_disk = None
            mx_pass = None
            if f:
                # files were imported under new root; map basename
                local = os.path.join(MATRIX, os.path.basename(f))
                on_disk = os.path.exists(local)
                if on_disk and local.endswith(".json"):
                    try:
                        d = load_json(local)
                        mx_pass = (d.get("_matrix") or {}).get("pass")
                    except Exception:
                        mx_pass = "load_error"
            phase = r.get("phase", "")
            is_warm = phase.startswith("warm")
            ok = bool(r.get("pass"))
            rows.append({"cell": key, "arm": arm, "length": length,
                         "block": block, "phase": phase, "pass": ok,
                         "note": r.get("note"), "file": f, "on_disk": on_disk,
                         "warm": is_warm})
            if not ok:
                log = "%s/server-%s-blk%d.log" % (MATRIX, arm, block)
                local_f = (os.path.join(MATRIX, os.path.basename(f))
                           if f else None)
                failed.append({"cell": key, "arm": arm, "length": length,
                               "block": block, "phase": phase,
                               "note": r.get("note"), "file": f,
                               "local_file": local_f,
                               "server_log": log})
    return {"cells": cells, "rows": rows, "failed": failed,
            "infra_incidents": summary.get("infra_incidents", []),
            "block_order": summary.get("block_order", {})}
