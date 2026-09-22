"""Tracker + analysis-contract smoke tests (workstation only, no devices).

- config/tracker.json loads, required keys exist, core model choices FROZEN.
- Locked arms (AGENTS.md: retain / host_d2h+host_h2d_direct / recompute /
  nvme_spill-indexed) exist in the analysis driver and are never renamed
  without human approval.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCKED_ARMS = {"retain", "host_d2h", "host_h2d_direct", "recompute",
               "nvme_spill"}


def test_tracker_loads():
    p = REPO / "config" / "tracker.json"
    data = json.loads(p.read_text())
    assert "model_mechanism" in data and "prefix_lengths" in data
    assert data["prefix_lengths"]["status"] in {"PROPOSED", "FROZEN", "OPEN"}


def test_tracker_core_frozen():
    data = json.loads((REPO / "config" / "tracker.json").read_text())
    for key in ("model_mechanism", "model_realism"):
        assert data[key]["status"] == "FROZEN", key
    assert data["emc_sampler"]["status"] == "FROZEN"


def test_locked_arms_exist():
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from tierkv import analysis_main
    assert LOCKED_ARMS <= set(analysis_main.ARMS), analysis_main.ARMS
    # matched-work contract: same 3 nominal lengths, 5 blocks, SEED0 pinned
    assert analysis_main.LENGTHS == [2048, 4096, 6011]
    assert analysis_main.BLOCKS == [0, 1, 2, 3, 4]
    assert analysis_main.SEED0 == 20260921
