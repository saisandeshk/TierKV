"""Smoke-test placeholder: tracker loads and required keys exist."""
import json
from pathlib import Path


def test_tracker_loads():
    p = Path(__file__).resolve().parents[1] / "config" / "tracker.json"
    data = json.loads(p.read_text())
    assert "model" in data and "prefix_lengths" in data
    assert data["prefix_lengths"]["status"] in {"PROPOSED", "FROZEN", "OPEN"}
