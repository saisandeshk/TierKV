# TierKV — Unified-Memory Tiered KV Offload on Orin-32GB vs A5000
# Core LOCKED: RETAIN vs HOST_D2H_H2D vs RECOMPUTE vs NVME_DIRECT, matched work, MemAvailable+EMC+TTFT/TPOT/joules.
# Choices live in config/tracker.json. Full gated plan: ~/.opencode/plan/TierKV-SRS-PLAN.md (not in git).

## Layout
```text
config/        Tracker + campaign templates (frozen copies go here with hashes)
docs/          EXPERIMENT (what/why), PROTOCOL (frozen measurement), DEVELOPMENT (device sync)
src/tierkv/    Client + analysis, Python 3.10+, no GPU import
scripts/       Device entry points (smoke, inventory, seal, plot)
tests/         Local correctness tests, no device needed
data/traces/   Versioned input specs (no model weights, no raw runs)
runs/          IGNORED device-local raw evidence
results/       Curated reports/figures selected for git
paper/         IEEE 2-page manuscript
```

## Quick start (local, no GPU)
```bash
python3 -m pytest -q
python3 -c "import tierkv; print(tierkv.__version__)"
```

## Device sync (Orin/A5000)
```bash
# on workstation after scaffold push:
# orin:  cd /media/ssd/saisandesh/TierKV && git pull origin main
# a5000: cd <writable-path>/TierKV && git pull origin main
# never commit runs/, *.log, credentials; record code+config hashes per campaign
```
