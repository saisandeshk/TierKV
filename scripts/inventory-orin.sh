#!/bin/bash
# Read-only Orin inventory. No workload, no config change, no sudo side effects.
set -euo pipefail
OUT="${1:-inventory.orin.json}"
python3 - <<'PY' > "$OUT"
import json, platform
print(json.dumps({"host": platform.node(), "note": "fill via T0.1 probe on Orin; do not hand-edit as verified"}, indent=2))
PY
echo "wrote $OUT (placeholder; T0.1 fills on device)"
