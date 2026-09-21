# EXPERIMENT (what + why, not frozen numbers)

Question: does host-pinned KV offload free physical DRAM on Orin-32GB shared LPDDR5, and where do NVMe-spill vs recompute win? A5000 discrete is the control (same op should free HBM there).

Arms (locked): RETAIN / HOST_D2H_H2D / RECOMPUTE_FLUSH_PREFILL / NVME_DIRECT_SPILL.
Independent requests only. No local-tool contenders. Sleep-only idle gap if used.
Workload: synthetic fixed-length prefixes (unique leading tokens, zero reuse) + ShareGPT slice as sensitivity. Greedy fixed output budget, verify actual token counts.
Numerics live in `config/tracker.json`. Do not edit workload here; propose change via tracker.
