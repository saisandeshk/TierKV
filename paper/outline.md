# TierKV paper outline (SRS 2-page budget) — DRAFT, numbers from sealed runs only

Title (narrow, no universal dominance): Host-Pinned KV Offload on Unified
vs Discrete GPUs: When Device-Hit Caching Beats Host Restore on Orin-32GB
— DRAFT TITLE, open to revision.

AI disclosure: [PLACEHOLDER — describe AI assistance per venue policy:
coding agent drafted analysis scripts and this outline; all numbers are
from sealed runs cited below; authors verified every claim.]

## Budget (2 pages SRS)

### 1. Intro — 0.5 page
- Problem: long-prefix serving pays full prefill per request unless KV is
  retained; host DRAM offload (HiCache-style) promises HBM relief.
- Question: does host-pinned offload free physical DRAM on Orin-32GB
  unified LPDDR5, and where do host/file restore vs recompute win?
- Narrow claim (preview): on discrete A5000, kernel-backend host restore is
  0.35x/0.17x retain TTFT (L2048/L6011) and frees HBM; on unified Orin,
  host spill frees no MemAvailable and restore equals a device hit, while
  the direct backend required for H2D costs +13.4s over recompute at L6011
  and file-tier restore costs ~17s vs ~0.13s device (~130x, descriptive).
- Version confound stated up front: Orin sglang 0.5.16 vs A5000 0.5.20.

### 2. Design — 0.5 page + 1 tier figure
- 4 locked arms: RETAIN / HOST_D2H(+H2D kernel) / RECOMPUTE(flush+prefill) /
  NVME_DIRECT_SPILL; matched work, greedy fixed budgets (FG 64-out,
  BG 128-out), paired randomized blocks n=5, warmups excluded.
- Truth sources: MemAvailable bracketing (never cudaMemGetInfo as physical),
  EMC as utilization-only proxy + raw mc_all timeline + bg_gap_max,
  rails/power energy windows, TTFT/TPOT/joules per cell.
- Figure 1 (tier diagram): device pool 8192 tok / host pool 16385 tok
  (2.0x) / file shards; eviction path prime+junk; direct vs kernel H2D path.
- Key design facts: Orin retain = device-hit resume; A5000 retain =
  post-eviction cold resume (denominators differ by design — absolute
  diffs are the honest cross-board comparison); A5000 blocks average
  bg1+bg4 cells, Orin matrix cells are bg4 only.

### 3. Eval — 0.5 page + 2 figures
- Figure 2 (TTFT ratios with 95% paired-block CIs, per-board panels, never
  pooled): A5000 host/retain 0.351 [0.334,0.374] L2048, 0.167 [0.164,0.169]
  L6011 (clears 10% margin, faster); Orin host_d2h ~= retain 0.978/0.993
  (practically equivalent); Orin h2d-direct +13.35s over recompute L6011
  vs A5000 h2d-direct +2.10s over retain (server-wide bg cost: bg_tpot 2.1x
  sealed audit; bg_gap_max 6.1x paired); file-tier ~17s vs 0.12–0.15s
  device (~130x, descriptive across strands).
- Figure 3 (interference): bg_gap_max retain 0.087s vs h2d 14.6s (Orin,
  L6011) + raw mc_all timeline sample; MemAvailable flat pre→post
  (pools preallocated; per-request deltas invisible on both boards).
- Ratio-of-ratios (independent resampling, boards never pooled):
  host/retain TTFT RoR Orin/A5000 2.79 L2048, 5.93 L6011 — direction held
  under version confound; reported as arch+version joint effect, never
  arch-only.
- Statistics: paired-block 95% bootstrap B=10000 seed 20260921, 10% margin;
  joules cross-board not compared (Orin rails vs A5000 GPU1 power.draw
  incommensurate — within-board ratios only, side by side).

### 4. Conclusion — 0.3 page
- Restate narrow claim only: host offload frees HBM and is fast on
  discrete; on unified Orin it frees no MemAvailable and ties device-hit,
  so retain dominates at 6K; direct backend is required for H2D (kernel
  path 3/3 fatal) but slow; file tier serves (~16s) yet costs ~130x device.
- Honest limits: version confound 0.5.20 vs 0.5.16 (no arch-only claims);
  n=5 blocks (wide CIs by construction); EMC utilization-only; A5000 pilot
  was n=1 sensitivity (never presented as main); ShareGPT slice is
  sensitivity, same direction.
- Next: 0.5.16 A5000 control to isolate arch; larger-block replication.

### 5. Refs — 0.2 page, 4–5 refs
1. SGLang HiCache (hierarchical KV cache design).
2. PagedAttention (vLLM; non-contiguous KV management baseline).
3. NVIDIA Tegra/Orin memory AppNote (unified memory behavior).
4. Arya-Simmhan / Pagoda (relevant prior — confirm exact citation).
5. LMCache (CPU-offload KV serving baseline).

## Claim sentences (verbatim, citable to sealed artifacts)
- C1: "On discrete A5000, kernel-backend host restore TTFT is 0.35x
  (L2048) / 0.17x (L6011) of retain (95% paired-block CIs [0.334,0.374] /
  [0.164,0.169], clears 10% margin; results/cross-summary.json)."
- C2: "On unified Orin, host spill frees no MemAvailable and host_d2h
  restore TTFT is practically equivalent to device-hit retain (ratios
  0.978/0.993; results/summary.json)."
- C3: "H2D requires the direct backend (kernel H2D 3/3 fatal); direct
  host restore costs +13.35s over recompute on Orin L6011 vs +2.10s over
  retain on A5000 (paired diffs; cross-board absolute diffs only,
  version-confounded)."
- C4: "Orin file-tier restore serves at ~17.0s vs 0.12–0.15s device-hit
  (~130x, Strand A vs B descriptive; results/campaign-v2-summary.json)."
- C5 (limits): "All cross-board directions are joint arch+version effects
  (sglang 0.5.16 Orin vs 0.5.20 A5000); n=5 blocks; EMC is a
  utilization proxy, not GB/s."

## Provenance (no invented numbers)
- Orin matrix: results/summary.json rev2 (60/60 meas PASS) +
  results/campaign-v2-summary.json (Strand A 15/15, Strand B 10/10).
- A5000 main: runs/imported/a5000-main (seal 168/168 OK, 80/80 PASS) +
  results/cross-summary.json rev1 (36 within + 12 RoR).
- A5000 pilot (/tmp/opencode/pilot, n=1): sensitivity only, never main.
- NOTHING committed (working tree dirty for review).
