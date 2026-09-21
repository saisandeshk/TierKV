# PROTOCOL (frozen before claim runs; pilot may use draft)

1. Identity: image SHA, model+tokenizer rev, precision, launch flags, code+config hashes, device ID, input hashes.
2. Client log (monotonic): intended/actual dispatch, first content, each event, completion, usage, errors. SSE gaps != token gaps; TPOT from token counts.
3. Physical truth: `/proc/meminfo MemAvailable` around copy (10-50ms), peak duplication; never `cudaMemGetInfo` as physical. EMC% is utilization, not GB/s. Energy = monitored rails (`VDD_GPU_SOC+VDD_CPU_CV+VIN_SYS_5V0`, verify 32GB labels, no double-count DDR) over full window; 500ms samples cannot do per-token energy.
4. Design: paired randomized blocks (>=5 main), warmups separate, failures/OOMs indexed. 95% paired-block intervals, 10% practical margin. Boards/platforms reported separately, never pooled.
5. System validity: services/co-users, swap deltas, clocks/temp/slope, storage state, thermal admission/cooldown. No drop_caches/reboot/clock changes/evictions.
