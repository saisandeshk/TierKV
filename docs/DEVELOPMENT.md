# DEVELOPMENT (device sync + boundaries)

- Workstation: edit/test here. Orin checkout: `/media/ssd/saisandesh/TierKV` (sudo OK). A5000 checkout: writable path TBD (no sudo).
- Sync: commit coherent checkpoints on workstation, push `main`, pull on device. Never commit `runs/`, logs, credentials, weights, images.
- Docker via existing restricted wrapper only. One server at a time, owned loopback port, stop via same controller. Preserve other users' work.
- Keep `src/tierkv` GPU-free; device paths in ignored local config. Record controller evidence dir per cell for seal/audit.
