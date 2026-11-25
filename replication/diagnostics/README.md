# Phase-aware diagnostics

Scripts used for the Nov 25-26 phase-aware collapse diagnosis (see
`replication/results/phaseaware_2026-11-25.md`). None of them write results into the repo.

| File | What it does |
|---|---|
| `diag_projection_init.m` | Initial bias rows, gate statistics and the first ~50 training iterations of the phase-lstm projection (CWL C2 DeepRNN target) for a chosen layer variant. |
| `grad_vs_T.m` | Gradient norm of the projection against sequence length T (shows the ~10x growth every 15-25 steps with phase bias 0). |
| `layer_compare_matched.m` | Fused vs original layer: forward and gradient agreement on one batch (T=20, double, layer alone). |
| `ram_watch.ps1` | Logs free RAM every 10 s and stops MATLAB below 0.5 GB. |
| `run_lib.ps1` | Helpers for queued runs: one MATLAB job at a time, start only with >= 2 GB free. |

The two PowerShell files are generalised from the tested night-run versions (machine paths
replaced by parameters) and have not been run in this exact form. The one-off night runners
(hard-coded dates, folder swaps) were left out.

Notes from the runs:
- Swap temporary layer edits in from prepared copies, restore them in a `finally` block and check the hash.
- Compare files as raw UTF-8 with CR stripped; PowerShell `Get-Content` misreads non-ASCII text.
- Make each stage its own script; pausing a running runner between folds means killing the runner and any MATLAB it just started.
- `runReplication` skips experiments the manifest marks done; reruns need `'force', true`.
