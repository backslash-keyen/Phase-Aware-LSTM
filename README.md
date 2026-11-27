# PhaseAwareLSTM
PhaseAwareLSTM provides a minimal, self-contained MATLAB implementation of projection-based transfer learning for low-channel EEG decoding.   The code accompanies the corresponding research work and focuses on clarity, reproducibility, and simplicity rather than full-scale dataset distribution.

## Replication

`replication/` reruns every experiment of the paper (V. K N, S. Sundaram, C. N. Gupta, *Phys. Scr.* **100** (2025) 126003) with the released code and builds its tables from fresh, seeded results. Start at [`replication/README.md`](replication/README.md). The released code itself is left as shipped.

## Working in this repository

| you want | read |
|---|---|
| the rules every session follows | [`CLAUDE.md`](CLAUDE.md) |
| Keyen's standing instructions and decisions | [`keyen.md`](keyen.md) |
| who owns which path | [`AGENTS.md`](AGENTS.md) |
| branch, PR and merge rules | [`docs/GIT_PROTOCOL.md`](docs/GIT_PROTOCOL.md) |
| what was done and found | [`LOGBOOK.md`](LOGBOOK.md) (search it, don't read it whole) |
| what is known and not started | [`BACKLOG.md`](BACKLOG.md) |
| the HTML wiki | `pip install -r docs/wiki/_build/requirements.txt`, then `mkdocs serve` |

Once per clone, turn on the hooks that keep `main` PR-only:

```
git config core.hooksPath .githooks
```
