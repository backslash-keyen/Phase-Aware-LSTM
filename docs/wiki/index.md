# Phase-Aware LSTM replication

This wiki documents the repository that replicates *Phase-aware LSTM projections for learning from
heterogeneous electroencephalogram montages* (V. K N, S. Sundaram, C. N. Gupta, *Phys. Scr.*
**100** (2025) 126003) from the authors' released MATLAB code.

Two rules hold the wiki together:

1. **The Guide explains; everything else points.** The Guide is for someone who has never seen the
   project. Every other page names the file that owns a fact instead of restating it, because a
   fact in two places drifts.
2. **Repo facts are computed, not typed.** The Reference pages are generated from the tree at
   build time. A section with no data says "not run yet".

## Where everything lives

| you want | read |
|---|---|
| what the project is, in plain words | [Overview](guide/overview.md) |
| the data, folds and domains | [The experiment](guide/experiment.md) |
| which results are valid, which await a rerun | [Where the replication stands](guide/status.md) |
| the rules every session follows | [`CLAUDE.md`](../../CLAUDE.md) |
| Keyen's standing instructions and decisions | [`keyen.md`](../../keyen.md), [Decisions index](reference/decisions.md) |
| who owns which path | [`AGENTS.md`](../../AGENTS.md) |
| branch and merge rules | [`docs/GIT_PROTOCOL.md`](../GIT_PROTOCOL.md) |
| how to run the replication | [`replication/README.md`](../../replication/README.md) |
| what was done and found | [`LOGBOOK.md`](../../LOGBOOK.md) |
| what is known and not started | [`BACKLOG.md`](../../BACKLOG.md) |
| every file in the repository | [Tree map](reference/tree.md) |

## Build this wiki

```
pip install -r docs/wiki/_build/requirements.txt
mkdocs serve                     # live preview at http://127.0.0.1:8000
mkdocs build --strict            # static HTML in site/ (never committed)
WIKI_LINKS=local mkdocs build    # links to local files instead of GitHub
PLSTM_RESULTS=replication/results mkdocs build   # list the PC's run manifest
```

On Windows PowerShell set the variables with `$env:WIKI_LINKS = "local"` first.
