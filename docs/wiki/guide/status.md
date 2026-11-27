# Where the replication stands

This page says which results are valid and where they are. It doesn't copy numbers: each line
links the file that holds them.

| part | state | evidence |
|---|---|---|
| CWL binary, DeepRNN: source and target baselines (FFNN, LSTM, GRU) | reproduced, both branches | [`replication/sourceReplication/README.md`](../../../replication/sourceReplication/README.md) |
| CHARM projection | reproduced | PR 10 |
| Dual-channel transfer row | the paper's row equals no transfer; a real one is variant `dualSourceTransfer` | [`replication/README.md`](../../../replication/README.md) |
| CWL 3-class | running | [`AGENTS.md`](../../../AGENTS.md) (Assigned now) |
| Phase-aware LSTM | paused; earlier numbers invalid (the layer never trained) | [`CLAUDE.md`](../../../CLAUDE.md), [`BACKLOG.md`](../../../BACKLOG.md) |
| Other target channel layouts | earlier results invalid; rebuild paused | [`BACKLOG.md`](../../../BACKLOG.md) |
| MI | not run yet | |

The per-run list is the generated [Results index](../reference/results.md).
