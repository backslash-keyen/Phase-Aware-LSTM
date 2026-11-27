# Reproducing a result

The commands, options and outputs are in the
[replication README](../../../replication/README.md); source-model runs and the two target
branches are in [`replication/sourceReplication/README.md`](../../../replication/sourceReplication/README.md).

Before a run on the PC (rules in [`CLAUDE.md`](../../../CLAUDE.md), Environment):

1. Take the lock: `mkdir J:\Phase-Aware-LSTM-logs\matlab.lock`. If it exists, another job runs.
2. Check at least 2 GB of RAM and enough disk are free.
3. Start with a dry run: `runAll('dryRun', true)`.
4. One gate at a time: one fold, then the rest.
