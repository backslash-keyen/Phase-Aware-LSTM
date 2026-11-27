# Code map

Where to look for what. The complete file list is the [Tree map](../reference/tree.md); ownership
is in [`AGENTS.md`](../../../AGENTS.md).

| you want | look in |
|---|---|
| the released entry point | [`projecitonTL_mainScript.m`](../../../projecitonTL_mainScript.m) (name as shipped) |
| released training, data loading, models | [`utils/`](../../../utils) (never edited) |
| one-call replication | [`replicatePaper.m`](../../../replicatePaper.m), [`replication/runAll.m`](../../../replication/runAll.m) |
| the seeded, resumable driver | [`replication/runReplication.m`](../../../replication/runReplication.m) |
| table layouts (no numbers) | [`replication/paperTables.m`](../../../replication/paperTables.m) |
| building the tables | [`replication/buildTables.m`](../../../replication/buildTables.m) |
| opt-in variants | [`replication/variants/`](../../../replication/variants) |
| every phase-aware layer version | [`replication/versions/README.md`](../../../replication/versions/README.md) |
| source-model replication | [`replication/sourceReplication/README.md`](../../../replication/sourceReplication/README.md) |
| diagnosis scripts, PC run helpers | [`replication/diagnostics/README.md`](../../../replication/diagnostics/README.md) |
