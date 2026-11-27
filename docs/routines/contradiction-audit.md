# Routine: contradiction audit

Run by the office manager after each stage lands, or on request. Re-check every claim at `HEAD`
before reporting it. Output: a list of findings, each with the files, which side is the authority,
and who owns the fix (`AGENTS.md`). Log the run in `LOGBOOK.md`.

## Checks

1. **Silent invariants agree everywhere.** Test subjects 14 and 15 (1192 trials); 250 Hz; 7 source /
   6 target subjects per fold with `rng(i)` twister in the fold builder; threefry for replication
   runs; source before target; 3 folds for side experiments. Search `CLAUDE.md`, `replication/**/README.md`,
   `docs/wiki/`, agent files. `CLAUDE.md` and the code are the authority.
2. **Parameters agree** between the `CLAUDE.md` table and the code it names (option files,
   `runReplication` defaults). The code wins; the table is fixed.
3. **No paper numbers stored.** `replication/paperTables.m`, tables, tests, docs and the wiki hold
   layout only. Any published value found is a finding for its owner.
4. **Released code unchanged.** `git diff --stat <initial commit ac1e221> HEAD -- utils projecitonTL_mainScript.m`
   is empty, and `replication/versions/originals/` is unchanged since it was added (`git log --oneline -- replication/versions/originals` shows only the adding commits).
5. **One owner per path.** Every `git ls-files` path falls under exactly one row of `AGENTS.md`.
6. **Dropped ideas stay dropped.** Nothing reopens the `CLAUDE.md` Settled list: parked models,
   PR 5 results, pre-fix phase-aware numbers, COG-BCI.
7. **Ambiguous names.** One name per thing (e.g. `phase-lstm` vs `phase-LSTM` in the layer name, the
   `notrig` versions, the two target branches), defined once.
8. **Cost.** Start-up reading is still `CLAUDE.md` + `keyen.md` + own handoff; `CLAUDE.md` under
   ~20 KB; agent files ~2-4 KB; no agent picks a bigger model than `keyen.md` sets.
