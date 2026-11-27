# Backlog

Work known and not started. **Current state only:** an item leaves when it is done (the logbook
records it). **Nothing starts until Keyen picks it**; order within a section is the suggested run
order, not a commitment. Running work is in `AGENTS.md` (Assigned now).

## Replication (primary)

1. Per-session z-score test, 3 folds (folds built on the PC under `C:\Users\keyen\plstm_zsess`).
   Paused until CWL 3-class finishes.
2. Target-channel extension (PR 5): rebuild the target layouts with the fold builder (the PR's
   results are invalid: target data was cut from source folds). Paused.
3. Source-accuracy improvements (plan in the project files, `source-model/source-accuracy-plan-2026-11-27.md`;
   shuffle forced to `'never'` and last-iteration net found). Later.

## Phase-aware (paused, 2026-11-27)

When Keyen resumes, in this order:

1. PR 7 fold 2 for crossFeed+A, crossFeedMag+A, lowInit, specUQ, normMem (~45 min).
2. Rerun the 5-fold pipeline study (PR 6).
3. Channel-layout phase rows (PR 5, after its rebuild).
4. halfAng rerun (released halfAng NaNs on folds 4 and 10).
5. notrig 10-fold.

## Parked (don't start unless asked)

See `CLAUDE.md`, Settled.
