# Handoff: pipeline engineer

## Next
- Finish CWL 3-class (branch `claude/cwl-3class`), then tell the z-score thread to resume.

## Settled
- See `CLAUDE.md` Domain invariants and Settled; `keyen.md` Decisions.

## Broken or open
- `dataFeeder_NEC21.m` (edited 1 Nov 2025) no longer reproduces the C2 folds.
- Phase-aware gradients still explode with Fix A (median ~3e21); paused.

## Already fixed, do not re-report
- RNG mismatch with the paper runs: use `'rngType','threefry'`.
- Phase-aware layer never trained (GradientThreshold overflow): known, Fix A in PR 7.

## For the coordinator
- (none)
