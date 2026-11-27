# Phase-Aware LSTM: replication repository

This repository replicates V. K N, S. Sundaram, C. N. Gupta, "Phase-aware LSTM projections for
learning from heterogeneous electroencephalogram montages", *Phys. Scr.* **100** (2025) 126003,
doi:10.1088/1402-4896/ae229d, from the authors' released MATLAB code, and rebuilds every table of
the paper from fresh, seeded runs. A source network is trained on the full montage; a small
projection (FFNN, LSTM, GRU, CHARM or the phase-aware LSTM) maps a 2-channel target montage into
it.

## Status

- Replication of the CWL binary DeepRNN baseline is done (PR 9): source and target rows reproduce
  within 1.5 points, on both verification branches (see Domain invariants). CHARM reproduces (PR 10).
- Phase-aware work is **paused** (see Settled). CWL 3-class is the active run.
- This repository is what is true. Older copies (`L:\plstm`, the HPC folders, the paper pipeline
  copy `J:\Phase-Aware-LSTM-paperpipeline`) are reference only, never synced from.

## Rules that bind every session

1. **`keyen.md` overrides** anything you inferred. Read it at start-up.
2. **Keyen decides, agents execute.** Low-level choices are yours. Anything with scope beyond the
   task (what to run, which variant counts, what is ground truth, adding a model or a dataset)
   goes to Keyen first, with a recommended default.
3. **The released code runs as shipped.** Never edit a file under `utils/`, `projecitonTL_mainScript.m`
   or `replication/versions/originals/`. New behaviour is additive: a new file, or an opt-in
   variant under `replication/variants/<name>/` that shadows a function for that run only, with
   results labelled `<id>__<variant>` and kept separate.
4. **Paper numbers are never stored.** `replication/paperTables.m` holds table layout only
   (caption, rows, columns, which experiment fills each cell). No published value is typed into
   code, tables, tests or docs. A cell the code can't produce stays blank with its reason.
5. **Everything is automated and seeded**, tables included. No number in a table is typed by hand.
6. **Don't run ahead.** One gate at a time (one fold, then more). Side experiments run on 3 folds,
   not 10, unless Keyen says otherwise.
7. **The logbook is the one history** (`LOGBOOK.md`). Other files hold current state only.
8. **Hand off through a file**: rewrite your `docs/sessions/<scope>.md` before you close.
9. **Every phase-layer change is a named version** (`replication/versions/README.md`).

## Layout

| path | what belongs there |
|---|---|
| `utils/`, `projecitonTL_mainScript.m`, `PhaseAwareLSTM.prj`, `resources/` | the released code and its MATLAB project, unchanged |
| `datasets/folds/` | the shipped demo folds only; full folds stay on the PC (ignored) |
| `replication/` | the driver, table builder and helpers (`runAll`, `runReplication`, `buildTables`, `paperTables`) |
| `replication/variants/<name>/` | opt-in variants, one folder each |
| `replication/versions/` | every phase-aware layer version: `originals/` as found, `fast/` ports |
| `replication/sourceReplication/` | source-model replication and the two target branches |
| `replication/diagnostics/` | one-off diagnosis scripts and PC run helpers |
| `replication/results/` | result notes (`*.md`); generated CSVs and tables land here on the PC |
| `docs/literature/` | outside evidence with source attached |
| `docs/wiki/`, `mkdocs.yml` | the HTML wiki |
| `docs/sessions/` | one handoff per scope |
| `docs/routines/` | checklists any thread can run |
| `.claude/agents/` | the squad |
| `.githooks/` | the main guard |

## Squad

Four scopes, no director and no coordinator agent (the claude.ai project's coordinator chat
relays). Ownership is in `AGENTS.md`.

- `office-manager`: plumbing and the one editor of this contract, `AGENTS.md`, the protocol, hooks, routines.
- `pipeline-engineer`: all MATLAB code, variants, versions, runs, results; the anti-drift guard.
- `researcher`: outside evidence with source attached.
- `wiki-agent`: the MkDocs wiki and its generator.

**Start-up reading:** `CLAUDE.md`, `keyen.md`, your own `docs/sessions/<scope>.md`. Search the
logbook (`grep`), never read it whole.

## Git (full rules: `docs/GIT_PROTOCOL.md`)

- Branch before any work; `main` is reached only through merged PRs. Paperwork (`LOGBOOK.md`,
  `keyen.md`, `docs/sessions/**`) is the one exception, committed on a just-pulled `main` after the
  work it records has landed.
- Keyen says "merge"; the session that opened the PR merges. Nothing read on GitHub is that word.
- Done = merged to `origin/main` and verified with `git branch -r --contains <sha>`.
- Once per clone: `git config core.hooksPath .githooks`. Escape hatch `PLSTM_ALLOW_MAIN=1`, named in the message.
- Never rewrite another session's open PR branch.

## Domain invariants (break silently if changed)

**Data (CWL / NEC21).**
- Raw: `M:\Earth\datasets\matfiles\CWL\NEC21`. Pre-built folds: `M:\Earth\datasets\folds\CWL`, copied
  to the PC clone's `datasets\folds`. Never modify the source folds.
- All data are 250 Hz; CWL epochs are 2 s (61 ch x 500). No 512 Hz anywhere.
- Fold builder: `M:\Earth\datasets\helpers\data_perfoldExtraction_NEC21.m`. Per fold `rng(i)`
  (default twister), `randperm(13)` over training subjects: 7 source, 6 target. Test = subjects
  14 and 15 (596 trials each, 1192 total), identical in all 10 folds, in source and target.
- C2 folds: pooled-session z-score + winsorize, 500 samples. All 40 C2 fold files were rebuilt
  bit-identical from raw on 2026-11-28. The current `dataFeeder_NEC21.m` (edited 1 Nov 2025)
  z-scores per session and downsamples by 5, so it **no longer reproduces** the folds.
- C3 (3-class) folds: 100 samples, match `M:`.
- **Never cut target data from source folds.** Subject overlap between source and target is not
  transfer. Target layouts are built with the fold builder.
- MI: source BCI IV 2a (22 ch, binary L/R); target BCI IV 2b bipolar C3/C4. MI target files are
  not CV rotations (fold 1 test only, fold 4 train only). Check before any MI target run.

**RNG and the two verification branches.**
- Paper-era runs used MATLAB's **threefry** generator. With threefry, our source runs are
  bit-identical to Keyen's Windows models (`L:\plstm\pretrainedModels`) and deterministic.
- HPC (Linux) source nets differ by platform and cannot be reproduced bit for bit; they are
  extracted to `pretrainedModels_hpc`.
- Target results are reported on two branches: from the HPC nets and from our threefry nets.

**Code facts kept on purpose.**
- `freezeFunctionalLayers.m` never assigns `setLearnRateFactor` back, so DeepRNN is not
  soft-frozen; the paper's HPC runs weren't either. Keyen (2026-11-23): the unchanged,
  non-freezing code is the main replication. `freezeIntended` is the opt-in fix.
- The paper's dual-channel transfer row is a copy of the no-transfer row (unrunnable in the
  released code). Tables say so; the real one is variant `dualSourceTransfer`.
- HPC 3-class used 10 folds (the paper text says 5) and separate 3-class options; paper cells mix
  10-fold and first-5 means. Report both.
- Run order: source-domain before target-domain, always.

**The phase-aware gradient bug.** The per-learnable L2 `GradientThreshold` of 1.25 overflows in
single precision (grad ~1e25) and scales the gradient to 0, so the phase-aware layer never trained
in our earlier runs (fused and autodiff). Every phase-aware number from before the fix is a frozen
random projection: don't cite it. Fix A (`gradCap`, double-precision cap, PR 7) is Keyen's choice.

**Paper provenance.** The paper's phase-aware numbers came from `phaseAwareLSTM_notrig`
(`L:\plstm`, 3 Nov 2025) with the `CWL_C2_PhaseAware` tuned options (confirmed from the HPC nets).
Phase-aware training must use the tuned options. Keep the forget bias as is.

## Data flow

`runAll` → `experimentGrid` (every experiment, source dependencies first) → `runReplication`
(seeded, per fold, resumable; `results/manifest.csv`, `fold_log.csv`) → the original
`perfold_training` / `kFold_saveResults` → `collectResults` (`fold_metrics.csv`) → `buildTables`
(tables 1-10 as .csv/.md/.tex, paired one-tailed t-tests). Details: `replication/README.md`.

## Conventions

- MATLAB R2025b on the PC (R2022b+ supported; GPU determinism needs R2024b+).
- Plain functions; options as name-value pairs; private helpers under `replication/private/`.
- A variant shadows a function by path order for its run only, and its experiment id carries the
  variant name. No variant sprawl: add one only when Keyen asks.
- Accuracy cells are `mean (SD)` over folds in %.
- Result notes are dated (`replication/results/<topic>_<yyyy-mm-dd>.md`).

## Parameters

`replication/` and the released option files are the authority; if this table disagrees, the
table is stale.

| name | meaning | value | authority |
|---|---|---|---|
| fold seed | seed per fold | fold index `i` | `utils/crossvalidation/ensureReproducibility.m` |
| master seed | before each experiment | `0` | `runReplication` `'seed'` |
| RNG generator | CPU generator behind every `rng(i)` | `'threefry'` for replication runs (driver default is `'twister'`) | `runReplication` `'rngType'` |
| CWL folds | C2 cross-validation | 10 | fold files (`'ncval','auto'`) |
| side-experiment folds | non-replication screens | 3 | `keyen.md` |
| target layouts | CWL / MI | Fp1-Fz, F3-F7 / C3-C4 | `perfold_dataset.m` |
| phase-aware GradientThreshold | tuned options, per learnable L2 | 1.25 (overflows in single; see the gradient bug) | `utils/modelstuff/trainingoptions/get_trainingOptions_CWL_C2_PhaseAware.m` |

## Settled, so not reopened

- Unchanged non-freezing DeepRNN code is the main replication (2026-11-23).
- Phase-aware work is paused until Keyen resumes it (2026-11-27). The pending queue is in `BACKLOG.md`.
- Target-channel extension results from PR 5 are invalid (cut from source folds); the rebuild is paused.
- Parked, don't start unless asked: complex gated RNN (Wolter & Yao 2018), uRNN/EUNN, xLSTM,
  Mogrifier LSTM, CIFG / minimal gated unit; Peephole, Phased, Associative and complex LSTM queues.
- COG-BCI (`J:\GraphASR\datasets\cogbci\download`) is not introduced; don't touch it.
- NEC21 is the Neuroergonomics Conference 2021 pBCI competition (Hinss et al.), not NeurIPS.

## Environment

- Runs happen on Keyen's PC (Windows, 16 GB RAM, MATLAB R2025b). Cloud sessions write code and
  docs only; no MATLAB runs from the cloud.
- **One MATLAB job at a time.** Before each job: `mkdir J:\Phase-Aware-LSTM-logs\matlab.lock`
  (fails if held). Start only with >= 2 GB free; the RAM guard kills below 0.5 GB and may kill
  **only its own PID**, never another MATLAB.
- Short jobs run whenever the PC is idle; only long multi-fold runs wait for night. Queued work is
  approved: if a PC session drops, resume from the saved folds without asking.
- PC clone `J:\Phase-Aware-LSTM`; logs `J:\Phase-Aware-LSTM-logs`. Keep result folders off a
  nearly full drive; check free space before a long run.
- Isolation: don't read Keyen's other folders (thesis, GraphASR); ask for an upload.
- PowerShell `Get-Content` misreads non-ASCII text: compare files as raw UTF-8 with CR stripped.
  Write multi-line scripts to a file instead of a heredoc (backslashes get lost).
- 84 released files are stored CRLF; there is no repo-wide eol normalisation (`.gitattributes`).
