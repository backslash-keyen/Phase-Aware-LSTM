# Replication: seeded runs and automatic tables

This folder reruns every experiment of

> V. K N, S. Sundaram, C. N. Gupta, "Phase-aware LSTM projections for learning from heterogeneous electroencephalogram montages", *Phys. Scr.* **100** (2025) 126003, doi:10.1088/1402-4896/ae229d

with the released code, then builds all ten tables of the paper from the fresh results. Nothing is typed in by hand. The paper's published numbers are not stored anywhere in this repository: `paperTables.m` encodes only each table's layout (caption, column groups, sections, row labels) and which experiment fills each cell.

**No original file is changed.** Everything here is new. The only way it alters the released code's behaviour is the opt-in `shims/` folder, described below, which is off by default.

## Run it

From MATLAB (R2022b+ with Deep Learning Toolbox), with the full datasets under `datasets/folds/` in the same layout as the demo fold:

```matlab
addpath replication
runAll('dryRun', true)       % plan: what will run, what is missing, nothing trained
runAll()                     % train everything the released code supports, then build tables
runAll('useShims', true)     % also F3-F7, temporal masking and channel selection (see below)
buildTables()                % rebuild tables only, from results already on disk
```

Or open `replicatePaper.m` at the repo root, set its three options and press Run.

`runAll` is resumable. Each finished experiment is logged in `results/manifest.csv`, and a rerun skips it (`'force', true` reruns). Source-domain experiments always run before target-domain ones, because target runs load the models the source runs save to `pretrainedModels/`. A target run whose source models are missing is logged as `blocked` and not started.

Useful options of `runReplication` / `runAll`: `tables` (only what some tables need, e.g. `{'table2','table4'}`), `datasets`, `typeNets`, `seed`, `ncval` (`'auto'` counts the fold files), `parTrain`, `stopOnError`.

## Seeding and provenance

* The released code already seeds each fold: `perfold_training.m` calls `ensureReproducibility(i)` right before `trainNetwork` (`rng(i)`, `gpurng(i)`, deterministic GPU algorithms), and `makeDatastore.m` calls `rng(1)` before shuffling. So the fold seed is the fold index, as in the paper's runs.
* `runReplication` additionally sets a master seed (`rng(seed,'twister')`, `gpurng(seed,'Philox')`) before every experiment, so nothing that runs before the fold seed depends on leftover RNG state.
* Folds are trained serially by default (`parTrain = false`), so run order is fixed.
* Every manifest row records: config, master and fold seeds, the SHA-256 of every fold file used, MATLAB release, GPU and driver, host, git commit (and whether the tree had uncommitted changes), start and end time, and the `cvalresults/*.mat` file the run wrote.
* **GPU caveat.** `ensureReproducibility.m` calls `deep.gpu.deterministicAlgorithms`, which exists only from R2024b. On an older release with a GPU the original code fails at the first fold, so the driver stops up front with an explanation. Use R2024b+, or hide the GPU so training runs on CPU.

## Outputs

```
replication/results/
  manifest.csv          one row per experiment attempt (status, seeds, hashes, result file)
  fold_metrics.csv      one row per experiment x fold: train/val/test accuracy, macro
                        sensitivity and F1, test AUC, training time
  tables/
    table1.csv .md .tex ... table10.csv .md .tex
    tableN_stats.csv    one-tailed paired t-tests across folds (accuracy tables)
```

Accuracy cells are `mean (SD)` across folds in %, as in the paper. A blank cell means its experiment has no results yet, and the `.csv` and the table footnotes say why (`not-run`, `unsupported`, `missing-data`, `blocked`, `not-in-released-code`). Cells the paper itself leaves as "—" stay "—".

The stats files use the paper's test: one-tailed paired t-tests over folds on test accuracy, (a) phase-aware LSTM against every other projection of the same network, layout and column group, and (b) every transfer row against the no-transfer row of the same network and layout. They are computed in base MATLAB (`private/pairedTTest1.m`, same result as `ttest(a,b,'Tail','right')`), so the Statistics Toolbox is not needed.

## What fills each table

Domains are the ones `perfold_dataset.m` already knows. Data folders are relative to `datasets/folds/<CWL/NEC21_C{2,3} | MI/MI_C2>/`.

| Domain | Data folder | Montage |
|---|---|---|
| `source-domain` | `fullchannels/` | full montage |
| `source-domain-2` | `dualchannelsource/` | source restricted to the target's 2 channels |
| `target-domain` | `targetdomain1/` | Fp1–Fz (CWL), C3–C4 (MI) |
| `target-domain-2` | `targetdomain2/` | F3–F7 (CWL) |
| `temporalmasking-1` / `-2` | `temporalMasking_Fp1_Fz/`, `temporalMasking_F3_F7/` | masked target data |

| Table | Rows × columns | Experiments (dataset, classes, domain, layer) | Metric |
|---|---|---|---|
| 1 Hyperparameters | {DeepRNN, EEGNet, Transformer} × {CWL, MI} | none: read from `get_trainingOptions` for source and target domain | MiniBatchSize, InitialLearnRate, L2Regularization, MaxEpochs (after the 30 / 15 epoch caps) |
| 2 Source domain | net × {binary, multi-category}; CWL and MI-2a, Val/Test | `source-domain`, layer = net; CWL N2 / N3, MI N2 | accuracy mean (SD) |
| 3 Target baselines | No-TL; channel selection Fp1–Fz \| C3–C4; channel selection F3–F7 | No-TL: `target-domain`, layer = net. Channel selection: `source-domain-2` then `target-domain` with layer `baseTransfer` (shim) | accuracy |
| 4 Projection baselines, binary | net × {FFNN, LSTM, GRU, CHARM}; Fp1–Fz \| C3–C4 and F3–F7 | `target-domain` / `target-domain-2` (shim), layers `ffnn`, `lstm`, `gru` | accuracy |
| 5 F3–F7 multi-category | No-TL, channel selection, net × projections | CWL N3, `target-domain-2` (shim) | accuracy |
| 6 Phase-aware baselines, binary | net × {Phased-LSTM, Complex-LSTM, Phase-Aware-LSTM}; both layouts | layer `phase-lstm` | accuracy |
| 7 Phase-aware, Fp1–Fz multi-category | net × the same three | CWL N3, `target-domain`, `phase-lstm` | accuracy |
| 8 Temporal masking | DeepRNN × {Baseline, FFNN, LSTM, GRU, phase-LSTM}; both masks | CWL N2, `temporalmasking-1/2` (shim) | accuracy |
| 9 Phase parametrization | atan, atan2-a, atan2-b, tan-half-angle, numeric-sin-cos | tan-half-angle = released `phase-lstm`, CWL N2, `target-domain` | accuracy |
| 10 Complexity | {DeepRNN, EEGNet} × {FFNN, LSTM, GRU, complexLSTM, phaseLSTM} | CWL N2, `target-domain` | params, FLOPs, disk size, train time, inference time and memory, measured from the trained networks |

### Gaps in the released code

These cells stay blank, with the reason in the footnotes, until the missing code is added to the repository:

* **Not in the repository at all:** CHARM, Phased-LSTM and Complex-LSTM projections (Tables 4–7, 10); the atan, atan2-a, atan2-b and numeric-sin-cos phase parametrizations (Table 9); `get_trainingOptions_Graz_EEGNet.m` and `get_trainingOptions_Graz_Transformer.m`, so every MI EEGNet and MI Transformer cell (Tables 1–4, 6).
* **F3–F7 channel selection** (Tables 3, 5): `perfold_dataset.m` has a single `dualchannelsource/` folder, so there is no way to address a second, F3–F7 dual-channel source.
* **Temporal-masking data** (Table 8): `utils/crossvalidation/perFold_data_nec21_temporalMasking.m`, which made these folds, calls functions that are not in the repository and writes to a hard-coded `E:\` path. The masked folds must already exist under `temporalMasking_*`.

### The opt-in shims (`'useShims', true`)

`get_trainingOptions.m` and `define_layerArchitecture.m` accept only `source-domain` and `target-domain`, so every other domain that `perfold_dataset.m` can load fails before training. `shims/` holds two same-named wrappers that, when put first on the path:

* map `source-domain-2` to the source rules and `target-domain-2`, `temporalmasking-1/2` to the target rules, then call the **original** function unchanged (`shims/callOriginal.m`), so the options and architectures are exactly the original ones for that domain family;
* build `layerName = 'baseTransfer'` (channel selection), which `get_pretrainedLayers.m` loads but nothing builds, by fine-tuning the whole dual-channel source network with the original target-domain options. **This is an assumption**, our reading of "transfer learning with common channel selection".

Without the shims those rows are logged as `unsupported`.

### Other assumptions, flagged in the table footnotes

* Table 8 "Baseline" = DeepRNN trained directly on the masked target data (no transfer).
* Table 9 uses the DeepRNN backbone, and Table 10 uses CWL binary with the Fp1–Fz layout. The paper does not name either.
* Table 10 measurements are made on the machine that runs `buildTables`: FLOPs is the analytic estimate 2 × T × (projection weights), and inference memory is the projection activations for one trial plus all learnables.
* The number of folds is the number of fold files found (`'ncval','auto'`). Set it explicitly if the data folders hold more folds than the paper used.

## Files

| File | Role |
|---|---|
| `paperTables.m` | Layout of Tables 1–10 and the experiment behind every cell |
| `experimentGrid.m` | Every experiment the tables need, plus their source dependencies, in run order |
| `runReplication.m` | Seeded, resumable driver; writes `results/manifest.csv` |
| `collectResults.m` | Per-fold metrics from the saved `evals` → `results/fold_metrics.csv` |
| `buildTables.m` | Tables (.csv, .md, .tex) and paired t-tests |
| `runAll.m`, `../replicatePaper.m` | One-call entry points |
| `shims/` | Opt-in domain wrappers (see above) |
| `private/` | Helpers: path mirrors of the original loaders, hashing, t-test |
