# Source-model replication (CWL C2, DeepRNN)

This folder retrains the CWL 2-class DeepRNN source models the way the paper's runs did. It also checks how close the retrained models come to the paper's own nets, and runs the target baselines from two sets of source nets.

## Findings

### 1. The paper's runs used the threefry generator

`makeDatastore.m` calls `rng(1)` before it shuffles the datastores, and `ensureReproducibility.m` calls `rng(i)` before training. `rng(seed)` keeps whatever generator is active at that point. The Nov 2025 paper runs, on both the HPC and the authors' Windows PC, ran with **threefry** active. `runReplication` used to force **twister** (`rng(seed,'twister')`). As a result, its trial order and weight initialisation matched no paper run.

`checkShuffleOrder.m` rebuilds the shuffle under each generator and compares it with the validation order stored in an HPC net. Only threefry reproduces it:

| generator | reproduces the HPC trial order |
|---|---|
| twister (the old runner default) | no |
| **threefry** | **yes, exactly** |
| philox, combRecursive, simdTwister, multFibonacci | no |

`runReplication` now takes `'rngType'`. The default is still `'twister'`, so existing results and ids don't change. Use `'rngType','threefry'` to match the paper.

### 2. With threefry, the source models reproduce bit for bit

| run | result |
|---|---|
| fold 1, trained twice on this PC | bit-identical to each other (deterministic) |
| all 10 folds vs the authors' Windows source models (`pretrainedModels`, 4 Nov 2025) | **bit-identical**, every parameter of every fold |
| mean over 10 folds | val 67.95, test 60.22 |
| HPC run behind the paper's table | val 67.23, test 60.20 |

The HPC nets can't be reproduced bit for bit on Windows. They share the initialisation with the Windows nets: on fold 1 the LSTM input weights have cosine 0.62, and independent seeds give about 0. They drift apart during training, though, because the HPC uses a different platform and GPU. The test accuracy agrees to within 0.02 points on average. The per-fold differences are up to 3.7 points, at the level of seed noise.

### 3. The paper's target runs started from the HPC nets

In the paper's target results (FFNN, phase-aware LSTM, ...), the transferred LSTM layers have cosine 1.000 with the nets inside the HPC source result file. Against the Windows `pretrainedModels` files they reach at most 0.51. So the exact source models behind the paper's target numbers exist only inside `cval_DeepRNN_source-domain_2_DeepRNN_251104_1925.mat`. `extractHPCSourceNets.m` writes them out as loadable `sourceDomainClassifier_N2_DeepRNN_<i>.mat` files, and does the same for the dual-channel `sourceDomain2Classifier` nets.

### 4. In the paper's results, "transfer with common channel selection" equals no transfer

The HPC no-transfer and baseTransfer (Fp1–Fz) result files report identical accuracies on every fold. Their trained nets are bit-identical too. The baseTransfer file stores the dual-channel source nets, but training never used them: the trained nets only share the seeded initialisation with them (cosine about 0.65).

The released code can't run this baseline either. `define_layerArchitecture.m` has no case for `baseTransfer`, and the released runner can't train the dual-channel source (`source-domain-2`). `runTargetBranch` therefore leaves it out.

**The dual-channel transfer baseline needs no run of its own.** The paper's row is an exact copy of no transfer, because the dual-channel source nets were never used. Any faithful rerun gives the no-transfer numbers.

**A real dual-channel transfer** is available as the opt-in variant `dualSourceTransfer` (`replication/variants/dualSourceTransfer`), which goes beyond the released code:
- **Source:** it trains the DeepRNN source on the Fp1–Fz source folds (`dualchannelsource/`), with the source-domain options. The result is `sourceDomain2Classifier_N2_DeepRNN_<i>.mat`.
- **Target:** it starts the target model from that dual-channel source net and fine-tunes all of it, with the same target options as the other baselines.

Its results are a separate, labelled row (`<id>__dualSourceTransfer`). They don't replace the unchanged replication row.

```matlab
runReplication('ids',{'cwl_N2_source-domain-2_DeepRNN_deeprnn'},'variant','dualSourceTransfer','rngType','threefry');
runReplication('ids',{'cwl_N2_target-domain_DeepRNN_basetransfer'},'variant','dualSourceTransfer','rngType','threefry');
```

## How to reproduce

Work in a scratch copy of the repository. The original code writes models to `pretrainedModels/` and results to `cvalresults/` under the current folder. The fold files go in `datasets/folds/CWL/NEC21_C2/` (`fullchannels`, `dualchannelsource`, `targetdomain1`).

```matlab
addpath('replication'); addpath(fullfile('replication','sourceReplication'));

% Stage 1: one fold, trained twice, compared with the authors' models
T = runSourceReplication(1, 'refDir', 'L:\plstm\pretrainedModels\cwl\cwl1\DeepRNN');

% Stage 2: all 10 folds
T = runSourceReplication(2, 'refDir', 'L:\plstm\pretrainedModels\cwl\cwl1\DeepRNN');

% The paper's own source nets, from the HPC result files (read only)
d = 'L:\plstm\HPC\PhaseAwareLSTM\cvalresults\CWL\DeepRNN';
out = fullfile('pretrainedModels','cwl','cwl1','DeepRNN');
extractHPCSourceNets(fullfile(d,'cval_DeepRNN_source-domain_2_DeepRNN_251104_1925.mat'), out, 'sourceDomainClassifier_N2_DeepRNN');
extractHPCSourceNets(fullfile(d,'cval_DeepRNN_source-domain-2_2_DeepRNN_251105_2234.mat'), out, 'sourceDomain2Classifier_N2_DeepRNN');

% Checks
checkShuffleOrder('datasets/folds/CWL/NEC21_C2/fullchannels/source_fold1.mat', fullfile(out,'sourceDomainClassifier_N2_DeepRNN_1.mat'))
compareSourceNets(fileA, fileB)

% Target baselines from whichever source nets are in pretrainedModels/ (fold 1 first)
runTargetBranch(1); runTargetBranch(1:10);
```

The two target branches are two scratch copies:
- **hpc:** `pretrainedModels/` holds the extracted HPC nets. This tests the target code given the paper's exact source models.
- **ours:** `pretrainedModels/` holds the threefry nets retrained here. This tests the whole pipeline from scratch.

Each source fold takes about 4.5 minutes on an RTX-class GPU under R2025b.

## Target baselines (CWL C2, DeepRNN, Fp1–Fz, test accuracy %)

Results are filled in as the runs finish.

Fold 1, branch hpc (the paper's exact source nets):

| baseline | HPC fold 1 | branch hpc fold 1 |
|---|---|---|
| no transfer | 56.46 | 56.54 |
| FFNN projection | 56.04 | 61.91 |
| LSTM projection | 61.24 | 62.67 |
| GRU projection | 56.21 | 61.74 |

The training options, the architecture and the starting source net are identical to the HPC run. Both runs fine-tune the transferred source layers far from their start: the soft freeze in `freezeFunctionalLayers.m` is a no-op in both. So the fold-level differences come from the training trajectory (GPU numerics). No transfer, which starts from scratch, agrees to within 0.1 point. A single fold can't separate that noise from a real gap, so the 10-fold means are the test.

10-fold means (val / test %, 27 Nov 2026, R2025b):

| baseline | paper / HPC | branch hpc (paper's source nets) | branch ours (threefry retrain) |
|---|---|---|---|
| no transfer | 59.52 / 56.29 | 58.15 / 54.78 | 58.15 / 54.78 |
| FFNN projection | 60.96 / 57.06 | 59.98 / 56.06 | 60.13 / 55.76 |
| LSTM projection | 62.28 / 56.61 | 62.14 / 57.68 | 62.13 / 57.20 |
| GRU projection | 58.66 / 54.74 | 59.39 / 55.74 | 58.44 / 54.81 |
| dual-channel transfer, as released (= no transfer, needs no run) | 59.52 / 56.29 | 58.15 / 54.78 | 58.15 / 54.78 |
| dual-channel transfer, real (`dualSourceTransfer`, beyond the paper) | — | 62.65 / 57.72 | 59.82 / 55.63 |

Used for real, the dual-channel source nets beat no transfer by 0.9 to 2.9 test points. The HPC branch ends up close to the best projection (LSTM, 57.68). The two branches differ by about 2 points. Their dual-channel source nets differ, because the HPC nets were trained on a different platform. The HPC dual-channel sources also score higher: val 60.08 / test 56.48, against 58.01 / 55.16 for ours. On top of that, the per-fold SD is about 3 points. Our threefry dual-channel source nets are bit-identical to the authors' Windows files on all 10 folds.

Every mean test accuracy lies within 1.5 points of the paper. The mean absolute per-fold difference from the HPC run is 0.5 to 2.7 points, which is the platform noise of a single fine-tuning run. No transfer doesn't use a source net, so it comes out identical in both branches. Its 1.5-point gap from the HPC is therefore pure Windows-vs-HPC training noise. All four baselines fall between 54.7 and 57.7 in every column. Their order differs between columns: the paper has FFNN > LSTM > no transfer > GRU, while both branches have LSTM first. So the paper's small gaps between these baselines, about 2 points, are not resolved by a single 10-fold run.
