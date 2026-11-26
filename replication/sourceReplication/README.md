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

The released code can't run this baseline either. `define_layerArchitecture.m` has no case for `baseTransfer`, and the released runner can't train the dual-channel source (`source-domain-2`). `runTargetBranch` therefore leaves it out. Its paper value equals the no-transfer value.

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

10-fold means (test %):

| baseline | paper / HPC | branch hpc | branch ours |
|---|---|---|---|
| no transfer | 56.29 | | |
| FFNN projection | 57.06 | | |
| LSTM projection | 56.61 | | |
| GRU projection | 54.74 | | |
| (dual-channel transfer: equals no transfer in the paper's results, not runnable) | 56.29 | — | — |
