# CWL 3-class (NEC21_C3) DeepRNN replication

Results of the CWL 3-class (low / medium / high workload) DeepRNN rows, run the same way as the
2-class work in `replication/sourceReplication` (two branches of source nets, MATLAB threefry RNG,
released code unchanged in the first pass). Runs finished on 28 Nov 2026, R2025b, one GTX 1650.

## Setup

- **Folds.** `datasets/folds/CWL/NEC21_C3` = `M:\Earth\datasets\folds\CWL\NEC21_C3` byte for byte
  (40 of 40 files). They were built on 5 Nov 2025 with 100 samples per trial, so they were used as they are.
- **Target layout.** All runs are Fp1-Fz (`target-domain`, `targetdomain1/`). This is the layout of
  every HPC 3-class run script and result file. See table `table5fp1fz` in `paperTables.m`.
- **Two option passes.**
  - *released*: the released `get_trainingOptions.m`, which has no 3-class branch. 3-class is
    trained with the 2-class DeepRNN options: LR 0.0029, mini-batch 197, 30 source / 15 target epochs.
  - *paper opts*: variant `c3PaperOptions`, the options stored in the HPC 3-class result files.
    These are `get_trainingOptions_CWL_C3_DeepRNN.m` with beta1 0.9: LR 0.004687, mini-batch 234,
    68 source / 50 target epochs, best-validation network.
- **Two source branches.**
  - *HPC nets*: the source nets stored in the HPC file
    `cval_DeepRNN_source-domain_3_DeepRNN_251105_0820.mat`, extracted with `extractHPCSourceNets.m`.
  - *ours*: our threefry retrain (`runSourceReplication(…, 'ncats', 3)`).
  - No transfer does not load a source net, so it has no HPC-nets column. The HPC has no 3-class
    dual-channel source, so dual-channel transfer exists only in the ours branch (variant `dualSourceTransfer`).
- **CHARM** is CHARM-base (variant `charmBase`, PR 10).
- **Determinism.** Source fold 1 retrains bit-identically (16 of 16 parameters), also while a
  second MATLAB job runs on the same GPU.
- **No bit-identity reference.** The only Windows 3-class source files
  (`L:\plstm\pretrainedModels\cwl\cwl2\DeepRNN`, folds 1-3, 1 Nov 2025) use an older recipe
  (LR 0.0124, mini-batch 246) on folds from before the 5 Nov rebuild. They cannot serve as a
  bit-identity reference the way the 2-class files did.

## Results, 10 folds

Each cell is mean val / mean test, with the test SD in brackets (%). *HPC* is the matching HPC
3-class result file (10 folds, 5 Nov 2025).

| Row | HPC nets, released | HPC nets, paper opts | Ours, released | Ours, paper opts | HPC |
|---|---|---|---|---|---|
| Source (full montage) | n/a | n/a | 45.72 / 42.76 (3.65) | 45.49 / 41.35 (1.95) | 46.63 / 42.33 (2.09) |
| Dual-channel source | n/a | n/a | 46.35 / 42.81 (0.87) | 46.15 / 42.49 (0.67) | no HPC file |
| No transfer | n/a | n/a | 46.28 / 42.39 (1.62) | 47.36 / 42.89 (0.62) | no HPC file |
| Dual-channel transfer | n/a | n/a | 47.30 / 42.91 (0.52) | 45.73 / 41.58 (2.94)¹ | no HPC file |
| FFNN | 47.36 / 43.12 (0.64) | 47.57 / 42.99 (0.44) | 47.47 / 43.18 (0.78) | 46.95 / 42.62 (0.43) | 47.40 / 42.81 (0.92) |
| LSTM | 46.96 / 42.87 (0.79) | 47.19 / 42.63 (0.88) | 47.07 / 43.08 (0.50) | 47.16 / 42.69 (0.70) | 47.18 / 42.65 (0.85) |
| GRU | 45.86 / 42.20 (2.25) | 45.46 / 41.79 (3.60)² | 47.30 / 42.93 (0.56) | 47.38 / 42.76 (1.18) | 46.76 / 42.75 (1.19) |
| CHARM-base | 46.73 / 42.28 (0.83) | 47.14 / 42.84 (0.97) | 47.23 / 42.56 (0.71) | 47.20 / 42.73 (1.23) | 47.46 / 43.14 (0.82) |

¹ Fold 1 collapsed to one class (33.33). The other nine folds average 42.50.
² Fold 4 collapsed (31.94). The other nine folds average 42.88.

## Results, first 5 folds

Some 3-class cells in the paper match the mean of the first five HPC folds rather than all ten,
so the same table over folds 1-5 is given too.

| Row | HPC nets, released | HPC nets, paper opts | Ours, released | Ours, paper opts | HPC |
|---|---|---|---|---|---|
| Source (full montage) | n/a | n/a | 44.55 / 44.37 (3.17) | 43.30 / 40.30 (1.11) | 46.40 / 42.43 (1.33) |
| Dual-channel source | n/a | n/a | 46.02 / 42.60 (0.91) | 45.87 / 42.27 (0.49) | no HPC file |
| No transfer | n/a | n/a | 47.18 / 42.79 (0.61) | 47.84 / 43.20 (0.74) | no HPC file |
| Dual-channel transfer | n/a | n/a | 47.51 / 43.18 (0.53) | 44.73 / 40.78 (4.18)¹ | no HPC file |
| FFNN | 47.86 / 43.21 (0.62) | 48.13 / 43.02 (0.57) | 47.70 / 43.66 (0.69) | 46.87 / 42.75 (0.39) | 47.94 / 43.26 (0.90) |
| LSTM | 47.05 / 42.86 (0.72) | 47.70 / 42.94 (0.98) | 47.26 / 43.04 (0.70) | 47.60 / 42.61 (0.86) | 47.20 / 42.82 (1.03) |
| GRU | 46.76 / 42.63 (1.36) | 45.02 / 41.21 (5.22)² | 47.67 / 43.06 (0.65) | 47.74 / 42.70 (1.31) | 47.63 / 42.91 (0.89) |
| CHARM-base | 47.64 / 42.91 (0.43) | 47.77 / 43.47 (0.86) | 48.09 / 43.02 (0.64) | 48.04 / 43.33 (0.29) | 47.89 / 43.13 (1.10) |

## Reading the tables

- Every projection cell is within 0.9 points of the HPC 10-fold test mean (most within 0.5),
  except the two cells with one collapsed fold.
- The projections, no transfer and dual-channel transfer all land at 42-43 % test. The 3-class
  target task barely separates them: the spread between rows is smaller than the fold-to-fold SD.

## Reproduce

From the root of a scratch copy whose `datasets/folds/CWL/NEC21_C3` holds the folds:

```matlab
addpath('replication'); addpath(fullfile('replication','sourceReplication'));
runSourceReplication(1, 'ncats', 3);  runSourceReplication(2, 'ncats', 3);   % add 'variant','c3PaperOptions' for the paper pass
runReplication('ids',{'cwl_N3_source-domain-2_DeepRNN_deeprnn'},'folds',1:10,'variant','dualSourceTransfer','rngType','threefry');
runTargetBranch(1:10, 'ncats', 3, 'layers', {'deeprnn','basetransfer'}, 'variant', {'dualSourceTransfer'});
runTargetBranch(1:10, 'ncats', 3, 'layers', {'ffnn','lstm','gru'});
runTargetBranch(1:10, 'ncats', 3, 'layers', {'charm'}, 'variant', 'charmBase');
```

For the HPC branch, put the extracted HPC nets in `pretrainedModels/cwl/cwl2/DeepRNN` first:

```matlab
extractHPCSourceNets('L:\plstm\HPC\PhaseAwareLSTM\cvalresults\CWL\DeepRNN\cval_DeepRNN_source-domain_3_DeepRNN_251105_0820.mat', ...
    fullfile('pretrainedModels','cwl','cwl2','DeepRNN'), 'sourceDomainClassifier_N3_DeepRNN');
```

Each source run writes about 1.2 GB (10 nets of 60 MB plus the result file).
