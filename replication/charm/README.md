# CHARM projection baseline

The paper's tables have a CHARM row (Tables 4, 5, 7 and 10) next to the FFNN, LSTM and
GRU projections, but the released code has no CHARM. The layer and builder behind
the paper's numbers are in the Nov 2025 project (`L:\plstm\helperLibrary`):

* `modelframe/architectures/layerfunctions/baselines/charmLayer.m` (30 Oct 2025)
* `modelframe/modelfunctions/projectionbaslines/get_pLayer_charmLayer.m` (3 Nov 2025)

The HPC run of 4 Nov 2025 (`cval_DeepRNN_target-domain_2_CHARM_251104_1613.mat`)
used them for the CWL DeepRNN Fp1-Fz cell: 10-fold val 60.13, test 56.34.

CHARM is from Saeed, Grangier, Pietquin and Zeghidour, *Learning from heterogeneous
EEG signals with differentiable channel reordering* (arXiv:2010.13694). Each input
channel is encoded on its own (1-D conv, global max-pool over time) into a key
`h_j`. M learned canonical embeddings `c_i` score the keys, a softmax over the input
channels gives the reordering matrix `p_ij = softmax_j(c_i . h_j)`, and the output
channel i is `sum_j p_ij x_j` at every time step.

## Variants

All four are opt-in variants of `runReplication` (`replication/variants/charmCommon`
plus one folder each). Use at most one at a time. Each one writes its own fold
files and results, under `<id>__<variant>`.

| Variant | What it is |
|---|---|
| `charmBase` | **The replication row.** The Nov 2025 layer and builder, ported unchanged. Only the label changed: the original comments call it CHARM-CQ, but it has no values, no attention and no residual MLP, so it is CHARM-base (paper Sec. 2.1.1, Eq. 3). One conv layer (filter 8, d = 32, no activation) and global max-pool. M = the source net's input size (61). `sequenceInputLayer(2)` → `charmLayer` → source layers 2:end. CHARM is new and random at target time. |
| `charm3Conv` | The paper's CHARM-base mconv: three conv layers with 32 feature maps, filter 8, stride 1 (Sec. 2.3.1). The paper doesn't name the activations, so this uses ReLU after conv 1 and conv 2. Conv 3 is linear, then the max-pool. Everything else is as in `charmBase`. |
| `charmInstNorm` | `charmBase` with `instanceNormalizationLayer` between the input and CHARM. Each channel of each trial is normalised over time and has a learnable scale and offset (Sec. 2.3: "we first apply an InstanceNormalization, which normalizes each channel independently"). The paper doesn't say whether the scale and offset are learnable. This variant follows the usual InstanceNorm default and learns them. |
| `charmSourceTrained` | The paper's transfer protocol, adapted. See below. |

Not done here: CHARM-CKV and CHARM-CQ (attention over keys and values), the L1
penalty on `p`, and CMSAugment. An L1 penalty on `p` is constant because each row of
`p` is a softmax, so it has no effect here.

## How the source-trained transfer works

**Paper (Sec. 2.3.2, 3.3).** CHARM is trained jointly with the classifier, a 1-D CNN,
on the source data: TUH Seizure, 21 channels. The classification head is then
dropped. The target, CHB-MIT, has 17 channels, which are zero-padded to the source
count. Two settings follow:

* **fixed**: the transferred layers, CHARM included, are frozen and only a new linear head is trained;
* **fine-tuned**: the whole network is trained on the target.

**Here.**

1. **Source run** (`cwl_N2_source-domain_DeepRNN_charm__charmSourceTrained`). The net is
   `sequenceInputLayer(61)` → CHARM-base (M = 61, d = 32) → the DeepRNN source net
   (LSTM 128, LSTM 64, ...). It is trained from scratch on the full-montage source
   folds, with the same options as the plain DeepRNN source (LR 0.0029, mini-batch
   197, 30 epochs). The models are saved to
   `pretrainedModels_charm/cwl/cwl1/DeepRNN/sourceDomainCHARMClassifier_N2_DeepRNN_<i>.mat`,
   so the plain source nets in `pretrainedModels/` are never touched.
2. **Transfer** (`cwl_N2_target-domain_DeepRNN_charm__charmSourceTrained`). CHARM's
   weights don't depend on the number of input channels: the conv filters run on
   each channel separately, and the canonical embeddings score each channel's key.
   So the source-trained CHARM can read the 2 target channels (Fp1, Fz) directly.
   Only the input layer is swapped, from 61 to 2 channels. CHARM still produces 61
   canonical channels for the DeepRNN. The classifier head is kept, because the
   classes are the same. Then the whole network is fine-tuned with the usual target
   options (15 epochs). This is the paper's fine-tuned setting.
   `freezeFunctionalLayers` is called as for every baseline, and it freezes nothing.

**Differences from the paper.**

* **No zero padding.** With zero-padded channels, CHARM-base would still give those
  channels softmax weight, but they would add nothing. The output would only shrink
  towards zero. Feeding the 2 real channels avoids that and is possible only because
  CHARM is channel-count independent.
* **No channel-drop augmentation.** The paper trains with CMSAugment (random channel
  shuffles and drops). Without it, the source-trained CHARM never sees a 2-channel
  input before the transfer. CMSAugment would be the next change to try if this row
  is weak.
* **The source net is trained again with CHARM in front.** This variant does not use
  the HPC or threefry source nets (it runs on one branch only). Its DeepRNN part
  starts from a fresh threefry init, which differs from the plain source run because
  CHARM's init draws random numbers first.

## Running

Use a scratch copy of the repository, with `datasets/` and `pretrainedModels/` laid out
as for `replication/sourceReplication/runTargetBranch.m`:

```matlab
addpath replication/charm
checkCharm                                          % layer and network checks, CPU, seconds
hpcCharmReference('L:\plstm\HPC\PhaseAwareLSTM\cvalresults\CWL\DeepRNN\cval_DeepRNN_target-domain_2_CHARM_251104_1613.mat', 'charm_hpc_reference.csv')
runCharmBranch(1)                                   % fold 1 of every variant
runCharmBranch(1:10, 'variants', {'charmBase'})     % 10 folds (resumes)
```

Or call `runReplication('ids', {'cwl_N2_target-domain_DeepRNN_charm'}, 'variant', 'charmBase', 'rngType', 'threefry')`
directly. Without a CHARM variant, the CHARM cells are reported as unsupported.

## Results

CWL C2, DeepRNN, Fp1-Fz target, threefry. Two source-net branches: the HPC nets
the paper's target runs started from, and our threefry nets (bit-identical to the
Nov 2025 Windows nets). The paper's CHARM cell is the HPC run: val 60.13, test 56.34.

10-fold mean (std) accuracy in %. "Mean abs diff" is the mean, over folds, of the absolute test difference from the HPC fold.

| Variant | Branch | Val | Test | Mean abs diff (test) vs HPC fold |
|---|---|---|---|---|
| Paper (HPC run) | | 60.13 | 56.34 | |
| `charmBase` | HPC nets | 59.79 (2.60) | 56.43 (1.77) | 1.47 |
| `charmBase` | threefry | 60.39 (4.08) | 56.27 (2.20) | 0.91 |
| `charm3Conv` | HPC nets | 59.51 (1.54) | 56.03 (1.16) | 0.58 |
| `charm3Conv` | threefry | 59.61 (1.49) | 55.88 (0.68) | 0.88 |
| `charmInstNorm` | HPC nets | 52.32 (0.93) | 51.33 (0.29) | 5.02 |
| `charmInstNorm` | threefry | 52.05 (0.71) | 51.34 (0.42) | 5.00 |
| `charmSourceTrained` | threefry | running | | |

* **The replication row reproduces.** `charmBase` lands within 0.1 points of the paper's test mean on both branches.
* **Three conv layers change nothing.** The fold-to-fold spread is smaller, but the means are the same.
* **Instance normalisation hurts, by about 5 points, close to chance.** It rescales Fp1 and Fz per trial before they reach the source DeepRNN, which was trained on raw-scale signals. The fine-tuning does not recover from that.
* **Source-trained, source stage (first 5 folds).** CHARM + DeepRNN validates at 67.6 on the 61-channel source, like the plain DeepRNN source (67-68). Its source test accuracy is only 51.2, against 60 for the plain source. So CHARM fits the training subjects' channels and does not generalise to the test subjects.
