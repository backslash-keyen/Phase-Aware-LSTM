# Phased LSTM projection baseline

The paper's phase-aware baseline tables (Tables 6 and 7) have a Phased-LSTM row: the Phased LSTM of Neil, Pfeiffer and Liu (NIPS 2016) used as the projection in front of the pretrained source network, like the FFNN, LSTM and GRU projections. The released code has no Phased LSTM. This folder adds it as opt-in variants, one per version, for the CWL 2-class DeepRNN cells (Fp1-Fz target).

## Where the paper's row comes from

The Nov 2025 project behind the paper (`L:\plstm`, copied read-only to `J:\Phase-Aware-LSTM-paperpipeline`) ran the row on the HPC with the run script `experiments/CWL/C2/baseline_phase/RNN/td1/target_phasedLSTM_cwl_RNN_C2.m` (layerName `phased-LSTM`). Its pieces are ported here without changes:

| Nov 2025 file | Here | md5 |
|---|---|---|
| `modelframe/architectures/layerfunctions/baselines/phasedLSTM.m` (1 Nov 2025) | `variants/phased/common/phasedLSTM.m` | `ffd200b79c5444bd789f30f2caef53f5` |
| `modelframe/modelfunctions/phasebaselines/get_pLayer_phasedLSTM.m` | `variants/phased/common/get_pLayer_phasedLSTM.m` | `3e06a7e3b5765eacfc88b530ec46a0ef` |
| the `phased-lstm` case of `modelframe/get_trainingOptions.m` | `variants/phased/common/get_trainingOptions.m` | (rewritten: same options) |

Options (the Nov 2025 routing): `get_trainingOptions_CWL_C3_DeepRNN` (Adam, LR 0.004687, mini-batch 234, gradient threshold 1.98, L2 0.0097, best-validation output), 15 target epochs, Shuffle `never`. Every version uses these options, so the versions differ only in the layer.

## How the Nov 2025 layer differs from Neil et al. 2016

| | Neil et al. 2016 | Nov 2025 `phasedLSTM.m` |
|---|---|---|
| Time gate (Eq. 6) | rises 2φ/r_on to 1 at r_on/2, falls 2 − 2φ/r_on to 0 at r_on, αφ when closed | rises φ/r_on to 0.1 at 0.1·r_on, jumps to 2.9, falls 3 − φ/r_on to 2 at r_on. So k > 1 on 90 % of the open phase, and the update c = k c̃ + (1 − k) c extrapolates |
| r_on | 0.05, fixed (learned only in their first task) | 0.01 (the constructor default overrides the property's 0.05) |
| Leak α | 1e-3 in training, 0 at test | 0 (`setTrainMode`/`setTestMode` exist but nothing calls them) |
| Period τ at init | exp(U(0, 3)) for the frequency tasks | exp(U(0, 1)), i.e. 1 to 2.7 samples (the comment says U(0, 3)) |
| τ and s learned | yes | declared learnable, but `computeTimeGate` takes `extractdata` of both, so their gradient is zero and only L2 weight decay moves them |
| Peepholes (Eqs. 1, 2, 4) | yes (Graves 2013 LSTM; Lasagne's LSTM, which Neil kept, has them on) | none |
| Everything else | | the same: unit forget bias, Glorot W, orthogonal U, time t = sample index, h_t and c_t held when the gate is closed (Eqs. 7 to 10) |

## Versions

Each version is a folder `replication/variants/phased/<name>` with a one-line `phasedSettings.m`; the settings live in `variants/phased/common/phasedVersions.m`. Every version except `phasedBase` uses `phasedLSTMCfg`, which has every difference above as a setting and a hand-written backward pass. With the `phasedBase` settings it draws the same initial weights as `phasedLSTM.m` (same rand calls in the same order) and gives the same output and gradients; `checkPhased.m` tests this, and tests the backward pass of every version against autodiff.

| Version | Kind | What it is |
|---|---|---|
| `phasedBase` | replication | the Nov 2025 layer and builder, ported unchanged (autodiff) |
| `phasedGate` | correction | the time gate of Neil Eq. 6; everything else as phasedBase. The other corrections build on it, because with k up to 2.9 the other changes are hard to read |
| `phasedGateRon` | correction | phasedGate + r_on 0.05 |
| `phasedGateLeak` | correction | phasedGate + leak 1e-3 in training, 0 at test |
| `phasedGateTau` | correction | phasedGate + τ ~ exp(U(0, 3)) |
| `phasedGateLearn` | correction | phasedGate + gradients reach τ and s |
| `phasedGateNoDecay` | correction | phasedGate + no L2 decay on τ and s (they stay at init) |
| `phasedGatePeep` | correction | phasedGate + peepholes |
| `phasedNeil` | faithful | all corrections: Neil et al. 2016 as written |
| `phasedIdeal` | ideal | phasedNeil + τ log-uniform on [40, 500] samples + r_on learned per unit |
| `phasedIdealWide` | ideal | phasedIdeal with r_on starting at 0.5 |

Per-parameter factors (builder `get_pLayer_phasedVersion.m`): τ and s keep learn-rate 1 and get the L2 factor of the version (1 as in Nov 2025, 0 in phasedGateNoDecay, phasedNeil and the ideal versions). r_on has learn-rate 0 unless it is learned. Peepholes have learn-rate 0 and L2 0 unless they are on. A learned τ is kept at 0.5 samples or more, and a learned r_on in [0.001, 1].

### Why the ideal version looks like this

The CWL trials are 500 samples at 250 Hz (2 s), and the projection has to feed a source LSTM that was trained on EEG sampled at every step.

* **Gate, leak, peepholes, learned timing:** as Neil et al. The gate of Eq. 6 is what lets gradients pass through the open phase, the leak carries gradient through the closed phase, and Sec. 3 shows that τ, s and r_on can all be learned. Peepholes are part of their LSTM, and Gers, Schraudolph and Schmidhuber (JMLR 2002) introduced them for precise timing.
* **Periods 40 to 500 samples, log-uniform:** time here is the sample index, so with r_on = 0.05 a period below 20 samples has an open phase shorter than one sample, which the gate then hits only by chance. A 40-sample floor gives every open phase at least two samples. The ceiling is the trial length. Neil Fig. 4b shows that longer periods learn 500-step sequences faster, and the Clockwork RNN (Koutník et al. 2014) spreads its module periods exponentially (1 to 256 steps), which the log-uniform draw does too. In time, 40 to 500 samples is 0.16 to 2 s, so 0.5 to 6 Hz, the delta and theta range.
* **r_on learned:** Neil fix r_on at 0.05 for the compute saving, which does not matter here. Their first task learns it, so each unit can set its own duty cycle.
* **phasedIdealWide:** Chen, Liang and Wang (IJAIA 2019) compared LSTM and Phased LSTM on regularly sampled gait data and found that the time gate's skipped updates cost accuracy. Starting r_on at 0.5 lets each unit update for half of its cycle and leaves it to training to make the gate sparser.

## Running

From the root of a run tree (`replication/` and `utils/` from this branch, `datasets/` and `pretrainedModels/` linked):

```matlab
addpath('replication'); addpath(fullfile('replication','phased'));
checkPhased('pretrainedModels/cwl/cwl1/DeepRNN/sourceDomainClassifier_N2_DeepRNN_1.mat')
runPhasedBranch(1)                                  % fold 1, every version
runPhasedBranch(1:10, 'variants', {'phasedBase'})   % 10 folds of one version (resumes)
phasedSummary([], 'phased_hpc_reference.csv')
```

`runPhasedBranch` calls `runReplication('ids', {'cwl_N2_target-domain_DeepRNN_phased-lstm'}, 'variant', <version>, 'rngType', 'threefry')`. The two branches are the two sets of source nets, as for the other target baselines (`replication/sourceReplication/README.md`): `hpc` (the source nets the paper's target runs started from) and `ours` (the threefry retrain, bit-identical to the Nov 2025 Windows nets). `hpcPhasedReference` reads the paper's own per-fold results from the HPC cval file on `L:`.

## Results

CWL C2 DeepRNN, Fp1-Fz target, test accuracy (%), mean (SD) over folds. Filled in as the runs finish.

| Version | HPC source nets, fold 1 | threefry source nets, fold 1 | HPC source nets, 10 folds | threefry source nets, 10 folds |
|---|---|---|---|---|
| `phasedBase` | | | | |
| `phasedGate` | | | | |
| `phasedGateRon` | | | | |
| `phasedGateLeak` | | | | |
| `phasedGateTau` | | | | |
| `phasedGateLearn` | | | | |
| `phasedGateNoDecay` | | | | |
| `phasedGatePeep` | | | | |
| `phasedNeil` | | | | |
| `phasedIdeal` | | | | |
| `phasedIdealWide` | | | | |
