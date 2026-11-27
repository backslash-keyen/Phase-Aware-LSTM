# Phase-aware LSTM: the paper's layer and tracked corrections

The paper's CWL phase-aware results come from `phaseAwareLSTM_notrig`
(`L:\plstm\helperLibrary\modelframe\phaseAwareLSTM_notrig.m`, 3 Nov 2025,
run on the HPC on 5 Nov 2025) trained with the tuned options
`get_trainingOptions_CWL_C2_PhaseAware` and 20 target epochs. This folder
keeps that layer as the replication version and adds every correction as its
own named version, so each change to the phase layer is tracked and can be
run on its own.

Setup of every run: CWL C2, DeepRNN source nets extracted from the paper's
HPC files (`pretrainedModels_hpc`, see `replication/sourceReplication`),
target domain Fp1/Fz, `rngType` `threefry` (the generator of the paper runs),
experiment `cwl_N2_target-domain_DeepRNN_phase-lstm`.

## Versions

| version | variant | parent | change |
|---|---|---|---|
| paper | `notrigPhaseLSTM` | - | the paper layer, copied unchanged (autodiff, original speed), and its options routing |
| port | `pc_port` | paper | `phaseAwareLSTMCorr` with the defaults: same model, hand-written backward (faster) |
| gradCap | `pc_gradCap` | port | learnable gradients capped at L2 1.25 in double before trainNetwork |
| forgetRow | `pc_forgetRow` | gradCap | unit-forget +1 of b1 on the forget gate f, not on the candidate gate i |
| noPhaseBias | `pc_noPhaseBias` | gradCap | phase bias b2 = 0 (paper: candidates start rotated by 45 deg) |
| cosSin | `pc_cosSin` | gradCap | phase gates see [cos; sin] of angle(c) instead of atan2 |
| complexH | `pc_complexH` | gradCap | phase gates see the complex hidden state o tanh(\|c\|) e^{i angle(c)} |
| lowRec | `pc_lowRec` | gradCap | U and Q drawn at half scale |

The registry is `common/phaseCorrVersions.m`; one configuration field per
correction (`gradCap`, `forgetRow`, `phaseBias`, `feedback`, `recScale`),
documented in `common/phaseAwareLSTMCorr.m`. A new correction is a new field
(default = paper behaviour), a row in the registry and a folder
`versions/<id>/get_pLayer_phaseLSTM.m`.

### Why gradCap comes first

On this PC the paper layer never trains. Its gradients reach about 1e21 to
1e30 at T = 500. trainNetwork clips each learnable to GradientThreshold 1.25
with the l2norm method in single precision: the squared norm overflows to
Inf, the clip factor 1.25/Inf is 0, and every update of the phase layer is
zero while the rest of the network trains. The paper's own HPC nets show the
same on folds 4, 6 and 10 (phase weights equal to the initialisation).
gradCap takes the norm in double and rescales first, which is exactly the clip
the options ask for. Every later version keeps gradCap, so each one tests a
single further change against it.

### What the later corrections address

- **forgetRow.** `initializeUnitForgetGate` puts +1 on rows H+1:2H. In this
  layer rows 1:H scale the old memory (the forget gate) and rows H+1:2H scale
  the new candidate, so the +1 lands on the input gate.
- **noPhaseBias.** b2 uses the same init, so theta_i starts at atan(1):
  every candidate enters the memory rotated by 45 degrees.
- **cosSin, complexH.** The fed-back phase atan2(Im c, Re c) has derivative
  1/|c|, which is unbounded when the memory nearly cancels, and jumps by 2 pi
  at the branch cut. cosSin removes the jump only; complexH feeds back the
  complex hidden state, whose derivative is bounded (tanh(r)/r is smooth at 0).
- **lowRec.** Recurrent gain over 500 steps.

## How to run (repository root of a run tree)

```matlab
addpath('replication'); addpath(fullfile('replication','phaseLayer'));
checkPhaseCorr();                         % about a minute, trains nothing
phaseCorrGradProbe('fold', 2);            % raw gradient norms vs T per version
runReplication('ids', {'cwl_N2_target-domain_DeepRNN_phase-lstm'}, 'folds', 2, ...
    'variant', 'pc_complexH', 'rngType', 'threefry');
comparePaperPhaseNets(fullfile('cvalresults','CWL','DeepRNN','folds', ...
    'cwl_N2_target-domain_DeepRNN_phase-lstm__pc_complexH'), paperFile);
```

`PHASECORR_MAXEPOCHS` overrides the 20 epochs for short checks.

## Checks

`checkPhaseCorr`: (1) `port` against the paper layer, same seed: identical
initial learnables, output and gradients equal in double; (2) every version
against an independent plain-dlarray implementation differentiated by
dlgradient (CPU double, GPU single); (3) initial biases and Q size; (4) the
cap keeps the direction, caps the norm and zeroes Inf/NaN.

## Results

Fold 2 first, then 10 folds. Filled in as runs finish.

Every version below runs on fold 2 and then on all 10 folds, the paper version included.

| version | fold 2 val | fold 2 test | phase layer trained | 10-fold test |
|---|---|---|---|---|
| paper (notrigPhaseLSTM) | | | | |
| port | | | | |
| gradCap | | | | |
| forgetRow | | | | |
| noPhaseBias | | | | |
| cosSin | | | | |
| complexH | | | | |
| lowRec | | | | |
| ideal | | | | |
| idealReadout | | | | |
| idealNoOmega | | | | |
| idealNoNorm | | | | |
| idealAtanRot | | | | |
