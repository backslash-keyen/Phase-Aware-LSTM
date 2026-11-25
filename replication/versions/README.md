# Phase-aware LSTM layer versions

Every phase-aware LSTM layer found in Keyen's folders, each with a fast
(fused GPU kernel) port that computes the same thing. The originals stay where
they are and are never changed; a verbatim copy of each is in `originals/`.
The released layer (`utils/.../phaseAwareLSTM.m`) is untouched.

- `phaseVersionList.m`: the registry below, as a MATLAB struct array.
- `originals/<nn>_<id>/`: verbatim copies (reference only, never on the path).
- `fast/<id>/phaseAwareLSTM_<id>_fast.m`: one small class per version. It
  holds the original's learnables, initialisation and a `config()` that names
  what makes the version different (angle map, clip, eps floors, carry, sign,
  fed-back phase). The arithmetic lives in one shared engine,
  `fast/common/phaseVersionEngine.m`: one fused kernel per time step forward,
  one backward, hand-written gradients.
- `checkPhaseVersions.m`: compares each fast port with its original (short, trains nothing).
- `runPhaseVersions.m`: trains chosen versions on chosen folds, one at a time.

Dedicated ports that already existed are reused, not duplicated: notrig is
`variants/notrigFastPhaseLSTM`, ver5 is `variants/ver5FastPhaseLSTM`,
halfAng (the released model) is `variants/fusedPhaseLSTM` from PR 2.

## How to run (on the PC, from the repository root)

```matlab
addpath('replication/versions');
checkPhaseVersions();                              % a few minutes; expect fwd and grad < 1e-6
runPhaseVersions();                                % notrig and ver5, fold 2
runPhaseVersions('ids', 'all', 'folds', 2);        % every ported version, fold 2 screen
runPhaseVersions('ids', {'notrig'}, 'folds', 1:10);
```

Each run is `cwl_N2_target-domain_DeepRNN_phase-lstm` (Fp1/Fz, our own source
models) through `runReplication` with variant `pv_<id>` (or the dedicated
variant for notrig/ver5) and the tuned phase-aware options
(`get_trainingOptions_CWL_C2_PhaseAware`, 20 epochs, no shuffle,
best-validation). Results are stored like every other variant
(`<id>__pv_<version>`) and a rerun resumes from the next unsaved fold.
Jobs run one at a time; before each one the runner waits for 0.5 GB free RAM.
A single version is also `runReplication('ids', {...}, 'folds', 2, 'variant', 'pv_ver1')`.

## Registry

Numbered oldest first. "unit-forget" = the usual LSTM forget-gate bias layout
(1 on the forget rows, 0 elsewhere); b2 is the phase bias, b1 the gate bias.

| # | id | date | parent | change versus parent | status |
|---|---|---|---|---|---|
| 01 | `cstmArch1Mar25` | 2025-03-31 | - | first Euler-LSTM: theta = max(atan(P),eps); every gate, the candidate, h and the fed-back phase floored at eps; signed output tanh(sign(Re+Im)\|c\|); fed-back phase = o_phase.*phase; memory NOT carried between steps; phase bias b2 = unit-forget x pi/2 | fast |
| 02 | `cstmArch2Apr25` | 2025-04-01 | cstmArch1Mar25 | polar memory carried between steps (\|c\|, atan2(Im, Re+eps)); candidate in polar form \|tanh g\| e^{i atan p} | fast |
| 03 | `cstmArch1Jun25` | 2025-06-26 | cstmArch1Mar25 | same as cstmArch1Mar25 except b2 = unit-forget x 2*pi | fast |
| 04 | `pal12Oct25` | 2025-10-12 | cstmArch1Mar25 | same as cstmArch1Mar25 except b2 = unit-forget x pi | fast |
| 05 | `ver1` | 2025-10-12 | pal12Oct25 | eps floors removed except on theta (max(atan,eps)) | fast |
| 06 | `ver2` | 2025-10-12 | ver1 | fed-back phase = phase (no o_phase factor) | fast |
| 07 | `ver3` | 2025-10-12 | ver2 | output sign = sign(Re*Im) | fast |
| 08 | `ver4` | 2025-10-12 | ver2 | unsigned output h = o.*tanh(\|c\|) | fast |
| 09 | `ver6` | 2025-10-12 | ver2 | theta = atan(P+eps) instead of max(atan(P),eps) | fast |
| 10 | `tanh` | 2025-10-12 | ver1 | theta = pi*tanh(P); b2 = unit-forget x 1 | fast |
| 11 | `sigmoid` | 2025-10-12 | ver1 | theta = pi*sigmoid(P); b2 = unit-forget x 1 | fast |
| 12 | `rotAngle` | 2025-10-12 | ver1 | theta = P (direct angle); b2 = unit-forget x 1 | fast |
| 13 | `atan2` | 2025-10-12 | ver1 | theta = atan2(Qx, Wp) (atan2 of the recurrent and input parts) | not ported: typo hiddenstate (lower case) means h is never updated, so the layer always outputs zeros; atan2v1 is the fixed version |
| 14 | `atan2v1` | 2025-10-12 | atan2 | atan2 fixed: h updated, signed and floored at eps | fast |
| 15 | `atan2v2` | 2025-10-12 | atan2v1 | theta = atan2(Wp+Qx, W3x+b3+Qx) (learned phantom denominator) | fast |
| 16 | `atan2v3` | 2025-10-12 | atan2v1 | theta = atan2(Wp, Qx) (arguments swapped) | fast |
| 17 | `atan2v4` | 2025-10-20 | atan2v1 | eps floors removed; fed-back phase = phase | fast |
| 18 | `atan2v5` | 2025-10-20 | atan2v4 | unsigned output | fast |
| 19 | `pal18Oct25` | 2025-10-18 | pal12Oct25 | memory carried between steps (the no-carry bug fixed) | fast |
| 20 | `ver5` | 2025-10-23 | pal18Oct25 | no eps floors; theta = atan(P); imaginary candidate tanh(P_c); unsigned output; fed-back phase = phase | fast |
| 21 | `ver7` | 2025-10-23 | ver5 | theta = max(atan(P),eps); debug printing | fast |
| 22 | `ver10M` | 2025-10-23 | ver5 | fed-back phase = phase + atan(P_o) | fast |
| 23 | `notrigOct25` | 2025-10-23 | ver10M | cos/sin of atan written without trig (1/sqrt(1+P^2)+eps); fed-back phase = phase + sigmoid(P_o) | fast |
| 24 | `phasebackup` | 2025-10-23 | ver5 | scratch rewrite with sanitize(), gate normalisation and safe unit vectors | not ported: not a runnable layer (file phasebackup.m declares class phaseAwareLSTM_ver5; the recurrence overwrites the real memory before the imaginary update uses it, and the fed-back phase is never updated) |
| 25 | `notrig` | 2025-11-03 | ver5 | clip +-5 on all pre-activations; b2 = unit-forget x 1; fed-back phase = phase; \|c\| floored at 1e-12 | fast |
| 26 | `halfAng` | 2025-11-06 | notrig | theta via tan-half-angle t = tanh(P/2); clip +-20; b1 zeroed from row 2H; b2 = 0 | fast |
| 27 | `halfAngV2` | 2025-11-06 | halfAng | tanh(0.6 P) instead of tanh(P/2); new init (glorot-uniform, block-orthogonal, W2 and Q x 0.5, b1 forget +1 on rows 1:H, input -0.5) | fast |

Not ported:

- **atan2** (13, 2025-10-12): typo hiddenstate (lower case) means h is never updated, so the layer always outputs zeros; atan2v1 is the fixed version
- **phasebackup** (24, 2025-10-23): not a runnable layer (file phasebackup.m declares class phaseAwareLSTM_ver5; the recurrence overwrites the real memory before the imaginary update uses it, and the fed-back phase is never updated)

Duplicates (same computation as a listed version, not copied):

| file | why it is not listed separately |
|---|---|
| ver8, ver9 (M:, 23 Oct 2025) | ver5 plus debug printing (ver8 and ver9 are identical files) |
| ver10 (L:\plstm, 28 Oct 2025) | same computation as ver5 |
| phaseAwareLSTM_noTrig (M:\Earth\helperLibrary, 29 Oct 2025) | same computation as notrig (3 Nov); only an unused Op line and the class name differ |
| M:\Earth\toolbox\+phaseLSTM (13 Mar 2026) | repackaging, no new model: PhaseAwareLSTM = pal18Oct25, _ver1/_ver5/_ver10/_atan2/_noTrig = the Oct 2025 files, _halfAngle = halfAngV2 (init helpers renamed) |
| phaseAwareLSTM_halfAng (released, utils/modelstuff/layerdefinitions) | same as halfAng except |c| floor 1e-9 instead of 1e-12 |

Where each original lives (read-only):

| id | origin (read-only) |
|---|---|
| `cstmArch1Mar25` | `L:\PhysciaScriptaCodes_Ver1\OtherCodesDifferentDirectory\PhaseAwareLSTMs\ProjectionTransferLearning\helperFunctions\modelFunctions\cstm_arch1.m` |
| `cstmArch2Apr25` | `L:\PhysciaScriptaCodes_Ver1\OtherCodesDifferentDirectory\PhaseAwareLSTMs\ProjectionTransferLearning\helperFunctions\cstm_arch2.m` |
| `cstmArch1Jun25` | `L:\PhysciaScriptaCodes_Ver1\OtherCodesDifferentDirectory\PhaseAwareLSTMs\ProjectionTransferLearning\helperFunctions\cstm_arch1.m` |
| `pal12Oct25` | `L:\plstm\helperLibrary\modelframe\architectures\layerfunctions\phaseAwareLSTM.m` |
| `ver1` | `L:\plstm\helperLibrary\modelframe\architectures\layerfunctions\phaseAwareLSTMs_versions\phaseAwareLSTM_ver1.m` |
| `ver2` | `L:\plstm\helperLibrary\modelframe\architectures\layerfunctions\phaseAwareLSTMs_versions\phaseAwareLSTM_ver2.m` |
| `ver3` | `L:\plstm\helperLibrary\modelframe\architectures\layerfunctions\phaseAwareLSTMs_versions\phaseAwareLSTM_ver3.m` |
| `ver4` | `L:\plstm\helperLibrary\modelframe\architectures\layerfunctions\phaseAwareLSTMs_versions\phaseAwareLSTM_ver4.m` |
| `ver6` | `L:\plstm\helperLibrary\modelframe\architectures\layerfunctions\phaseAwareLSTMs_versions\phaseAwareLSTM_ver6.m` |
| `tanh` | `L:\plstm\helperLibrary\modelframe\architectures\layerfunctions\phaseAwareLSTMs_versions\phaseAwareLSTM_tanh.m` |
| `sigmoid` | `L:\plstm\helperLibrary\modelframe\architectures\layerfunctions\phaseAwareLSTMs_versions\phaseAwareLSTM_sigmoid.m` |
| `rotAngle` | `L:\plstm\helperLibrary\modelframe\architectures\layerfunctions\phaseAwareLSTMs_versions\phaseAwareLSTM_directRotAngle.m` |
| `atan2` | `L:\plstm\helperLibrary\modelframe\architectures\layerfunctions\phaseAwareLSTMs_versions\phaseAwareLSTM_atan2.m` |
| `atan2v1` | `L:\plstm\helperLibrary\modelframe\architectures\layerfunctions\phaseAwareLSTMs_versions\phaseAwareLSTM_atan2_ver1.m` |
| `atan2v2` | `L:\plstm\helperLibrary\modelframe\architectures\layerfunctions\phaseAwareLSTMs_versions\phaseAwareLSTM_atan2_ver2.m` |
| `atan2v3` | `L:\plstm\helperLibrary\modelframe\architectures\layerfunctions\phaseAwareLSTMs_versions\phaseAwareLSTM_atan2_ver3.m` |
| `atan2v4` | `L:\plstm\helperLibrary\modelframe\architectures\layerfunctions\phaseAwareLSTMs_versions\phaseAwareLSTM_atan2_ver4.m` |
| `atan2v5` | `L:\plstm\helperLibrary\modelframe\architectures\layerfunctions\phaseAwareLSTMs_versions\phaseAwareLSTM_atan2_ver5.m` |
| `pal18Oct25` | `M:\Earth\helperLibrary\modelframe\architectures\layerfunctions\phaseAwareLSTM.m` |
| `ver5` | `L:\plstm\helperLibrary\modelframe\architectures\layerfunctions\phaseAwareLSTMs_versions\phaseAwareLSTM_ver5.m` |
| `ver7` | `M:\Earth\helperLibrary\modelframe\architectures\layerfunctions\phaseAwareLSTMs_versions\phaseAwareLSTM_ver7.m` |
| `ver10M` | `M:\Earth\helperLibrary\modelframe\architectures\layerfunctions\phaseAwareLSTMs_versions\phaseAwareLSTM_ver10.m` |
| `notrigOct25` | `M:\Earth\archive\phaseAwareLSTM_notrig.m` |
| `phasebackup` | `M:\Earth\archive\phasebackup.m` |
| `notrig` | `L:\plstm\helperLibrary\modelframe\phaseAwareLSTM_notrig.m` |
| `halfAng` | `L:\plstm\helperLibrary\modelframe\phaseAwareLSTM_halfAng.m` |
| `halfAngV2` | `M:\Earth\archive\phaseAwareLSTM_halfAng_v2.m` |

## Verification

Offline (Octave, double precision, H 3, C 2, B 2, T 6), each fast port against
its verbatim original: forward within 1e-9 relative for all 25 versions (most
about 1e-16); the hand-written gradients match finite differences of the
original to about 1e-9 (tanh 6e-7 and rotAngle 8e-8, where the recurrence is
very sensitive at large weight scale). halfAngV2 was checked at batch size 1
because the original only runs there.

On the PC (MATLAB R2025b, GTX 1650, 2026-11-25), `checkPhaseVersions`: the
output Y matches the original for all 28 cases (most about 1e-14, tanh 3e-8,
notrigOct25 3e-9, halfAng 4e-7, halfAngV2 9e-6), the gradients to 1e-11 or
better except tanh 5e-7, notrigOct25 3e-8, halfAng 2e-6 and halfAngV2 5e-5, and
trainNetwork runs for every one. One forward+backward at H 30, B 197, T 500
takes about 1.5 to 2.8 s with the shared engine and about 0.6 s with the
dedicated ports (notrigFastPhaseLSTM, ver5FastPhaseLSTM, fusedPhaseLSTM).

## Comparison

Fold-2 screen results go here as they come in (test accuracy, tuned options,
our own source models). Earlier 10-fold results with PR 2 variants
(2026-11-25, test): matched half-angle 61.2 ± 2.6, released + π on the input
rotation 60.8 ± 2.3, LSTM projection 57.4 ± 3.7, released with default options
51.8 ± 3.4 (7 of 10 folds collapsed).

| version | fold 2 val | fold 2 test | min/fold | note |
|---|---|---|---|---|
| notrig | 70.6 | 61.6 | 6.1 | notrigFastPhaseLSTM |
| ver5 | 68.7 | 61.2 | 5.6 | ver5FastPhaseLSTM |

## Output-sign study (2026-11-25)

Not original versions: four layers that differ from `piInputPhaseLSTM`
(released layer + b2 = π on the input rotation, tuned options) **only in
the output h**. Folders `fast/piSign_<mode>/`, variants `pv_piSign_<mode>`.

| mode | h | note |
|---|---|---|
| `none` | o·tanh(\|c\|) | released and paper (eq. 7: only the magnitude goes downstream); control, should match piInputPhaseLSTM |
| `sum` | o·tanh(sign(Re+Im)·\|c\|) | the sign rule of the early versions |
| `realProj` | o·tanh(\|c\|)·cos φ | real projection of the polar memory |
| `tanhRe` | o·tanh(Re c) | tanh of the real part |

Not included: a 2H output [real; imaginary]. The projection's output width
must equal the source model's input width, so doubling it breaks the transfer.

Fold 2 (2026-11-25, tuned options, our source models), val / test, about 13 min each:
none 65.9 / 60.0, sum 67.2 / 61.2, realProj 70.0 / 64.0, tanhRe 68.3 / 64.1, none collapsed.
The control (none) is the same model as piInputPhaseLSTM, which gave 62.4 test on
fold 2, so differences of about 2 points between implementations are rounding
drift over T = 500. One fold does not settle a 4-point gain.

| version (cont.) | fold 2 val | fold 2 test | min/fold | note |
|---|---|---|---|---|
| cstmArch1Mar25 | 67.7 | 62.5 | 13.5 | pv_ |

```matlab
for m = {'none','sum','realProj','tanhRe'}
    runReplication('ids', {'cwl_N2_target-domain_DeepRNN_phase-lstm'}, 'folds', 2, 'variant', ['pv_piSign_' m{1}]);
end
```
