# Other CWL target channel layouts (extension study)

**Not part of the paper replication.** The replication keeps the paper's CWL target layouts, Fp1–Fz (`targetdomain1/`) and F3–F7 (`targetdomain2/`). Those are simply rows 1–2 and 3–4 of the NEC21 source montage. This folder tests two-channel layouts chosen from the MATB workload literature instead (frontal-midline theta plus parietal alpha, and MATB-specific posterior and anterior-frontal findings; see the project note `channels/matb-channel-literature.md`).

| Layout | Channels | Why |
|---|---|---|
| `FzPz` | Fz, Pz | classic workload pair, strongest evidence |
| `FzPOz` | Fz, POz | MATB effects largest posteriorly (Hinss 2023, Chu 2022) |
| `AF3AF4` | AF3, AF4 | anterior frontal, MATB-II (Chandra 2015); weak evidence |
| `FzP4` | Fz, P4 | fallback if POz is missing |
| `Fp1Fz` | Fp1, Fz | in-study reference: the paper's pair, cut from the same source folds |

## Steps (from the repo root, one MATLAB job at a time)

```matlab
addpath replication/channels
% 1. montage labels in source row order -> replication/channels/nec21_channels.txt
% 2. check only: targetdomain1 == rows {Fp1,Fz} (and targetdomain2 == {F3,F7}) of every source fold
buildTargetLayoutFolds('checkOnly', true)
% 3. build the folds (writes datasets/folds/CWL/NEC21_C2/targetlayout_<name>/, source folds untouched)
buildTargetLayoutFolds()
% 4. screen: LSTM projection and phase-aware projection (pi fix, tuned options) per layout
runTargetLayouts('folds', 2)
```

The new folds are the same trials, split and labels as the source folds; only the two channel rows differ. The check in step 2 refuses to write anything unless the existing paper target folds are exactly the `{Fp1, Fz}` rows of the source folds under the label file, which validates both the labels and the slicing.

Runs use the opt-in variant `tl_<layout>` (`replication/variants/targetLayout` shadows `perfold_dataset.m` for CWL `target-domain` only; `replication/variants/tl_<layout>/activeTargetLayout.m` names the layout). Everything else is the normal target-domain path: the pretrained DeepRNN source model of each fold, the options and the projection builders. Results are stored apart, as `<id>__tl_<layout>` (LSTM) and `<id>__piInputPhaseLSTM__tl_<layout>` (phase-aware), and the released files and replication defaults are unchanged.

The paper's `targetdomain1` folds (Oct 2025) turned out not to be exact slices of the `fullchannels` folds (Jul 2025): the check fails on fold 1. So the screen also builds `Fp1Fz` from the same source folds (`'check', false`), and every layout, including the reference, uses the source folds' own trials and split.

## Correction (2026-11-26): use the target subjects

The layouts above (`FzPz`, `FzPOz`, `AF3AF4`, `Fp1Fz`) were cut from the **source** folds, so the projection trained on the same subjects the source model was pretrained on. Their results are not valid for transfer. The real target folds come from `M:\Earth\datasets\helpers\data_perfoldExtraction_NEC21.m` / `dataFeeder_NEC21.m`: for fold i, `rng(i); p = randperm(13)`. Source uses train p(1:4) and val p(5:7). Target uses train p(8:11) and val p(12:13). Both test on subjects 14-15. Each subject is z-scored and winsorized per channel, downsampled 5x, and restricted to classes 0 vs 2.

The `*T` layouts (`tl_Fp1FzT`, `tl_FzPOzT`, `tl_AF3AF4T`) are built with that generator from the raw subject files, with `targetChannel` set to the layout's rows. `Fp1FzT` must reproduce `targetdomain1` exactly before any other layout is used. The reference for comparison is the existing Fp1/Fz target runs.
