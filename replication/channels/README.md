# Other CWL target channel layouts (extension study)

**Not part of the paper replication.** The replication keeps the paper's CWL target layouts, Fp1–Fz (`targetdomain1/`) and F3–F7 (`targetdomain2/`). Those are simply rows 1–2 and 3–4 of the NEC21 source montage. This folder tests two-channel layouts chosen from the MATB workload literature instead (frontal-midline theta plus parietal alpha, and MATB-specific posterior and anterior-frontal findings; see the project note `channels/matb-channel-literature.md`).

| Layout | Channels | Why |
|---|---|---|
| `FzPz` | Fz, Pz | classic workload pair, strongest evidence |
| `FzPOz` | Fz, POz | MATB effects largest posteriorly (Hinss 2023, Chu 2022) |
| `AF3AF4` | AF3, AF4 | anterior frontal, MATB-II (Chandra 2015); weak evidence |
| `FzP4` | Fz, P4 | fallback if POz is missing |

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
