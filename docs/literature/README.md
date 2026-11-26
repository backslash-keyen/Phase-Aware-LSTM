# Literature on the CWL dataset

Published results from other papers that use the same CWL data, kept for comparison. **Every number in this folder comes from the cited papers. None of them are results from this repository.**

## The dataset

CWL (NEC21 in this repo) is the passive BCI competition dataset of the **Neuroergonomics Conference 2021**:

> Hinss MF, Darmet L, Somon B, Jahanpour E, Lotte F, Ladouce S, Roy RN. *An EEG dataset for cross-session mental workload estimation: Passive BCI competition of the Neuroergonomics Conference 2021.* Zenodo, 2021. doi:10.5281/zenodo.5055046

It has 15 subjects and 3 sessions recorded one week apart. The task is MATB-II at 3 workload levels. It has 62 channels recorded at 500 Hz, released at 250 Hz in 2-s epochs, with 61 channels after preprocessing.

## Main comparison: Zhou et al., IEEE TIM 2023

> Zhou Y, Wang P, Gong P, Wei F, Wen X, Wu X. *Cross-Subject Cognitive Workload Recognition Based on EEG and Deep Domain Adaptation.* IEEE Transactions on Instrumentation and Measurement, 72:2518912, 2023. doi:10.1109/TIM.2023.3276515

Their Tables VII (session S1) and VIII (session S2) are transcribed in [`zhou2023_tim_matb_crosssubject.csv`](zhou2023_tim_matb_crosssubject.csv).

| Setup | Zhou et al. 2023 | This repo (CWL C2 source) |
|---|---|---|
| Classes | Binary, low vs high | Binary |
| Channels | 61 | 61 |
| Sessions | S1 and S2, each analysed separately | Pooled into folds |
| Split | Cross-subject, leave-one-subject-out (15 folds) | Pre-built train/val/test folds |
| Target labels | Unlabeled target subject (unsupervised domain adaptation) | Labeled training data |
| Best accuracy | DDA 75.64 ± 8.84 (S1), 74.07 ± 7.81 (S2) | See `replication/results` |
| Plain LSTM baseline | 54.68 ± 4.00 (S1), 56.06 ± 5.77 (S2), 2 LSTM layers | — |

**Protocol difference.** The data and classes match our source model. The split does not: Zhou et al. hold out whole subjects, one session at a time. Their numbers are a reference range for the 61-channel source task, not a like-for-like benchmark. Our 2-channel target results have no counterpart in their paper.

Transcription note: in Table VIII (S2), ShallowCNN's SEN/SPE/F1 standard deviations are identical to EEGNet's in the published table. They are copied as printed.

## Other papers on the same data

| Paper | Venue | Classes | Protocol | Reported accuracy | Use for us |
|---|---|---|---|---|---|
| Roy RN, Hinss MF, Darmet L, et al. *Retrospective on the First Passive Brain-Computer Interface Competition on Cross-Session Workload Estimation* | Frontiers in Neuroergonomics 3, 2022. doi:10.3389/fnrgo.2022.838342 | 3 | Cross-session: train on S1+S2, test on S3 | Winner 54.26% (Riemannian + electrode selection). Chance 33%, significance bound ~38%. Deep learning entries at chance | Dataset description and organisers' paper; benchmark only if the 3-class work resumes |
| Sun W, Li J. *AdaptEEG: A Deep Subdomain Adaptation Network with Class Confusion Loss for Cross-Subject Mental Workload Classification* | IEEE J. Biomedical and Health Informatics 29(3):1940–1949, 2025. doi:10.1109/JBHI.2024.3513038 | 3 | Cross-subject LOSO, S1 and S2 separately | 65.82 ± 10.52 (S1), 66.13 ± 10.96 (S2) | 3-class only. The paper calls the data "MATB, 15 subjects" without naming the source |
| Sun W, Li J. *Deep Subdomain Adaptation Network Improves Cross-Subject Mental Workload Classification* | Int. Conf. on Cyberworlds 2024 (IEEE), pp. 138–143. doi:10.1109/CW64301.2024.00049 | 3 | Cross-subject LOSO | Conformer+SA 65.03 (DSAN value in a figure only) | 3-class only |

Checked and excluded because they use other data: Guan et al., TNSRE 2023 (cross-task); Zhou et al., TNSRE 2022 (cross-task); Wang et al., SCVCNet (STEW/EEGMAT/N-back); Beiramvand et al., IEEE TIM 2024.

Compiled 2026-11-26. Zhou et al. was read in full text; the other entries were read from open-access copies.
