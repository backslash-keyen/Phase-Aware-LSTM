# The experiment

The authority for every fact on this page is [`CLAUDE.md`](../../../CLAUDE.md) (Domain
invariants); this page explains them.

## Datasets

- **CWL (NEC21):** mental workload from the Neuroergonomics Conference 2021 passive-BCI
  competition. 61 channels, 250 Hz, 2 s epochs. Binary (C2) and 3-class (C3) versions. Sources:
  [`docs/literature/README.md`](../../literature/README.md).
- **MI:** motor imagery. Source BCI IV 2a (22 channels), target BCI IV 2b (bipolar C3/C4).

## Folds

Each CWL fold splits the 13 training subjects at random (seeded by the fold index) into 7 source
and 6 target subjects. Subjects 14 and 15 are the test set in every fold, for source and target
alike, so test results are comparable across folds and no test subject is ever trained on.

Source and target never share a subject. A target layout cut from source folds would, and its
results would not measure transfer.

## Domains

| domain | data | montage |
|---|---|---|
| source | `fullchannels/` | full |
| target | `targetdomain1/` | Fp1-Fz (CWL), C3-C4 (MI) |
| target 2 | `targetdomain2/` | F3-F7 (CWL) |

The full list, with the tables each fills, is in the
[replication README](../../../replication/README.md).

## Run order

Source runs come first: each target run loads the source network its fold saved. The replication
uses MATLAB's threefry generator, the one the paper's runs used, so our source networks match the
authors' Windows ones bit for bit. The authors' Linux (HPC) networks can't be reproduced exactly,
so target results are reported twice: from the HPC networks and from ours.

Next: [Where the replication stands](status.md).
