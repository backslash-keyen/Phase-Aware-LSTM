# Glossary

| term | meaning |
|---|---|
| source network | the decoder trained on the full montage (DeepRNN, EEGNet or Transformer) |
| projection | the small network that maps the target montage into the source network's input |
| phase-aware LSTM | the paper's recurrent projection that carries input phase with its own parameters |
| CWL, NEC21 | the cognitive-workload dataset from the Neuroergonomics Conference 2021 competition |
| C2, C3 | binary and 3-class versions of CWL |
| fold | one random 7 source / 6 target subject split; test is always subjects 14 and 15 |
| variant | an opt-in change to one run (e.g. `freezeIntended`), labelled `<id>__<variant>` |
| version | a named phase-aware layer (`replication/versions/`) |
| HPC branch, threefry branch | target results from the authors' Linux source nets, and from ours |
| paperwork set | `LOGBOOK.md`, `keyen.md`, `docs/sessions/`: committed straight to `main` after the work lands |
| handoff | a scope's `docs/sessions/<scope>.md`, rewritten every session |
