# Overview

## The problem

EEG decoders are usually trained on a full cap of electrodes. In use, a cheaper headset may record
only two channels. Retraining from scratch on two channels throws away what the full-cap model
learned.

## The idea

Train a **source network** on the full montage (61 channels for the CWL workload data). Then, for
the two-channel **target montage**, train a small **projection** that maps the two channels into
the input the source network expects, and fine-tune. The paper compares projections: a
feed-forward net, an LSTM, a GRU, CHARM, and its own **phase-aware LSTM**, a recurrent unit that
carries the phase of the input signal with its own parameters.

## What this repository does

It reruns every experiment of the paper with the released code, unchanged, and builds the paper's
tables from the fresh runs. Nothing is typed by hand, and the paper's own numbers are never stored:
the table builder knows each table's layout and which experiment fills each cell, nothing more.
Cells the released code can't produce stay blank with the reason.

Where the released code has a known fault, the fix is an **opt-in variant** with its own labels,
so the unchanged run stays the main replication.

Next: [The experiment](experiment.md).
