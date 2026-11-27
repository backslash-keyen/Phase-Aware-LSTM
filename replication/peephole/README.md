# Peephole LSTM projection

A projection baseline that is not in the paper: the LSTM projection row (CWL C2, DeepRNN, Fp1–Fz target) with its `lstmLayer` replaced by an LSTM with peephole connections. Everything else is the LSTM row: the same builder (`get_pLayer_LSTM.m`, 61 hidden units = source input size), the same frozen source layers and the same training options (`get_trainingOptions_DeepRNN_CWL`, 15 target epochs). So each version is compared with the LSTM projection fold by fold.

Each version is an opt-in variant `ph_<version>` (`replication/private/variantInfo.m`). It only touches target runs with layer `lstm` and is stored as `cwl_N2_target-domain_DeepRNN_lstm__ph_<version>`.

```matlab
addpath('replication'); addpath(fullfile('replication','peephole'));
checkPeephole();                                  % correctness + speed, a few minutes
runPeepholeBranch(1);                             % fold 1, every version
runPeepholeBranch(1:10, 'versions', {'gers2002'}) % 10 folds of one version (resumes)
peepholeSummary({}, lstmFoldDir, 'hpc_reference.csv')
```

## Source

Gers, Schraudolph & Schmidhuber (2002), *Learning precise timing with LSTM recurrent networks*, JMLR 3:115–143 (`PeepholeLSTM-original.pdf` in Keyen's literature folder). Peepholes were first described by Gers & Schmidhuber (2000, IJCNN); the JMLR paper gives the full forward and backward pass and the experimental settings, so it is the reference here.

## Layer

`common/peepholeLSTMLayer.m`, one configurable layer. Rows are in `lstmLayer` order [i; f; z; o]:

* a_i = W_i x_t + R_i h_{t−1} + p_i ⊙ c_{t−1} + b_i, and a_f in the same way (eqs. 19–20)
* z = g(W_z x_t + R_z h_{t−1} + b_z), c_t = f ⊙ c_{t−1} + i ⊙ z (eqs. 1, 4)
* a_o = W_o x_t + R_o h_{t−1} + p_o ⊙ c_t + b_o: the output gate is updated after the cell, so it sees c_t (eq. 21, "update order")
* h_t = o ⊙ h(c_t) (eq. 5)

One cell per memory block, as in all the paper's experiments, so the peepholes are diagonal (3 weights per cell). States start at zero for every trial. Forward and backward are hand-written and vectorised per time step (CPU or GPU).

## Versions (`common/peepholeVersions.m`)

Each correction or variant is `gers2002` with exactly one field changed.

| version | group | what it is |
|---|---|---|
| `gers2002` | faithful | the paper's peephole LSTM: truncated gradient (Sec. 2.2, 3.2: error reaches the past only through the CEC, never through R h_{t−1} or back through the peepholes), g = identity and no output squashing (Sec. 4.1, eq. 5), all weights U[−0.1, 0.1], gate biases i 0 / f −2 / o +2 (Sec. 4.1) |
| `fullGrad` | correction | exact BPTT gradient. The truncation was an efficiency device of the 2002 online learner; full gradients are the standard since Graves & Schmidhuber (2005) |
| `squashIn` | correction | g = tanh. The identity g was chosen for the paper's timing tasks; tanh is the LSTM cell input of Hochreiter & Schmidhuber (1997) and Greff et al. (2017) |
| `squashOut` | correction | h = tanh. Gers et al. removed it "for lack of empirical evidence"; Greff et al. (2017) found removing the output activation hurts |
| `forgetPos` | correction | forget bias +1 instead of −2. The −2 makes the cell forget by default, suited to their reset-heavy timing tasks; +1 (remember by default) is from Gers et al. (2000) and Jozefowicz et al. (2015) |
| `glorotInit` | correction | `lstmLayer` initialisation (Glorot W, orthogonal R, unit-forget b, peepholes 0) instead of U[−0.1, 0.1] |
| `outPeepPrev` | variant | the output gate peeps c_{t−1} (one-phase update), testing the paper's update-order point |
| `noPeep` | variant | peepholes off: the paper's traditional LSTM (its Sec. 2) under the same training rule, so `gers2002 − noPeep` is the effect of the peepholes alone |
| `ideal` | ideal | the vanilla LSTM of Greff et al. (2017): peepholes with the paper's update order, full BPTT, g = h = tanh, forget bias +1, Glorot/orthogonal init, peepholes starting at 0 (so it starts as an LSTM and departs only if the peepholes help) |
| `idealCIFG` | ideal | `ideal` with the coupled input–forget gate i = 1 − f, which Greff et al. found costs nothing and removes a quarter of the gate weights |

Not adopted from the paper: its training rule (online SGD with momentum, learning rate 1e−5, weight updates after every target). The projection rows all share the LSTM row's Adam options, so the comparison is about the layer.

Not in the literature folder (cited from the literature, PDFs not checked): Gers & Schmidhuber 2000, Gers, Schmidhuber & Cummins 2000, Graves & Schmidhuber 2005, Jozefowicz et al. 2015, Greff et al. 2017.

## Checks (`checkPeephole.m`, 27 Nov 2026, R2025b)

* All 10 versions, sequence and last output: output and every gradient equal an independent plain-dlarray implementation differentiated by `dlgradient` (truncated versions: with the graph cut where the paper truncates) to 1e−15.
* `ideal` with peepholes off and `lstmLayer`'s weights equals `lstmLayer` (output and gradients, to single precision: the built-in layer rounds internally).
* Initial values of every version; GPU single vs CPU double within 2e−7.
* Every version builds inside the real LSTM projection (61 hidden units, fold-1 source net) and trains under `trainNetwork`.
* `glorot` does not draw the same W and R as `lstmLayer` from the same seed, so `ideal` and the LSTM row start from different random weights.

## Runs

Two branches, `rngType` threefry, as for the other baselines: source nets of the paper's HPC target runs (`pretrainedModels_hpc`) and our threefry nets (`_srcrepl`). Fold 1 for every version, then 10 folds. Results go here when they finish.
