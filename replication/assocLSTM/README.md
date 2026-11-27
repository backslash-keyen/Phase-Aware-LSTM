# Associative LSTM projection baseline

Not in the paper or in the released code. The Associative LSTM (Danihelka, Wayne, Uria,
Kalchbrenner and Graves, *Associative Long Short-Term Memory*, ICML 2016) is the published
design closest to the phase-aware layer: a complex-valued memory, real gates, and keys
that rotate what is stored and read. Here it sits in the CWL C2 DeepRNN Fp1-Fz target
table as one more projection in front of the pretrained source net, like the FFNN, LSTM
and GRU projections.

## The layer (`common/associativeLSTMLayer.m`)

N complex cells are held as Nh = 2N real values, h = [Re h; Im h] (Eq. 16). There are S
redundant copies of the memory. At each time step:

| Step | Equation | Paper |
|---|---|---|
| pre-activations | Z = W x_t + U h_{t-1} + b | Eqs. 17-18 |
| gates (real, one per complex cell, applied to Re and Im alike) | g_f, g_i, g_o = sigmoid(Z) | Eqs. 19-21 |
| keys and update | r_i = bound(r̂_i), r_o = bound(r̂_o), u = bound(û) | Eqs. 23-25 |
| memory, copy s | c_s = g_f ⊙ c_s + (P_s r_i) ⊛ (g_i ⊙ u) | Eqs. 26-28 |
| read-out | h = g_o ⊙ bound( (1/S) Σ_s (P_s r_o) ⊛ c_s ) | Eqs. 29-30 |

bound(z) = z / max(1, |z|) per complex number (Eq. 22). ⊛ is complex multiplication. The P_s
are constant random permutations of the N cells, the same for Re and Im. They are drawn
from their own threefry stream (seed 1), so the global RNG stream is untouched.

* The forward and backward passes are hand-written and vectorised per time step, on CPU or GPU.
* `checkAssocLSTM` compares them with `dlgradient` of a plain dlarray implementation. That implementation is written one copy at a time from the equations, and the comparison covers every version.
* It also checks Eq. 36: with the keys fixed at 1 and one copy, the memory follows the LSTM cell update.

## The projection (`common/get_pLayer_assocLSTM.m`)

The graph is the released `get_pLayer_LSTM`, with `lstmLayer(61)` replaced:

`sequenceInputLayer(2)` → `associativeLSTMLayer(N = 61)` → `fullyConnectedLayer(61)` readout → source layers 2:end

`freezeFunctionalLayers` is called as for every baseline, and it freezes nothing. Training uses
the same options as the FFNN / LSTM / GRU rows (`get_trainingOptions_DeepRNN_CWL`, 15
target epochs, gradient threshold 82.4).

## Choices the paper leaves open (version `paper`)

| Choice | Here | Why |
|---|---|---|
| Size | N = 61 complex cells (Nh = 122) | The source DeepRNN reads 61 channels. Nh = 2N must be even, so Nh = 61 is impossible. 61 complex cells match the phase-aware layer's 61. |
| Output to the source net | h (122 values) through a learned linear readout to 61 | The paper's networks always read h through learned output weights. `realOut` and `modOut` drop the readout. |
| Copies | 4 | The paper's main setting (Sec. 6.2.1: "4 copies were enough to provide a dramatic improvement"); 1 and 8 are versions. |
| W_hu | learned | Eq. 18 as written. The paper sets it to 0 on 4 of its 6 tasks; that is version `noWhu`. |
| Forget-gate bias | 0 | The paper biases only its LSTM baseline (Fig. 3: Associative LSTM "solved the task quickly without biasing the forget gate"). |
| Initialisation | W Glorot, U orthogonal, b 0 | Not given in the paper. These are the same initialisers as `lstmLayer` and the phase-aware layer. |
| Gradient clipping | as the other rows (82.4) | The paper used Adam without clipping; the options are kept equal across the projection rows instead. Gradient norms stay near 1e4, so there is no single-precision overflow (the phase layer's problem) and no cap is needed. |

## Versions

Each version is a runReplication variant, `al_<id>`, and writes its own fold files under
`<id>__al_<id>`. Each one changes one thing from its parent.

| Version | Parent | Change |
|---|---|---|
| `paper` | | Danihelka 2016 as written, with the choices above |
| `forgetBias1` | paper | forget bias +1 (Gers et al. 2000) |
| `noWhu` | paper | W_hu = 0: u is an embedding of x_t only |
| `copies1` | paper | 1 copy |
| `copies8` | paper | 8 copies |
| `unitKeys` | paper | keys normalised to modulus 1, instead of bound() (modulus ≤ 1) |
| `realOut` | paper | no readout: the 61 real parts Re h go to the source net |
| `modOut` | paper | no readout: the 61 moduli \|h\| go to the source net, as in the phase-aware layer |
| `ideal` | | literature-grounded version, below |
| `idealNoOmega` | ideal | no memory rotation |

## Ideal version

Each change from `paper` is grounded in the Associative LSTM paper itself or in the papers in Keyen's LSTM-Literature folder (unitary RNNs, LRU via RotRNN):

1. **Unit-modulus keys.** The paper's retrieval argument (Eq. 8) holds only if each key has
   modulus 1. Then the conjugate is the inverse and the cross terms are pure rotations that
   cancel across copies. bound() only caps the modulus at 1, so a small key stores and reads
   weakly. Unit-modulus keys follow Plate's complex HRRs and the unitary RNNs (Arjovsky
   et al. 2016).
2. **Forget bias +1.** Gers et al. (2000), and the paper's own LSTM baseline (Fig. 3). Over
   500 samples, a memory that starts half-forgotten at every step loses the start of the trial.
3. **Memory rotation at a learned frequency.** c_s = g_f ⊙ e^{iω} c_s + …, with one ω per cell,
   shared by the copies. This is the eigenvalue phase of the LRU and RotRNN, and it gives each
   cell an oscillation it can track. ω is initialised log-spaced over 1-40 Hz at 250 Hz, as in
   the ideal phase-aware layer, so the projection starts with a bank of EEG-band rotations. The
   permutations act on keys only, so a slot rotates the same way in every copy and retrieval
   still lines up.

The readout stays as in the paper, so the ideal version differs from `paper` only in these
three components. `idealNoOmega` removes the rotation.

## Running

Use a run tree laid out as for `replication/sourceReplication/runTargetBranch.m`
(`datasets/`, and `pretrainedModels/` holding the branch's source nets):

```matlab
addpath replication/assocLSTM
checkAssocLSTM                                   % gradient checks, CPU, seconds
smokeAssocLSTM({'paper'}, 1, 4)                  % network build + timing on fold 1
runAssocBranch(1)                                % fold 1 of every version
runAssocBranch(1:10, 'versions', {'paper'})      % 10 folds (resumes)
assocSummary
```

Or call `runReplication('ids', {'cwl_N2_target-domain_DeepRNN_assoc-lstm'}, 'variant', 'al_paper', 'rngType', 'threefry')`
directly. Without an `al_` variant, the cell is reported as unsupported.

## Results

CWL C2, DeepRNN, Fp1-Fz target, threefry. There are two source-net branches: the HPC nets
the paper's target runs started from, and our threefry nets. Pending.
