# Ideal complex LSTM layer: design draft

Draft for discussion (27 Nov 2026). It is grounded in the papers in Keyen's
LSTM-Literature folder; each choice names its source. Once agreed, the layer
becomes one more version family in this folder (`pc_ideal*`), with one
version per component so that each one is tested on its own.

## What the paper layer does differently from the literature

| aspect | paper layer (notrig) | literature |
|---|---|---|
| state fed back | magnitude h and the angle atan2(Im c, Re c), through separate real U and Q | the complex state itself, as [Re h; Im h] (Wolter & Yao 2018 eq. 7, Danihelka 2016 eq. 16) |
| derivative of the fed-back state | 1/\|c\|, unbounded; 2 pi jump at the branch cut | bounded: tanh(\|z\|) z/\|z\| (Hirose, in Wolter & Yao eq. 5) or bound(z) (Danihelka eq. 22). scuRNN (Maduranga 2019) shows the analogous singularity of modReLU at 0 |
| rotation | angle atan(P) in (-90, 90) deg from a clipped pre-activation, for both the memory and the candidate | unit-modulus factors: unitary matrices (Arjovsky 2016, scuRNN), input-driven unit-modulus keys (Danihelka), rotations (RUM, RotRNN) |
| gates | real sigmoids scaling the magnitude | the same (Wolter & Yao sec. 4.5, Danihelka eq. 19-21) |
| forget bias | +1 lands on the candidate gate | gates start open (Wolter & Yao: bias 4; Gers 2002: forget gate) |
| recurrent gain | orthogonal init, unconstrained afterwards | norm-preserving or bounded transition (Arjovsky, Wolter & Yao sec. 4.4, RotRNN normalisation). ceRNN (Shafran 2018) finds that allowing decay helps on real tasks |

## Proposed layer

Per unit j, complex memory c (stored as Re and Im), real input x:

1. Gates (real): f, i, o = sigmoid(W_g x + U_g [Re h; Im h] + b_g), with the
   +1 bias on f.
2. Candidate (complex, bounded): u = bound(W_u x + U_u [Re h; Im h] + b_u),
   with bound(z) = tanh(|z|) z/|z|. Smooth at 0, modulus < 1.
3. Rotation (unit modulus): theta = omega + pi tanh(W_r x + U_r [Re h; Im h] + b_r),
   rotation e^{i theta}.
   - omega is a learned per-unit frequency (radians per sample), like the
     eigenvalue phases of the LRU and RotRNN. Its init spreads over the EEG
     bands at the sampling rate.
   - The tanh term is an input-driven phase shift, like the Associative LSTM
     keys.
4. Memory: c_t = f e^{i theta} c_{t-1} + (1 - f) i u.
   - The eigenvalue of the transition has modulus f < 1, so the memory is
     stable and can decay (ceRNN).
   - The coupled input (1 - f) keeps |c| <= 1, like the GRU-style update of
     Wolter & Yao (eq. 10), in the spirit of the LRU and RotRNN normalisation.
     sqrt(1 - f^2), the LRU form, was dropped because its derivative is
     unbounded as f -> 1. An optional version uses plain i instead.
5. Hidden (complex, bounded): h = o bound(c).
   - Fed back: [Re h; Im h].
   - Output: |h|, as in the paper (Keyen, 27 Nov), giving H real channels
     into the source net. The paper layer also outputs only the magnitude.
     A learned readout W_o [Re h; Im h], as in Wolter & Yao eq. 7, is the
     version idealReadout.
6. Training: gradCap stays (overflow-safe clip). There is no atan2, no clip of
   the pre-activations and no 1/|c| anywhere, so gradients should stay finite
   in single precision. phaseCorrGradProbe will check this before any training.

Implemented as phaseAwareLSTMIdeal (common/); versions pc_ideal* below.

Optional later (one version each): orthogonal or Cayley-parameterised
U (Wolter & Yao sec. 4.4, scuRNN); several rotated copies of the memory
(Associative LSTM); a learned-frequency time gate (Phased LSTM).

## Versions to run (each one change from ideal)

| version | change |
|---|---|
| ideal | as above, output \|h\| |
| idealReadout | output W_o [Re h; Im h] |
| idealNoOmega | omega = 0: input-driven rotation only |
| idealNoNorm | memory input i u instead of (1 - f) i u |
| idealAtanRot | rotation from atan(P) as in the paper, to measure the angle map alone |

## Open questions for Keyen

- Sampling rate: answered, 250 Hz (500 samples = 2 s). omega_j = 2 pi f_j / 250
  rad per sample, with f_j log-spaced over 1 to 40 Hz (delta to low gamma), so
  omega spans about 0.025 to 1.0 rad per sample.
- Output: answered, |h| as in the paper; the readout is one version.
- Missing from the folder: Trabelsi 2018 (Deep Complex Networks: complex init and normalisation) and Orvieto 2023 (LRU). RotRNN summarises the LRU, so it is optional.
