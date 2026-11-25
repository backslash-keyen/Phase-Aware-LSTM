# Phase-aware LSTM: collapse diagnosis and fixes (2026-11-24 to 26)

CWL, 2 classes, DeepRNN source network, target domain (Fp1/Fz), phase-lstm projection.
Run on Keyen's PC (MATLAB R2025b, GTX 1650) from a copy of this branch; our own source
models unless noted; seed 0. Accuracies are train / val / test.

## 1. Why the released layer collapses

- The released phaseAwareLSTM collapses to one class (exactly 50%) on 7-9 of 10 folds.
- Not caused by: the speed rewrite, the source model set (ours, Oct 29, Oct 30 N2),
  reusing the fold-1 source model, the training options alone, or a signed output.
- At initialisation the projection gradient grows about 10x every 15-25 time steps
  (norm 645 at T=10, ~1e33 at T=500, NaN on some folds), in single and double precision.
- Released initialisation: phase bias b2 = 0; b1 unit-forget with the +1 on rows H+1:2H
  (the input gate in this layer) and row 2H zeroed by an off-by-one.

## 2. Fold-2 screen (released layer unless noted; tuned options = get_trainingOptions_CWL_C2_PhaseAware, 20 epochs)

| Setting | Train | Val | Test | Outcome |
|---|---|---|---|---|
| Released, default options | 50.0 | 50.0 | 50.0 | collapsed |
| Released, tuned options | 50.0 | 50.0 | 50.0 | collapsed |
| Clip ±5, b2 = 0 | 50.0 | 50.0 | 50.0 | collapsed |
| b2 = 1 on input rotation (rows H+1:2H) | 67.0 | 67.0 | 61.0 | learns |
| Clip ±5, b2 = 1 on input rotation | 68.6 | 69.8 | 63.0 | learns |
| b2 = π/4 on input rotation | 67.0 | 69.1 | 62.6 | learns |
| b2 = π/2 on input rotation | 64.3 | 68.0 | 60.8 | learns |
| b2 = π on input rotation | 66.6 | 71.0 | 62.4 | learns |
| b2 = π on input + candidate | 66.4 | 70.6 | 62.9 | learns |
| b2 = π on all four blocks | 66.4 | 71.7 | 62.3 | learns |
| b2 = π on candidate only | 58.1 | 62.1 | 57.1 | weak |
| b2 = π on forget only | 56.7 | 60.3 | 55.5 | weak |
| b2 = π on forget + input | 52.3 | 53.9 | 51.6 | near chance |
| b2 = π on forget, input, output | 50.0 | 50.0 | 50.0 | one class |
| b2 = π on output only (rows unused) | 50.0 | 50.0 | 50.0 | one class (= b2 = 0) |
| b2 = π on f, i, o, default options | 52.1 | 53.9 | 50.3 | near chance |
| halfAngMatched (clip ±5, full b1, b2 = 1) | 66.3 | 69.6 | 60.8 | learns |
| notrig, Nov 2025 layer (atan map, same settings) | 66.6 | 69.7 | 61.3 | learns |
| noCarry (no memory carry), default options | 50.0 | 50.0 | 50.0 | collapsed |
| LSTM projection (unchanged code) | 56.6 | 58.5 | 54.5 | |

The input-rotation phase bias decides it; any positive value from π/4 to π works. π on the
forget rotation hurts unless the candidate rows also get π. Clip, full b1 and angle map
(atan vs half-angle) change test accuracy by about 1-2 points.

## 3. Ten folds (test accuracy, %)

| Fold | halfAngMatched | piInputPhaseLSTM (released + π on input rotation) | Released, default | LSTM projection |
|---|---|---|---|---|
| 1 | 62.8 | 58.6 | 60.2 | 61.4 |
| 2 | 60.8 | 62.4 | 50.0 | 54.5 |
| 3 | 60.7 | 60.9 | 50.0 | 61.1 |
| 4 | 62.5 | 60.9 | 50.0 | 53.4 |
| 5 | 61.9 | 57.4 | 52.8 | 63.0 |
| 6 | 63.7 | 63.1 | 50.0 | 54.4 |
| 7 | 61.2 | 63.1 | 50.0 | 54.5 |
| 8 | 54.3 | 57.9 | 55.1 | 54.9 |
| 9 | 62.1 | 59.7 | 50.0 | 61.0 |
| 10 | 62.0 | 63.8 | 50.0 | 55.7 |
| Mean ± SD | 61.2 ± 2.6 | 60.8 ± 2.3 | 51.8 ± 3.4 | 57.4 ± 3.7 |
| Collapsed | 0/10 | 0/10 | 7/10 | 0/10 |
| Time per fold | ~6.5 min | ~6.3 min | ~4 min | ~1.4 min |

The piInputPhaseLSTM column was produced with a temporary edit equivalent to the committed
piInputPhaseLSTM variant (released fused layer, b2 = π on rows H+1:2H, tuned options).

## 4. Fast (fused) vs original layer

- Layer alone, double precision, T = 20, identical parameters: max relative difference
  Y 6e-8, gradients 4e-7 (the fused layer drops a redundant cos/sin renormalisation).
- Full training, matched setting, fold 2: original 67.1 / 69.9 / 61.5 in 65 min,
  fused 66.3 / 69.6 / 60.8 in 6.3 min. Over T = 500 the recurrence amplifies rounding,
  so trained results differ by under a point, not bit for bit.
