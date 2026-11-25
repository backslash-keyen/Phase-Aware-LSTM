# Phase-aware LSTM layer versions (inventory, 2026-11-25)

Oldest first. "Parent" is the version each one is compared with. Dates are file dates (L:, M:) or commit dates (repo branch claude/project-thread-a4wh15, PR 2).

## Keyen's originals (read-only, L: and M:)

| # | Version | File | Date | Kind | Changes vs parent |
|---|---|---|---|---|---|
| 1 | cstm_arch1 (March) | L:\PhysciaScriptaCodes_Ver1\...\ProjectionTransferLearning\helperFunctions\modelFunctions\cstm_arch1.m | 31 Mar 2025 | first "Euler-LSTM" | theta = atan(H); b1 unit-forget; b2 = unit-forget x pi/2; many max(.,eps) clamps; no memory carry |
| 2 | cstm_arch2 | ...\helperFunctions\cstm_arch2.m | 1 Apr 2025 | variant of 1 | polar memory update cos/sin(f_phase + memoryPhase), memory carried; same clamps; b2 = pi/2 |
| 3 | cstm_arch1 (June) | ...\helperFunctions\cstm_arch1.m | 26 Jun 2025 | June 2025 runs (never collapsed) | as 1, but b2 = unit-forget x 2*pi; signed output; still no memory carry |
| 4 | phaseAwareLSTM_notrig (archive) | M:\Earth\archive\phaseAwareLSTM_notrig.m | 23 Oct 2025 | early notrig | not diffed |
| 5 | phaseAwareLSTM_noTrig | M:\Earth\helperLibrary\modelframe\architectures\layerfunctions\phaseAwareLSTM_noTrig.m | 29 Oct 2025 | notrig | identical computation to 6 (only class name; Op line active but unused) |
| 6 | phaseAwareLSTM_notrig (paper numbers) | L:\plstm\helperLibrary\modelframe\phaseAwareLSTM_notrig.m | 3 Nov 2025 | produced the paper's CWL phase-lstm results (run 5 Nov 2025, R2022b, HPC CPU) | Euler-LSTM with memory carried; theta = atan(H) via 1/sqrt(1+H^2); clip +-5; b1 and b2 = unit-forget (+1 on rows H+1:2H); h = o.*tanh(abs c); raw atan2 phase fed back via Q |
| 7 | phase-lstm-halfAng | L:\plstm (file name/date not yet checked) | Nov 2025 | half-angle run in the Nov project | same math as 8 |
| 8 | Released phaseAwareLSTM | utils\modelstuff\layerdefinitions\phaseAwareLSTM.m (repo import ac1e221, 23 Nov 2026) | authors' archive | the published layer | vs 6: theta via tan-half-angle t = tanh(H/2); clip +-20; b1 zeroed from row 2H (off-by-one, H-1 ones); b2 = 0; renormalisation by r |
| - | Later M: variants | M:\Earth\toolbox\+phaseLSTM\+variants\... | Mar 2026 | newer | not diffed |

## Copies on the PR 2 branch (opt-in variants; the released file is never changed)

| # | Variant | Class / file | Commit, date | Kind | Changes vs parent |
|---|---|---|---|---|---|
| v2 | fastPhaseLSTM | phaseAwareLSTMFast | e2facd1, 24 Nov 2026 | speed, bit-identical to 8 | outputs collected in a cell (no per-step full copy); loop-invariant work hoisted |
| v3 | fastestPhaseLSTM | phaseAwareLSTMFastest | cc00324, 24 Nov 2026 | speed, same model as 8 | one stacked [7H x 2H] matmul per step; 0.5 folded into weights; no r renorm (differs by ~1e-9) |
| v4 | backwardPhaseLSTM | phaseAwareLSTMBackward | cc00324, 24 Nov 2026 | speed, same model as 8 | v3 forward + hand-written backward pass |
| v5 | fusedPhaseLSTM | phaseAwareLSTMFused | ec58eea, 24 Nov 2026 | speed, same model as 8 (about 10x per fold) | v4 with one fused GPU kernel per step, forward and backward |
| d1 | signedPhaseLSTM | phaseAwareLSTMSigned | 6fff21d, 24-25 Nov 2026 | diagnostic | v5 with signed output h = o.*tanh(sign(cR+cI).*abs c) |
| d2 | noCarryPhaseLSTM | phaseAwareLSTMNoCarry | 57d7642, 25 Nov 2026 | diagnostic | 8 (autodiff) without memory carry; gates and fed-back phase clamped at eps (as 3) |
| d3 | notrigPhaseLSTM | phaseAwareLSTM_notrig (copy of 6) | 3660f07, 25 Nov 2026 | reproduction | 6 unchanged (verified bit-identical) + its options routing: tuned options, 20 epochs |
| d4 | halfAngMatchedPhaseLSTM | phaseAwareLSTMHalfAngMatched | 81d44ee, 25 Nov 2026 | diagnostic pair to d3 | v5 with 6's clip +-5, full b1, b2 = unit-forget, and d3's options; only the angle map differs from d3 |

Temporary diagnostic edits (not committed; run in _speedcheck and restored): v5 with b2 = pi; d4 with b2 = 0; d4 with released clip/b1 + b2 = pi; tonight's clip/phase-bias screen on 8.
