function R = hpcCharmReference(cvalFile, outCsv)
%HPCCHARMREFERENCE  Per-fold val / test accuracy (%) of the paper's CHARM run, as a CSV.
%
%   hpcCharmReference('L:\plstm\HPC\PhaseAwareLSTM\cvalresults\CWL\DeepRNN\cval_DeepRNN_target-domain_2_CHARM_251104_1613.mat', ...
%                     'charm_hpc_reference.csv')
%
% That file is the HPC run (4 Nov 2025) behind the paper's CWL DeepRNN Fp1-Fz
% CHARM cells: 10-fold val 60.13, test 56.34 (projectionModule_baseline.csv).
% Read only.
S = load(cvalFile, 'evals');
n = numel(S.evals);
fold = (1:n)'; val = nan(n, 1); test = nan(n, 1);
for i = 1:n
    e = S.evals{i};
    val(i)  = 100*e.accuracyVal;
    test(i) = 100*e.accuracyTest;
end
R = table(repmat({'charm'}, n, 1), fold, val, test, 'VariableNames', {'layer','fold','val','test'});
if nargin > 1, writetable(R, outCsv); end
fprintf('[hpcCharmReference] %d folds: val %.2f, test %.2f\n', n, mean(val), mean(test));
end
