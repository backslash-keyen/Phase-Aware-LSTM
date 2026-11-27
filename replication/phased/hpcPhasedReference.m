function R = hpcPhasedReference(cvalFile, outCsv)
%HPCPHASEDREFERENCE  Per-fold val / test accuracy (%) of the paper's Phased LSTM run, as a CSV.
%
%   hpcPhasedReference('L:\plstm\HPC\PhaseAwareLSTM\cvalresults\CWL\DeepRNN', 'phased_hpc_reference.csv')
%
% cvalFile is the HPC cval file behind the paper's CWL DeepRNN Fp1-Fz Phased-LSTM
% cell (run script experiments/CWL/C2/baseline_phase/RNN/td1/target_phasedLSTM_cwl_RNN_C2.m,
% layerName 'phased-LSTM'), or a folder: then the newest
% cval_DeepRNN_target-domain_2_phased*.mat in it is used. Read only.
if isfolder(cvalFile)
    d = dir(fullfile(cvalFile, 'cval_DeepRNN_target-domain_2_phased*.mat'));
    assert(~isempty(d), 'phased:noReference', 'No cval_DeepRNN_target-domain_2_phased*.mat in %s.', cvalFile);
    [~, j] = max([d.datenum]);
    cvalFile = fullfile(d(j).folder, d(j).name);
end
S = load(cvalFile, 'evals');
n = numel(S.evals);
fold = (1:n)'; val = nan(n, 1); test = nan(n, 1);
for i = 1:n
    e = S.evals{i};
    val(i)  = 100*e.accuracyVal;
    test(i) = 100*e.accuracyTest;
end
R = table(repmat({'phased-lstm'}, n, 1), fold, val, test, 'VariableNames', {'layer','fold','val','test'});
if nargin > 1, writetable(R, outCsv); end
fprintf('[hpcPhasedReference] %s, %d folds: val %.2f, test %.2f\n', cvalFile, n, mean(val), mean(test));
end
