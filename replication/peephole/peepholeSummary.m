function T = peepholeSummary(versions, lstmDir, refCsv)
%PEEPHOLESUMMARY  Per-fold and mean val / test accuracy (%) of the peephole runs in this tree.
%
%   peepholeSummary({}, 'J:\...\_tgt_hpc\cvalresults\cwl\DeepRNN\folds\cwl_N2_target-domain_DeepRNN_lstm', ...
%                   'J:\...\_tgt_hpc\hpc_reference.csv')
%
% Reads cvalresults/cwl/DeepRNN/folds/cwl_N2_target-domain_DeepRNN_lstm__ph_<version>/fold<i>.mat.
% lstmDir (optional): fold files of the plain LSTM projection of the same branch,
% for the paired difference. refCsv (optional): the HPC per-fold reference
% (rows with layer 'lstm').
addpath(fullfile(fileparts(mfilename('fullpath')), 'common'));
if nargin < 1 || isempty(versions), versions = {peepholeVersions().id}; end
if nargin < 2, lstmDir = ''; end
lstmT = nan(10, 1);
if ~isempty(lstmDir)
    for i = 1:10
        f = fullfile(lstmDir, sprintf('fold%d.mat', i));
        if isfile(f), S = load(f, 'evaluation'); lstmT(i) = 100*S.evaluation.accuracyTest; end
    end
end
hpcT = nan(10, 1);
if nargin > 2 && isfile(refCsv)
    R = readtable(refCsv, 'TextType', 'char'); R = R(strcmp(R.layer, 'lstm'), :);
    hpcT(R.fold) = R.test;
end
rows = {};
for k = 1:numel(versions)
    d = fullfile('cvalresults', 'cwl', 'DeepRNN', 'folds', ['cwl_N2_target-domain_DeepRNN_lstm__ph_' versions{k}]);
    for i = 1:10
        f = fullfile(d, sprintf('fold%d.mat', i));
        if ~isfile(f), continue; end
        S = load(f, 'evaluation');
        rows(end+1, :) = {versions{k}, i, 100*S.evaluation.accuracyVal, 100*S.evaluation.accuracyTest, lstmT(i), hpcT(i)}; %#ok<AGROW>
    end
end
if isempty(rows), T = table(); fprintf('[peepholeSummary] no peephole fold files here\n'); return; end
T = cell2table(rows, 'VariableNames', {'version','fold','val','test','lstmTest','hpcLstmTest'});
disp(T);
for k = 1:numel(versions)
    r = T(strcmp(T.version, versions{k}), :);
    if isempty(r), continue; end
    fprintf('[peepholeSummary] %-12s %2d folds: val %.2f (%.2f), test %.2f (%.2f); LSTM row on these folds %.2f (diff %+.2f); HPC LSTM %.2f\n', ...
        versions{k}, height(r), mean(r.val), std(r.val), mean(r.test), std(r.test), mean(r.lstmTest), mean(r.test - r.lstmTest), mean(r.hpcLstmTest));
end
end
