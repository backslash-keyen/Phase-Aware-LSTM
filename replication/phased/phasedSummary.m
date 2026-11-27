function T = phasedSummary(variants, refCsv)
%PHASEDSUMMARY  Per-fold and mean val / test accuracy (%) of the Phased LSTM runs in this tree.
%
%   phasedSummary({'phasedBase','phasedNeil'}, 'phased_hpc_reference.csv')
%
% Reads cvalresults/cwl/DeepRNN/folds/cwl_N2_target-domain_DeepRNN_phased-lstm__<variant>/fold<i>.mat
% (written by runReplication). With refCsv (hpcPhasedReference), adds the HPC fold
% values. Also writes phased_summary.csv (every fold row) next to the tree root.
if nargin < 1 || isempty(variants)
    addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'variants', 'phased', 'common'));
    variants = {phasedVersions().name};
end
ref = [];
if nargin > 1 && isfile(refCsv), ref = readtable(refCsv, 'TextType', 'char'); end
rows = {};
for k = 1:numel(variants)
    d = fullfile('cvalresults', 'cwl', 'DeepRNN', 'folds', ['cwl_N2_target-domain_DeepRNN_phased-lstm__' variants{k}]);
    for i = 1:10
        f = fullfile(d, sprintf('fold%d.mat', i));
        if ~isfile(f), continue; end
        S = load(f, 'evaluation');
        hv = NaN; ht = NaN;
        if ~isempty(ref) && any(ref.fold == i)
            hv = ref.val(ref.fold == i); ht = ref.test(ref.fold == i);
        end
        rows(end+1, :) = {variants{k}, i, 100*S.evaluation.accuracyVal, 100*S.evaluation.accuracyTest, hv, ht}; %#ok<AGROW>
    end
end
if isempty(rows), T = table(); fprintf('[phasedSummary] no Phased LSTM fold files here\n'); return; end
T = cell2table(rows, 'VariableNames', {'variant','fold','val','test','hpcVal','hpcTest'});
writetable(T, 'phased_summary.csv');
disp(T);
for k = 1:numel(variants)
    r = T(strcmp(T.variant, variants{k}), :);
    if isempty(r), continue; end
    fprintf('[phasedSummary] %-18s %2d folds: val %.2f (%.2f), test %.2f (%.2f); HPC on these folds val %.2f, test %.2f\n', ...
        variants{k}, height(r), mean(r.val), std(r.val), mean(r.test), std(r.test), mean(r.hpcVal), mean(r.hpcTest));
end
end
