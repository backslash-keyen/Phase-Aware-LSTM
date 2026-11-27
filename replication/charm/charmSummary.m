function T = charmSummary(variants, refCsv)
%CHARMSUMMARY  Per-fold and mean val / test accuracy (%) of the CHARM runs in this tree.
%
%   charmSummary({'charmBase','charm3Conv'}, 'charm_hpc_reference.csv')
%
% Reads cvalresults/cwl/DeepRNN/folds/cwl_N2_target-domain_DeepRNN_charm__<variant>/fold<i>.mat
% (written by runReplication). With refCsv (hpcCharmReference), adds the HPC
% fold values and the difference.
if nargin < 1 || isempty(variants)
    variants = {'charmBase','charm3Conv','charmInstNorm','charmSourceTrained'};
end
ref = [];
if nargin > 1 && isfile(refCsv), ref = readtable(refCsv, 'TextType', 'char'); end
rows = {};
for k = 1:numel(variants)
    d = fullfile('cvalresults', 'cwl', 'DeepRNN', 'folds', ['cwl_N2_target-domain_DeepRNN_charm__' variants{k}]);
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
if isempty(rows), T = table(); fprintf('[charmSummary] no CHARM fold files here\n'); return; end
T = cell2table(rows, 'VariableNames', {'variant','fold','val','test','hpcVal','hpcTest'});
disp(T);
for k = 1:numel(variants)
    r = T(strcmp(T.variant, variants{k}), :);
    if isempty(r), continue; end
    fprintf('[charmSummary] %-18s %2d folds: val %.2f (%.2f), test %.2f (%.2f); HPC on these folds val %.2f, test %.2f\n', ...
        variants{k}, height(r), mean(r.val), std(r.val), mean(r.test), std(r.test), mean(r.hpcVal), mean(r.hpcTest));
end
end
