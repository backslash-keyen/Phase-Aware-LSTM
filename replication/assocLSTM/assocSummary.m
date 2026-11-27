function T = assocSummary(versions)
%ASSOCSUMMARY  Per-fold and mean val / test accuracy (%) of the Associative LSTM runs in this tree.
%
%   assocSummary                       every version found
%   assocSummary({'paper','ideal'})
%
% Reads cvalresults/cwl/DeepRNN/folds/cwl_N2_target-domain_DeepRNN_assoc-lstm__al_<version>/fold<i>.mat
% (written by runReplication).
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'common'));
if nargin < 1 || isempty(versions), versions = {assocVersions().id}; end
versions = cellstr(versions);
rows = {};
for k = 1:numel(versions)
    d = fullfile('cvalresults', 'cwl', 'DeepRNN', 'folds', ['cwl_N2_target-domain_DeepRNN_assoc-lstm__al_' versions{k}]);
    for i = 1:10
        f = fullfile(d, sprintf('fold%d.mat', i));
        if ~isfile(f), continue; end
        S = load(f, 'evaluation');
        rows(end+1, :) = {versions{k}, i, 100*S.evaluation.accuracyVal, 100*S.evaluation.accuracyTest}; %#ok<AGROW>
    end
end
if isempty(rows), T = table(); fprintf('[assocSummary] no Associative LSTM fold files here\n'); return; end
T = cell2table(rows, 'VariableNames', {'version','fold','val','test'});
disp(T);
for k = 1:numel(versions)
    r = T(strcmp(T.version, versions{k}), :);
    if isempty(r), continue; end
    fprintf('[assocSummary] %-13s %2d folds: val %.2f (%.2f), test %.2f (%.2f)\n', ...
        versions{k}, height(r), mean(r.val), std(r.val), mean(r.test), std(r.test));
end
end
