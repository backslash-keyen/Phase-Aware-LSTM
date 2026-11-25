function folds = collectResults(varargin)
%COLLECTRESULTS  Per-fold metrics of every finished experiment, from disk.
%
%   folds = collectResults()
%
% Reads replication/results/manifest.csv, takes the latest 'done' row per
% experiment, loads the evals saved by kFold_saveResults.m and writes
% replication/results/fold_metrics.csv (one row per experiment x fold).
% Accuracies are fractions in [0,1], exactly as the original code stores them.

p = inputParser;
addParameter(p, 'manifest', '');
parse(p, varargin{:});

repDir     = fileparts(mfilename('fullpath'));
resultsDir = fullfile(repDir, 'results');
manifestFile = p.Results.manifest;
if isempty(manifestFile), manifestFile = fullfile(resultsDir, 'manifest.csv'); end
if ~isfile(manifestFile)
    error('replication:noManifest', 'No manifest at %s. Run runReplication first.', manifestFile);
end

m = readtable(manifestFile, 'TextType', 'char', 'Delimiter', ',', 'VariableNamingRule', 'preserve');
done = m(strcmp(m.status, 'done'), :);
[~, last] = unique(done.id, 'last');
done = done(last, :);

rows = {};
for e = 1:height(done)
    file = done.resultFile{e};
    if ~isfile(file)
        warning('replication:missingResult', '%s: result file not found (%s)', done.id{e}, file);
        continue
    end
    S = load(file, 'evals');
    for i = 1:numel(S.evals)
        ev = S.evals{i};
        r.id          = done.id{e};
        r.datasetType = done.datasetType{e};
        r.ncats       = done.ncats(e);
        r.typeNet     = done.typeNet{e};
        r.domain      = done.domain{e};
        r.layerName   = done.layerName{e};
        r.fold        = i;
        r.accTrain    = getf(ev, 'accuracyTrain');
        r.accVal      = getf(ev, 'accuracyVal');
        r.accTest     = getf(ev, 'accuracyTest');
        r.sensMacroVal  = cmField(ev, 'confusedMatrix_val',  'sensitivity_macro');
        r.sensMacroTest = cmField(ev, 'confusedMatrix_test', 'sensitivity_macro');
        r.f1MacroVal    = cmField(ev, 'confusedMatrix_val',  'f1_macro');
        r.f1MacroTest   = cmField(ev, 'confusedMatrix_test', 'f1_macro');
        r.aucTest       = rocAuc(ev);
        r.trainingTime  = getf(ev, 'trainingTime');
        r.resultFile    = file;
        rows{end+1} = r; %#ok<AGROW>
    end
end

if isempty(rows)
    folds = table();
    warning('replication:noResults', 'No finished experiments yet.');
    return
end
folds = struct2table([rows{:}]);
writetable(folds, fullfile(resultsDir, 'fold_metrics.csv'));
fprintf('[replication] %d folds from %d experiments -> %s\n', height(folds), height(done), ...
        fullfile(resultsDir, 'fold_metrics.csv'));
end

function v = getf(s, f)
if isfield(s, f) && ~isempty(s.(f)), v = double(s.(f)); else, v = NaN; end
end

function v = cmField(ev, cmName, f)
v = NaN;
if isfield(ev, cmName) && isfield(ev.(cmName), f)
    v = double(ev.(cmName).(f));
    if ~isscalar(v), v = mean(v, 'omitnan'); end
end
end

function v = rocAuc(ev)
v = NaN;
if ~isfield(ev, 'confusedMatrix_test') || ~isfield(ev.confusedMatrix_test, 'roc'), return; end
roc = ev.confusedMatrix_test.roc;
if isfield(roc, 'auc')
    v = double(roc.auc);
elseif isfield(roc, 'micro') && isfield(roc.micro, 'auc')
    v = double(roc.micro.auc);
end
end
