function manifest = runReplication(varargin)
%RUNREPLICATION  Seeded, resumable driver for every experiment in the paper.
%
%   runReplication()                         % all experiments, source then target
%   runReplication('tables', {'table2'})     % only what Table 2 needs
%   runReplication('dryRun', true)           % print the plan and what is missing
%   runReplication('datasets','CWL', 'ncats',2, 'typeNets','DeepRNN', 'domains','source-domain')
%                                            % one dataset / classes / model / domain at a time
%   runReplication('listOnly', true)         % print experiment ids and exit
%   runReplication('ids', {'cwl_N2_target-domain_DeepRNN_phase-lstm'}, 'folds', 3)
%                                            % one fold of one experiment
%
% Name-value options
%   tables      {}      restrict to the experiments these tables need
%   datasets    {}      e.g. {'CWL'}
%   typeNets    {}      e.g. {'DeepRNN'}
%   ncats       []      e.g. 2
%   domains     {}      e.g. 'source-domain' or 'target-domain'
%   layerNames  {}      e.g. {'phase-lstm','lstm'}
%   ids         {}      exact experiment ids (see 'listOnly')
%   listOnly    false   print the selected experiment ids and exit
%   seed        0       master seed, set before every experiment
%   ncval       'auto'  'auto' = number of fold files found, or a number
%   perFold     true    save every fold as soon as it finishes and resume
%                       from the next unfinished fold (see "Per-fold saving")
%   folds       []      with perFold: run only these folds, e.g. 3 or 4:6
%   variant     ''      '' = released code unchanged (default)
%                       'freezeIntended' = apply the DeepRNN learn-rate
%                       factors freezeFunctionalLayers.m intends (0.1 / 0.2),
%                       see replication/variants/freezeIntended; results are
%                       stored and labelled separately (<id>__freezeIntended)
%                       'targetDomain2' = run the F3-F7 layout
%                       ('target-domain-2') through the original target-domain
%                       options and builders, no freezing change (<id>__targetDomain2)
%                       'authorSource' = target runs load the authors' source
%                       models from sourceModelRoot, read-only (<id>__authorSource)
%                       'fastPhaseLSTM' = speed only: the phase-aware projection
%                       uses phaseAwareLSTMFast, a copy of phaseAwareLSTM.m with
%                       the same arithmetic but no full-size copy of Y per time
%                       step; see replication/variants/fastPhaseLSTM and its
%                       replication/checkPhaseLSTMVersions.m (<id>__fastPhaseLSTM)
%                       'fastestPhaseLSTM' = speed only, version 3: same model,
%                       stacked per-step arithmetic (<id>__fastestPhaseLSTM)
%                       'backwardPhaseLSTM' = speed only, version 4: version-3
%                       forward + hand-written backward (<id>__backwardPhaseLSTM)
%                       'fusedPhaseLSTM' = speed only, version 5: version 4 with
%                       one fused GPU kernel per time step (<id>__fusedPhaseLSTM)
%                       'signedPhaseLSTM' = DIAGNOSTIC, not the released model:
%                       version 5 with a signed hidden output (<id>__signedPhaseLSTM)
%                       At most one of the phase-LSTM speed variants.
%                       Variants combine: {'authorSource','targetDomain2'}
%   sourceModelRoot 'M:\Earth\pretrainedModels'  used by authorSource; same
%                       layout as pretrainedModels/ (cwl\cwl1\DeepRNN\...)
%   parTrain    false   only with perFold=false: parfor over folds
%   force       false   rerun experiments already marked done
%   dryRun      false   check data / pretrained models only, train nothing
%   stopOnError false   stop at the first failed experiment
%
% Seeding. The released code already seeds each fold: perfold_training.m
% calls ensureReproducibility(i) (rng(i), gpurng(i), deterministic GPU
% algorithms) right before trainNetwork, and makeDatastore.m calls rng(1)
% before shuffling. This driver adds a master seed before each experiment
% so nothing that runs before the fold seed depends on leftover RNG state,
% and records every seed in the manifest.
%
% Per-fold saving. With perFold=true (default) the driver does what
% kFold_crossValidation.m does, one fold at a time: it calls the original
% perfold_training(config, i) for each fold, and saves that fold's model and
% evaluation to cvalresults/<dataset>/<typeNet>/folds/<experiment id>/fold<i>.mat
% the moment it returns. A rerun skips folds already saved. Once every fold
% is saved, the original kFold_saveResults(config, trainedModels, evals)
% writes the usual cval_*.mat and summary CSV, exactly as an uninterrupted
% kFold_crossValidation run would. No original file is changed. Fold seeds
% are unchanged: perfold_training still calls ensureReproducibility(i).
% Source models were already saved per fold by the original code
% (perfold_saveModel.m, pretrainedModels/).
%
% Output: replication/results/manifest.csv, one row per experiment with
% status, config, seeds, data hashes, MATLAB/GPU info, git commit and the
% cvalresults/*.mat file the experiment wrote.

p = inputParser;
addParameter(p, 'tables',      {});
addParameter(p, 'datasets',    {});
addParameter(p, 'typeNets',    {});
addParameter(p, 'ncats',       []);
addParameter(p, 'domains',     {});
addParameter(p, 'layerNames',  {});
addParameter(p, 'ids',         {});
addParameter(p, 'listOnly',    false);
addParameter(p, 'seed',        0);
addParameter(p, 'ncval',       'auto');
addParameter(p, 'parTrain',    false);
addParameter(p, 'perFold',     true);
addParameter(p, 'folds',       []);
addParameter(p, 'variant',     '');
addParameter(p, 'sourceModelRoot', 'M:\Earth\pretrainedModels');
addParameter(p, 'force',       false);
addParameter(p, 'dryRun',      false);
addParameter(p, 'stopOnError', false);
addParameter(p, 'hashData',    true);
parse(p, varargin{:});
o = p.Results;
v = variantInfo(o.variant, o.sourceModelRoot);
if ~isempty(v.name) && ~o.perFold
    error('replication:variantPerFold', 'Variants run only with ''perFold'', true (results are stored separately).');
end
if o.perFold && o.parTrain
    error('replication:perFoldParTrain', 'perFold runs folds one at a time; use parTrain only with ''perFold'', false.');
end

repDir     = fileparts(mfilename('fullpath'));
projectLoc = fileparts(repDir);
resultsDir = fullfile(repDir, 'results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
manifestFile = fullfile(resultsDir, 'manifest.csv');

% configureExperiment leaves projectLoc empty, so every path in the
% original code is relative to the current folder: run from the repo root.
oldDir = cd(projectLoc);
restoreDir = onCleanup(@() cd(oldDir));
addpath(genpath(fullfile(projectLoc, 'utils')));
addpath(repDir);

% Variant folders shadow original functions: only the requested one may be on the path
vroot = fullfile(repDir, 'variants');
for d = dir(vroot)'
    if d.isdir && ~startsWith(d.name, '.') && contains(path, fullfile(vroot, d.name))
        rmpath(fullfile(vroot, d.name));
    end
end
for i = 1:numel(v.dirs)
    addpath(v.dirs{i}, '-begin');
    fprintf('[replication] VARIANT path: %s is first on the path\n', v.dirs{i});
end
restorePath = onCleanup(@() cellfun(@rmpath, v.dirs)); %#ok<NASGU>
if ~isempty(v.sourceModelRoot)
    fprintf('[replication] VARIANT authorSource: target runs load source models from %s (read-only)\n', v.sourceModelRoot);
end

preflight();

grid = experimentGrid('tables', o.tables, 'datasets', o.datasets, 'typeNets', o.typeNets, ...
                      'ncats', o.ncats, 'domains', o.domains, 'layerNames', o.layerNames, 'ids', o.ids, ...
                      'variant', o.variant);
if isempty(grid)
    fprintf('[replication] no experiment matches the filters.\n');
    manifest = readManifest(manifestFile);
    return
end
if o.listOnly
    fprintf('[replication] %d experiments selected (run order):\n', numel(grid));
    for g = 1:numel(grid)
        [ok, why] = supportCheck(grid(g), v);
        fprintf('  %3d  %-55s %s\n', g, grid(g).id, ternary(ok, '', ['(unsupported: ' why ')']));
    end
    manifest = readManifest(manifestFile);
    return
end
manifest = readManifest(manifestFile);
[sha, dirty] = gitCommit(projectLoc);
env = environmentInfo();

fprintf('[replication] %d experiments, master seed %d, commit %s%s\n', ...
    numel(grid), o.seed, sha, ternary(dirty, ' (uncommitted changes)', ''));

for g = 1:numel(grid)
    k = grid(g);
    row = baseRow(k, o, sha, dirty, env);

    prev = manifest(strcmp(manifest.id, k.id), :);
    if ~o.force && ~isempty(prev) && strcmp(prev.status{end}, 'done') && isfile(prev.resultFile{end})
        fprintf('[%3d/%d] skip (done)      %s\n', g, numel(grid), k.id);
        continue
    end

    [ok, reason] = supportCheck(k, v);
    if ~ok
        row.status = 'unsupported'; row.message = reason;
        manifest = logRow(manifest, manifestFile, row, g, numel(grid), o.dryRun);
        continue
    end

    try
        files = foldFiles(projectLoc, k);
    catch ME
        files = [];
        row.message = ME.message;
    end
    if isempty(files)
        row.status  = 'missing-data';
        folder = dataFolder(projectLoc, k.datasetType, k.ncats, k.domain);
        if ~isfolder(folder)
            row.message = sprintf('data folder not found: %s', folder);
        elseif isempty(row.message)
            row.message = sprintf('no fold files in %s', folder);
        end
        manifest = logRow(manifest, manifestFile, row, g, numel(grid), o.dryRun);
        continue
    end
    if isnumeric(o.ncval), ncval = o.ncval; else, ncval = numel(files); end
    if ~isequal([files(1:min(ncval,end)).fold], 1:ncval)
        row.status  = 'missing-data';
        row.message = sprintf('need folds 1..%d, found %s', ncval, mat2str([files.fold]));
        manifest = logRow(manifest, manifestFile, row, g, numel(grid), o.dryRun);
        continue
    end
    row.ncval = ncval;

    dep = sourceDependency(k);
    if ~isempty(dep)
        missingFolds = [];
        for i = 1:ncval
            if ~isfile(pretrainedFile(projectLoc, k, i, v.sourceModelRoot)), missingFolds(end+1) = i; end %#ok<AGROW>
        end
        if ~isempty(missingFolds)
            row.status  = 'blocked';
            row.message = sprintf('source models missing for folds %s (run %s first)', ...
                                  mat2str(missingFolds), dep.id);
            if ~o.dryRun
                manifest = logRow(manifest, manifestFile, row, g, numel(grid), o.dryRun);
                continue
            end
        end
    end

    if o.hashData
        try
            row.dataSha256 = strjoin(arrayfun(@(f) fileSha256(f.path), files(1:ncval), 'UniformOutput', false), ' ');
        catch ME
            row.dataSha256 = ['hash failed: ' ME.message];
        end
    end

    if o.dryRun
        if isempty(row.status), row.status = 'ready'; end
        fd = fullfile(projectLoc, 'cvalresults', lower(k.datasetType), k.typeNet, 'folds', k.id);
        nSaved = 0;
        for i = 1:ncval, nSaved = nSaved + foldIsSaved(foldFile(fd, i)); end
        if nSaved > 0
            row.message = strtrim(sprintf('%s %d/%d folds already saved', row.message, nSaved, ncval));
        end
        manifest = logRow(manifest, manifestFile, row, g, numel(grid), true);
        continue
    end

    % ---------------- run ----------------
    config = configureExperiment('verbose', false);
    config.typeNet     = k.typeNet;
    config.layerName   = k.layerName;
    config.domain      = k.domain;
    config.datasetType = lower(k.datasetType);
    config.ncats       = k.ncats;
    config.ncval       = ncval;
    config.parTrain    = o.parTrain;
    config = config.validateModel();
    if ~isempty(v.sourceModelRoot)
        % authorSource: get_pretrainedLayers.m reads these properties; nothing is written there
        config.sourceModelLoc_CWL_1 = fullfile(v.sourceModelRoot, 'cwl', 'cwl1');
        config.sourceModelLoc_CWL_2 = fullfile(v.sourceModelRoot, 'cwl', 'cwl2');
        config.sourceModelLoc_MI_1  = fullfile(v.sourceModelRoot, 'mi',  'mi1');
        config.sourceModelLoc_MI_3  = fullfile(v.sourceModelRoot, 'mi',  'mi3');
    end

    outDir = fullfile(projectLoc, 'cvalresults', config.datasetType, config.typeNet);
    before = listResults(outDir);
    setGlobalSeed(o.seed);
    row.startedAt = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    fprintf('[%3d/%d] run              %s (ncval=%d)\n', g, numel(grid), k.id, ncval);
    try
        if o.perFold
            foldDir = fullfile(outDir, 'folds', k.id);
            [complete, nSaved] = runFoldsResumable(config, k, o, foldDir, resultsDir, sha);
            if ~complete
                row.status  = 'partial';
                row.message = sprintf('%d of %d folds saved in %s; rerun to continue', nSaved, ncval, foldDir);
                row.finishedAt = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
                manifest = logRow(manifest, manifestFile, row, g, numel(grid), false);
                continue
            end
            [trainedModels, evals] = loadFolds(foldDir, ncval);
            if v.affects(k)
                % variant results never go next to the unchanged-code results
                row.resultFile = saveVariantResults(config, trainedModels, evals, v.nameFor(k), v.suffixFor(k), outDir);
                row.status = 'done';
                row.message = ['variant ' v.nameFor(k)];
                clear trainedModels evals
                row.finishedAt = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
                manifest = logRow(manifest, manifestFile, row, g, numel(grid), false);
                continue
            end
            kFold_saveResults(config, trainedModels, evals);   % original writer
            clear trainedModels evals
        else
            kFold_crossValidation(config);
        end
        after = listResults(outDir);
        newFiles = setdiff(after, before);
        pattern = sprintf('cval_%s_%s_%d_%s_', config.typeNet, config.domain, config.ncats, config.layerName);
        newFiles = newFiles(startsWith(newFiles, pattern));
        if isempty(newFiles)
            % same-minute rerun overwrote an existing file: take the newest match
            m = dir(fullfile(outDir, [pattern '*.mat']));
            [~, j] = max([m.datenum]);
            newFiles = {m(j).name};
        end
        row.resultFile = fullfile(outDir, newFiles{end});
        row.status = 'done';
    catch ME
        row.status  = 'failed';
        row.message = strrep(ME.message, newline, ' ');
        if o.stopOnError
            row.finishedAt = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
            manifest = logRow(manifest, manifestFile, row, g, numel(grid), false); %#ok<NASGU>
            rethrow(ME);
        end
    end
    row.finishedAt = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    manifest = logRow(manifest, manifestFile, row, g, numel(grid), false);
end

summarize(manifest, grid);
end

% =====================================================================
function [complete, nSaved] = runFoldsResumable(config, k, o, foldDir, resultsDir, sha)
% Run the folds that are not saved yet, saving each one as it finishes.
if ~exist(foldDir, 'dir'), mkdir(foldDir); end
todo = 1:config.ncval;
if ~isempty(o.folds), todo = intersect(todo, o.folds); end
for i = todo
    f = foldFile(foldDir, i);
    if ~o.force && foldIsSaved(f)
        fprintf('        fold %2d  saved earlier, skipping\n', i);
        continue
    end
    setGlobalSeed(o.seed);                       % same pre-fold RNG state on fresh and resumed runs
    t0 = tic;
    startedAt = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    [model, evaluation] = perfold_training(config, i);   % original: seeds rng(i) before training
    foldMeta = struct('experimentId', k.id, 'fold', i, 'masterSeed', o.seed, ...
        'foldSeed', sprintf('rng(%d) via ensureReproducibility', i), 'gitCommit', sha, ...
        'startedAt', startedAt, 'finishedAt', char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')), ...
        'matlab', version);
    tmp = [f '.partial.mat'];
    save(tmp, 'model', 'evaluation', 'foldMeta', '-v7.3');
    movefile(tmp, f, 'f');                       % never leave a half-written fold file
    logFold(resultsDir, k.id, i, toc(t0), evaluation, f);
    fprintf('        fold %2d  saved  test acc %.2f%%  (%s)\n', i, 100 * evaluation.accuracyTest, f);
    clear model evaluation
end
nSaved = 0;
for i = 1:config.ncval
    nSaved = nSaved + foldIsSaved(foldFile(foldDir, i));
end
complete = nSaved == config.ncval;
end

function file = saveVariantResults(config, trainedModels, evals, vname, vsuffix, outDir)
% Same content as kFold_saveResults.m, in cvalresults/<ds>/<typeNet>/variants/<name>/
vdir = fullfile(outDir, 'variants', vname);
if ~exist(vdir, 'dir'), mkdir(vdir); end
timestamp = char(datetime('now', 'Format', 'yyMMdd_HHmm'));
file = fullfile(vdir, sprintf('cval_%s_%s_%d_%s%s_%s.mat', config.typeNet, config.domain, ...
                config.ncats, config.layerName, vsuffix, timestamp));
variant = vname; %#ok<NASGU>
save(file, 'trainedModels', 'evals', 'variant', '-v7.3');
[mTr, mVa, mTe, mTime] = generateResultSummary(config, evals);   % original
csv = fullfile(vdir, sprintf('projectionModule_%s.csv', vname));
if ~isfile(csv)
    fid = fopen(csv, 'w');
    fprintf(fid, 'Variant,TypeNet,Domain,NCats,LayerName,TrainAcc,ValAcc,TestAcc,TrainTime,Timestamp\n');
    fclose(fid);
end
fid = fopen(csv, 'a');
fprintf(fid, '%s,%s,%s,%d,%s,%.4f,%.4f,%.4f,%.2f,%s\n', vname, config.typeNet, config.domain, ...
        config.ncats, config.layerName, mTr, mVa, mTe, mTime, timestamp);
fclose(fid);
fprintf('[replication] variant results -> %s\n', file);
end

function f = foldFile(foldDir, i)
f = fullfile(foldDir, sprintf('fold%d.mat', i));
end

function ok = foldIsSaved(f)
ok = false;
if ~isfile(f), return; end
try
    w = whos('-file', f);
    ok = all(ismember({'model','evaluation'}, {w.name}));
catch
    ok = false;
end
end

function [trainedModels, evals] = loadFolds(foldDir, ncval)
trainedModels = cell(1, ncval);
evals         = cell(1, ncval);
for i = 1:ncval
    S = load(foldFile(foldDir, i), 'model', 'evaluation');
    trainedModels{i} = S.model;
    evals{i}         = S.evaluation;
end
end

function logFold(resultsDir, id, fold, seconds, ev, file)
f = fullfile(resultsDir, 'fold_log.csv');
newFile = ~isfile(f);
fid = fopen(f, 'a');
c = onCleanup(@() fclose(fid));
if newFile
    fprintf(fid, 'id,fold,seconds,accTrain,accVal,accTest,file,finishedAt\n');
end
fprintf(fid, '%s,%d,%.1f,%.6f,%.6f,%.6f,%s,%s\n', id, fold, seconds, ...
    ev.accuracyTrain, ev.accuracyVal, ev.accuracyTest, strrep(file, ',', ';'), ...
    char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss')));
end

function preflight()
if isempty(which('trainNetwork'))
    error('replication:noDLT', 'Deep Learning Toolbox is required.');
end
if canUseGPU && isempty(which('deep.gpu.deterministicAlgorithms'))
    error('replication:gpuDeterminism', ['A GPU is available, but deep.gpu.deterministicAlgorithms ' ...
        '(R2024b+) is not. ensureReproducibility.m calls it and would fail. Use MATLAB R2024b or ' ...
        'newer, or hide the GPU (e.g. setenv CUDA_VISIBLE_DEVICES=-1 before starting MATLAB).']);
end
end

function setGlobalSeed(seed)
rng(seed, 'twister');
if canUseGPU
    gpurng(seed, 'Philox');
end
end

function env = environmentInfo()
v = ver('MATLAB');
env.matlab = sprintf('%s %s', v.Version, v.Release);
env.gpu = 'none';
if canUseGPU
    d = gpuDevice;
    env.gpu = sprintf('%s (driver %s)', d.Name, num2str(d.DriverVersion));
end
env.host = strtrim(getenv('COMPUTERNAME'));
if isempty(env.host), env.host = strtrim(getenv('HOSTNAME')); end
end

function row = baseRow(k, o, sha, dirty, env)
row = struct( ...
    'id', k.id, 'status', '', 'message', '', ...
    'datasetType', k.datasetType, 'ncats', k.ncats, 'typeNet', k.typeNet, ...
    'domain', k.domain, 'layerName', k.layerName, 'ncval', NaN, ...
    'masterSeed', o.seed, 'foldSeeds', 'rng(fold) via ensureReproducibility; rng(1) datastore shuffle', ...
    'parTrain', double(o.parTrain), ...
    'resultFile', '', 'dataSha256', '', ...
    'gitCommit', sha, 'gitDirty', double(dirty), ...
    'matlab', env.matlab, 'gpu', env.gpu, 'host', env.host, ...
    'startedAt', '', 'finishedAt', '', 'usedBy', k.usedBy);
end

function m = readManifest(file)
if isfile(file)
    m = readtable(file, 'TextType', 'char', 'Delimiter', ',', 'VariableNamingRule', 'preserve');
    for v = {'id','status','resultFile'}
        % an all-empty column comes back numeric; make every text column cellstr
        if ~iscell(m.(v{1})), m.(v{1}) = repmat({''}, height(m), 1); end
    end
else
    m = table(cell(0,1), cell(0,1), cell(0,1), 'VariableNames', {'id','status','resultFile'});
end
end

function m = logRow(m, file, row, g, n, dryRun)
fprintf('[%3d/%d] %-16s %s  %s\n', g, n, row.status, row.id, row.message);
if dryRun, return; end
% keep the CSV trivially parseable: no commas or newlines inside fields
f = fieldnames(row);
for i = 1:numel(f)
    if ischar(row.(f{i}))
        row.(f{i}) = regexprep(row.(f{i}), '[,\r\n]+', ';');
    end
end
t = struct2table(row, 'AsArray', true);
if ~isfile(file)
    writetable(t, file);
else
    writetable(t, file, 'WriteMode', 'append');
end
m = readManifest(file);
end

function names = listResults(outDir)
d = dir(fullfile(outDir, 'cval_*.mat'));
names = {d.name};
end

function summarize(m, grid)
if isempty(m) || height(m) == 0, return; end
fprintf('\n[replication] latest status per experiment:\n');
ids = {grid.id};
status = repmat({'(not logged)'}, size(ids));
for i = 1:numel(ids)
    r = find(strcmp(m.id, ids{i}), 1, 'last');
    if ~isempty(r), status{i} = m.status{r}; end
end
[u, ~, j] = unique(status);
for i = 1:numel(u)
    fprintf('   %-14s %d\n', u{i}, sum(j == i));
end
end

function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end
