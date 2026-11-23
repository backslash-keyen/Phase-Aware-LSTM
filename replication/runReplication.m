function manifest = runReplication(varargin)
%RUNREPLICATION  Seeded, resumable driver for every experiment in the paper.
%
%   runReplication()                         % all experiments, source then target
%   runReplication('tables', {'table2'})     % only what Table 2 needs
%   runReplication('dryRun', true)           % print the plan and what is missing
%   runReplication('datasets','CWL', 'ncats',2, 'typeNets','DeepRNN', 'domains','source-domain')
%                                            % one dataset / classes / model / domain at a time
%   runReplication('listOnly', true)         % print experiment ids and exit
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
%   parTrain    false   parfor over folds (fold seeds stay = fold index)
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
addParameter(p, 'force',       false);
addParameter(p, 'dryRun',      false);
addParameter(p, 'stopOnError', false);
addParameter(p, 'hashData',    true);
parse(p, varargin{:});
o = p.Results;

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

preflight();

grid = experimentGrid('tables', o.tables, 'datasets', o.datasets, 'typeNets', o.typeNets, ...
                      'ncats', o.ncats, 'domains', o.domains, 'layerNames', o.layerNames, 'ids', o.ids);
if isempty(grid)
    fprintf('[replication] no experiment matches the filters.\n');
    manifest = readManifest(manifestFile);
    return
end
if o.listOnly
    fprintf('[replication] %d experiments selected (run order):\n', numel(grid));
    for g = 1:numel(grid)
        [ok, why] = supportCheck(grid(g));
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

    [ok, reason] = supportCheck(k);
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
            if ~isfile(pretrainedFile(projectLoc, k, i)), missingFolds(end+1) = i; end %#ok<AGROW>
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

    outDir = fullfile(projectLoc, 'cvalresults', config.datasetType, config.typeNet);
    before = listResults(outDir);
    setGlobalSeed(o.seed);
    row.startedAt = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    fprintf('[%3d/%d] run              %s (ncval=%d)\n', g, numel(grid), k.id, ncval);
    try
        kFold_crossValidation(config);
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
