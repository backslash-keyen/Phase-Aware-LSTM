function out = buildTables(varargin)
%BUILDTABLES  Build every table of the paper from this repository's own runs.
%
%   buildTables()                        % all tables
%   buildTables('tables', {'table4'})    % one table
%   buildTables('variant', 'freezeIntended')  % same tables from the variant runs -> results/tables_freezeIntended/
%
% Inputs : replication/results/manifest.csv (written by runReplication)
%          the cvalresults/*.mat files the manifest points to
%          the training-option files (Table 1)
% Outputs: replication/results/tables/<id>.csv   long format, one row per cell
%          replication/results/tables/<id>.md    the table as laid out in the paper
%          replication/results/tables/<id>.tex   same, LaTeX tabular
%          replication/results/tables/<id>_stats.csv  one-tailed paired t-tests
%
% Accuracy cells are "mean (SD)" across folds in %, as in the paper. A
% cell is left blank when its experiment has not produced results; the
% long CSV and the table footnotes say why (not run, unsupported by the
% released code, missing data, ...). The paper's own numbers are never
% read or stored.

p = inputParser;
addParameter(p, 'tables', {});
addParameter(p, 'decimals', 2);
addParameter(p, 'statsMetric', 'accTest');
addParameter(p, 'variant', '');
parse(p, varargin{:});
o = p.Results;

repDir     = fileparts(mfilename('fullpath'));
projectLoc = fileparts(repDir);
v = variantInfo(o.variant);
if isempty(v.name)
    outDir = fullfile(repDir, 'results', 'tables');
else
    outDir = fullfile(repDir, 'results', ['tables_' strrep(v.name, '+', '_')]);   % never overwrites the unchanged-code tables
end
if ~exist(outDir, 'dir'), mkdir(outDir); end
oldDir = cd(projectLoc); restoreDir = onCleanup(@() cd(oldDir));
addpath(genpath(fullfile(projectLoc, 'utils')));
addpath(repDir);

T = paperTables();
if ~isempty(o.tables), T = T(ismember({T.id}, cellstr(o.tables))); end
T = applyVariant(T, v);

manifestFile = fullfile(repDir, 'results', 'manifest.csv');
manifest = table();
folds = table();
if isfile(manifestFile)
    manifest = readtable(manifestFile, 'TextType', 'char', 'Delimiter', ',', 'VariableNamingRule', 'preserve');
    for v = {'id','status','message','resultFile'}
        % an all-empty column comes back numeric; make every text column cellstr
        if ~iscell(manifest.(v{1}))
            manifest.(v{1}) = repmat({''}, height(manifest), 1);
        end
    end
    folds = collectResults();
else
    warning('replication:noManifest', 'No manifest yet: tables will be empty templates.');
end

out = struct('id', {}, 'files', {});
for t = 1:numel(T)
    switch T(t).kind
        case 'accuracy',        files = buildAccuracy(T(t), folds, manifest, outDir, o);
        case 'hyperparameters', files = buildHyper(T(t), outDir);
        case 'complexity',      files = buildComplexity(T(t), folds, manifest, projectLoc, outDir, o);
    end
    out(end+1) = struct('id', T(t).id, 'files', {files}); %#ok<AGROW>
    fprintf('[tables] %-8s -> %s\n', T(t).id, strjoin(files, ', '));
end
end

% =====================================================================
%  Accuracy tables (Tables 2-9)
% =====================================================================
function files = buildAccuracy(t, folds, manifest, outDir, o)
splits = {'Validation','Test'; 'accVal','accTest'};
long = {};
header = [{'Network'}, ternaryCell(any(~cellfun(@isempty, {t.rows.variant})), {'Projection Layer'}, {})];
colHead = {};
for gi = 1:numel(t.groups)
    for s = 1:size(splits,2)
        colHead{end+1} = sprintf('%s %s', t.groups{gi}, splits{1,s}); %#ok<AGROW>
    end
end
grid = cell(numel(t.rows), numel(colHead));
notes = {};
for r = 1:numel(t.rows)
    row = t.rows(r);
    for gi = 1:numel(t.groups)
        if gi > numel(row.cells), c = struct('kind','dash','id','','reason',''); else, c = row.cells{gi}; end
        for s = 1:size(splits,2)
            [txt, mu, sd, n, status, why] = cellValue(c, splits{2,s}, folds, manifest, o.decimals);
            grid{r, (gi-1)*2 + s} = txt;
            long(end+1, :) = {row.section, row.network, row.variant, t.groups{gi}, splits{1,s}, ...
                              mu, sd, n, status, c.id, why}; %#ok<AGROW>
            if ~isempty(why) && ~strcmp(status, 'dash'), notes{end+1} = why; end %#ok<AGROW>
        end
    end
    if ~isempty(row.note), notes{end+1} = row.note; end %#ok<AGROW>
end

L = cell2table(long, 'VariableNames', {'section','network','variant','group','split', ...
                                         'meanPct','sdPct','nFolds','status','experimentId','reason'});
files = {writeLong(L, outDir, t.id)};
files = [files, writeLayout(t, header, colHead, grid, unique(notes, 'stable'), outDir)];

S = accuracyStats(t, folds, o.statsMetric);
if ~isempty(S)
    f = fullfile(outDir, [t.id '_stats.csv']);
    writetable(S, f);
    files{end+1} = f;
end
end

function [txt, mu, sd, n, status, why] = cellValue(c, metric, folds, manifest, dec)
mu = NaN; sd = NaN; n = 0; why = c.reason;
switch c.kind
    case 'dash'
        txt = '—'; status = 'dash'; return
    case 'missing'
        txt = ''; status = 'not-in-released-code'; return
end
v = [];
if ~isempty(folds) && height(folds) > 0
    v = folds.(metric)(strcmp(folds.id, c.id));
end
if isempty(v)
    txt = '';
    [status, why] = latestStatus(c.id, manifest);
    return
end
v  = 100 * v(:);
mu = mean(v, 'omitnan'); sd = std(v, 'omitnan'); n = sum(~isnan(v));
txt = sprintf('%.*f (%.*f)', dec, mu, dec, sd);
status = 'done';
end

function [status, why] = latestStatus(id, manifest)
status = 'not-run'; why = 'experiment has not been run yet';
if isempty(manifest) || height(manifest) == 0, return; end
r = find(strcmp(manifest.id, id), 1, 'last');
if isempty(r), return; end
status = manifest.status{r};
why = manifest.message{r};
if isempty(why), why = status; end
end

function S = accuracyStats(t, folds, metric)
% One-tailed paired t-tests across folds (as in the paper):
%   (a) phase-aware LSTM vs every other row of the same block & column group
%   (b) every transfer row vs the no-transfer row of the same network/layout
S = table();
if isempty(folds) || height(folds) == 0, return; end
rows = {};
blocks = unique(strcat({t.rows.section}, '|', {t.rows.network}), 'stable');
for b = 1:numel(blocks)
    in = find(strcmp(strcat({t.rows.section}, '|', {t.rows.network}), blocks{b}));
    for gi = 1:numel(t.groups)
        ids = cell(size(in));
        for j = 1:numel(in)
            cc = t.rows(in(j)).cells;
            if gi <= numel(cc) && strcmp(cc{gi}.kind, 'exp'), ids{j} = cc{gi}.id; else, ids{j} = ''; end
        end
        ref = find(contains(ids, '_phase-lstm'), 1);
        if isempty(ref), continue; end
        for j = 1:numel(in)
            if j == ref || isempty(ids{j}), continue; end
            rows = addTest(rows, t, in(ref), in(j), t.groups{gi}, ids{ref}, ids{j}, folds, metric, 'phase-LSTM > other');
        end
    end
end
% (b) against no-transfer
for r = 1:numel(t.rows)
    for gi = 1:numel(t.rows(r).cells)
        c = t.rows(r).cells{gi};
        if ~strcmp(c.kind, 'exp') || startsWith(c.domain, 'source') || strcmpi(c.layerName, c.typeNet), continue; end
        base = expKey(c.datasetType, c.ncats, c.typeNet, c.domain, c.typeNet);
        rows = addTest(rows, t, r, NaN, t.groups{gi}, c.id, base.id, folds, metric, 'transfer > no-transfer');
    end
end
if ~isempty(rows)
    S = cell2table(vertcat(rows{:}), 'VariableNames', {'section','network','rowA','rowB','group', ...
        'experimentA','experimentB','hypothesis','metric','nPairs','meanDiffPct','tStat','df','pOneTailed'});
end
end

function rows = addTest(rows, t, ra, rb, group, idA, idB, folds, metric, hyp)
a = folds(strcmp(folds.id, idA), :);
b = folds(strcmp(folds.id, idB), :);
if isempty(a) || isempty(b), return; end
[f, ia, ib] = intersect(a.fold, b.fold);
if numel(f) < 2, return; end
[tStat, df, pval, md] = pairedTTest1(100*a.(metric)(ia), 100*b.(metric)(ib));
if isnan(rb), labelB = 'No transfer'; else, labelB = t.rows(rb).variant; end
rows{end+1} = {t.rows(ra).section, t.rows(ra).network, t.rows(ra).variant, labelB, group, ...
               idA, idB, hyp, metric, numel(f), md, tStat, df, pval};
end

% =====================================================================
%  Table 1 — hyperparameters, read from the training-option files
% =====================================================================
function files = buildHyper(t, outDir)
grid = cell(numel(t.rows), numel(t.columns));
long = {};
notes = {};
for r = 1:numel(t.rows)
    row = t.rows(r);
    vals = repmat({''}, 1, 5);
    why = '';
    try
        cs = makeConfig(row.datasetType, row.typeNet, 'source-domain');
        ct = makeConfig(row.datasetType, row.typeNet, 'target-domain');
        os = evalc_quiet(@() get_trainingOptions(cs));
        ot = evalc_quiet(@() get_trainingOptions(ct));
        vals = {num2str(os.MiniBatchSize), num2str(os.InitialLearnRate, 4), num2str(os.L2Regularization, 4), ...
                num2str(os.MaxEpochs), num2str(ot.MaxEpochs)};
    catch ME
        why = ME.message;
        notes{end+1} = sprintf('%s / %s: %s', row.datasetType, row.network, why); %#ok<AGROW>
    end
    grid(r, :) = [vals, {row.note}];
    long(end+1, :) = [{row.section, row.network}, vals, {row.note, why}]; %#ok<AGROW>
end
L = cell2table(long, 'VariableNames', {'section','network','Batch','LR','L2','EpochsSrc','EpochsTgt','Notes','error'});
files = {writeLong(L, outDir, t.id)};
notes{end+1} = 'Read from utils/modelstuff/trainingoptions via get_trainingOptions.m, after its source (30) / target (15) epoch caps.';
files = [files, writeLayout(t, {'Model'}, t.columns, grid, notes, outDir)];
end

function cfg = makeConfig(ds, typeNet, domain)
cfg = configureExperiment('verbose', false);
cfg.datasetType = lower(ds);
cfg.typeNet = typeNet;
cfg.domain = domain;
cfg = cfg.validateModel();
end

function v = evalc_quiet(fn)
% get_trainingOptions_*.m print a banner; keep the log clean
v = [];
evalc('v = fn();');
end

% =====================================================================
%  Table 10 — complexity and runtime, measured from the trained networks
% =====================================================================
function files = buildComplexity(t, folds, manifest, projectLoc, outDir, o)
grid = cell(numel(t.rows), numel(t.columns));
long = {};
notes = {};
for r = 1:numel(t.rows)
    c = t.rows(r).cells{1};
    vals = repmat({''}, 1, 6);
    nums = nan(1, 6);
    status = c.kind; why = c.reason;
    if strcmp(c.kind, 'exp')
        [status, why] = latestStatus(c.id, manifest);
        res = manifest(strcmp(manifest.id, c.id) & strcmp(manifest.status, 'done'), :);
        if ~isempty(res)
            try
                nums = complexityOf(res.resultFile{end}, c, projectLoc);
                tt = folds.trainingTime(strcmp(folds.id, c.id));
                nums(4) = mean(tt, 'omitnan');
                fmt = {'%d','%.2f','%.1f','%.1f','%.2f','%.1f'};
                for j = 1:6
                    if ~isnan(nums(j)), vals{j} = sprintf(fmt{j}, nums(j)); end
                end
                status = 'done'; why = '';
            catch ME
                status = 'failed'; why = ME.message;
            end
        end
    end
    if ~isempty(why) && ~strcmp(status, 'done'), notes{end+1} = why; end %#ok<AGROW>
    if ~isempty(t.rows(r).note), notes{end+1} = t.rows(r).note; end %#ok<AGROW>
    grid(r, :) = vals;
    long(end+1, :) = [{t.rows(r).section, t.rows(r).network, t.rows(r).variant}, num2cell(nums), {status, c.id, why}]; %#ok<AGROW>
end
L = cell2table(long, 'VariableNames', {'section','network','variant','params','MFLOPs','diskKB', ...
                                         'trainTimeS','inferenceMs','inferenceMemKB','status','experimentId','reason'});
files = {writeLong(L, outDir, t.id)};
notes = [notes, { ...
  '#Params: learnable parameters of the projection layer (the layer fed by the input layer), fold-1 network.', ...
  'FLOPs: analytic estimate 2 x T x (projection weights) per trial, T = trial length.', ...
  'Disk-Size: fold-1 trained network saved alone with save -v7.3.', ...
  'Training Time: mean trainingTime over folds, from the saved evals.', ...
  'Inference Time: median of 30 timed predict() calls on one test trial after warm-up, this machine.', ...
  'Inference Mem: bytes of the projection activations for one trial plus all learnables.'}];
files = [files, writeLayout(t, {'Network','Projection'}, t.columns, grid, unique(notes, 'stable'), outDir)];
end

function nums = complexityOf(resultFile, k, projectLoc)
S = load(resultFile, 'trainedModels');
net = S.trainedModels{1}.trainedNet;
nums = nan(1, 6);

C = net.Connections;
src = string(C.Source); dst = string(C.Destination);
projName = char(extractBefore(dst(find(src == "input", 1)) + "/", "/"));

lg = layerGraph(net);
isOut = arrayfun(@(L) isa(L, 'nnet.cnn.layer.ClassificationOutputLayer'), lg.Layers);
lg = removeLayers(lg, {lg.Layers(isOut).Name});
dl = dlnetwork(lg);
Lt = dl.Learnables;
inProj = strcmp(Lt.Layer, projName);
nums(1) = sum(cellfun(@numel, Lt.Value(inProj)));
isW = inProj & ~contains(lower(Lt.Parameter), 'bias');

files = foldFiles(projectLoc, k);
D = load(files(1).path, 'xtest');
X = D.xtest(:, :, 1);
T = size(X, 2);
nums(2) = 2 * T * sum(cellfun(@numel, Lt.Value(isW))) / 1e6;

tmp = [tempname '.mat'];
save(tmp, 'net', '-v7.3');
d = dir(tmp); nums(3) = d.bytes / 1024; delete(tmp);

predict(net, X);                              % warm-up
tt = zeros(1, 30);
for i = 1:30
    t0 = tic; predict(net, X); tt(i) = toc(t0);
end
nums(5) = 1000 * median(tt);

A = activations(net, X, projName);
learnBytes = sum(cellfun(@(v) numel(v) * 4, Lt.Value));
nums(6) = (numel(A) * 4 + learnBytes) / 1024;
end

% =====================================================================
%  Writers
% =====================================================================
function f = writeLong(L, outDir, id)
f = fullfile(outDir, [id '.csv']);
writetable(L, f);
end

function files = writeLayout(t, header, colHead, grid, notes, outDir)
hasVariant = any(strcmp(header, 'Projection Layer')) || any(strcmp(header, 'Projection'));
sections = {t.rows.section};

% ---------- Markdown ----------
md = {sprintf('**%s.** %s', upperFirst(t.id), t.caption), ''};
cols = [header, colHead];
md{end+1} = ['| ' strjoin(cols, ' | ') ' |'];
md{end+1} = ['|' repmat(' --- |', 1, numel(cols))];
last = '';
for r = 1:numel(t.rows)
    if ~strcmp(sections{r}, last)
        md{end+1} = ['| **' sections{r} '** |' repmat(' |', 1, numel(cols)-1)]; %#ok<AGROW>
        last = sections{r};
    end
    lead = {t.rows(r).network};
    if hasVariant, lead{end+1} = t.rows(r).variant; end %#ok<AGROW>
    md{end+1} = ['| ' strjoin([lead, grid(r,:)], ' | ') ' |']; %#ok<AGROW>
end
if ~isempty(notes)
    md{end+1} = '';
    for n = 1:numel(notes), md{end+1} = ['- ' notes{n}]; end %#ok<AGROW>
end
fmd = fullfile(outDir, [t.id '.md']);
writeText(fmd, strjoin(md, newline));

% ---------- LaTeX ----------
nc = numel(cols);
tex = {'\begin{table}[ht]', '\centering', sprintf('\\caption{%s}', texEsc(t.caption)), ...
       sprintf('\\label{tab:%s}', t.id), sprintf('\\begin{tabular}{%s}', repmat('l', 1, nc)), '\hline', ...
       [strjoin(cellfun(@texEsc, cols, 'UniformOutput', false), ' & ') ' \\'], '\hline'};
last = '';
for r = 1:numel(t.rows)
    if ~strcmp(sections{r}, last)
        tex{end+1} = sprintf('\\multicolumn{%d}{l}{\\textit{%s}} \\\\', nc, texEsc(sections{r})); %#ok<AGROW>
        last = sections{r};
    end
    lead = {t.rows(r).network};
    if hasVariant, lead{end+1} = t.rows(r).variant; end %#ok<AGROW>
    tex{end+1} = [strjoin(cellfun(@texEsc, [lead, grid(r,:)], 'UniformOutput', false), ' & ') ' \\']; %#ok<AGROW>
end
tex = [tex, {'\hline', '\end{tabular}', '\end{table}'}];
ftex = fullfile(outDir, [t.id '.tex']);
writeText(ftex, strjoin(tex, newline));
files = {fmd, ftex};
end

function writeText(f, s)
fid = fopen(f, 'w', 'n', 'UTF-8');
c = onCleanup(@() fclose(fid));
fwrite(fid, s, 'char');
end

function s = texEsc(s)
s = char(s);
s = strrep(s, '\', '\textbackslash{}');
for ch = {'&','%','#','_','$'}
    s = strrep(s, ch{1}, ['\' ch{1}]);
end
s = strrep(s, '—', '---');
s = strrep(s, '–', '--');
end

function s = upperFirst(s)
s = regexprep(s, '^table(\d+)$', 'Table $1');
end

function c = ternaryCell(cond, a, b)
if cond, c = a; else, c = b; end
end
