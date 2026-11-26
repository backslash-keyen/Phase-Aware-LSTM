function info = buildTargetLayoutFolds(varargin)
%BUILDTARGETLAYOUTFOLDS  Build CWL target folds for other two-channel layouts.
%
% EXTENSION STUDY, not part of the paper replication. The paper's CWL target
% layouts (Fp1-Fz, F3-F7) are rows 1-2 and 3-4 of the source montage. This
% builds other two-channel layouts from the SAME full-montage source folds:
% same trials, same train/val/test split, same labels; only the two channel
% rows differ. Nothing is changed in the source folds; output goes to
%   datasets/folds/CWL/NEC21_C<n>/targetlayout_<name>/target_fold<i>.mat
% and is read by the opt-in variants tl_<name> (variants/targetLayout).
%
%   info = buildTargetLayoutFolds()                       % default layouts
%   info = buildTargetLayoutFolds('layouts', {'FzPz',{'Fz','Pz'}}, 'folds', 2)
%   info = buildTargetLayoutFolds('checkOnly', true)      % labels + slice check only
%
% Options
%   labels     cellstr of the source montage, one per row of xtrain (61 for
%              NEC21). Default: read from channels/nec21_channels.txt.
%   layouts    {name, {ch1, ch2}; ...}. Default FzPz, FzPOz, AF3AF4, FzP4.
%   ncats      2 (default) or 3
%   folds      [] = every source_fold*.mat present
%   check      true (default): before writing, require that the existing
%              targetdomain1/target_fold<i>.mat equals rows {Fp1, Fz} of the
%              source fold exactly. This checks the label file AND that the
%              paper's target folds are plain slices of the source folds.
%   checkOnly  true: run the check only, write nothing
%   force      false (default): skip folds whose output already exists
%   ramGuardGB 0.5: stop if less RAM than this is free before a load
%
% Each written file has xtrain/xval/xtest (2 x T x N, same class as the
% source) and ytrain/yval/ytest unchanged, saved as -v7 like the originals,
% plus layout_info.csv in the output folder (channels, rows, source SHA-256).

p = inputParser;
addParameter(p, 'projectLoc', fileparts(fileparts(fileparts(mfilename('fullpath')))));
addParameter(p, 'labels', {});
addParameter(p, 'layouts', {'FzPz', {'Fz','Pz'}; 'FzPOz', {'Fz','POz'}; 'AF3AF4', {'AF3','AF4'}; 'FzP4', {'Fz','P4'}});
addParameter(p, 'ncats', 2);
addParameter(p, 'folds', []);
addParameter(p, 'check', true);
addParameter(p, 'checkOnly', false);
addParameter(p, 'force', false);
addParameter(p, 'ramGuardGB', 0.5);
parse(p, varargin{:});
o = p.Results;

labels = o.labels;
if isempty(labels)
    labels = readLabels(fullfile(fileparts(mfilename('fullpath')), 'nec21_channels.txt'));
end
labels = cellstr(labels(:));
base = fullfile(o.projectLoc, 'datasets', 'folds', 'CWL', sprintf('NEC21_C%d', o.ncats));
srcDir = fullfile(base, 'fullchannels');
refDir = fullfile(base, 'targetdomain1');

d = dir(fullfile(srcDir, 'source_fold*.mat'));
folds = sort(cellfun(@(s) str2double(regexp(s, '\d+', 'match', 'once')), {d.name}));
if ~isempty(o.folds), folds = intersect(folds, o.folds); end
if isempty(folds), error('buildTargetLayoutFolds:noData', 'No source folds in %s', srcDir); end

% resolve every layout before touching data
L = o.layouts;
rows = zeros(size(L, 1), 2);
for j = 1:size(L, 1)
    for c = 1:2
        r = find(strcmpi(labels, L{j,2}{c}));
        if numel(r) ~= 1
            error('buildTargetLayoutFolds:channel', 'Channel %s not found once in the montage (layout %s).', L{j,2}{c}, L{j,1});
        end
        rows(j, c) = r;
    end
end
refRows = [find(strcmpi(labels, 'Fp1')), find(strcmpi(labels, 'Fz'))];

info = table();
for i = folds
    ramGuard(o.ramGuardGB);
    srcFile = fullfile(srcDir, sprintf('source_fold%d.mat', i));
    S = load(srcFile, 'xtrain', 'xval', 'xtest', 'ytrain', 'yval', 'ytest');
    if size(S.xtrain, 1) ~= numel(labels)
        error('buildTargetLayoutFolds:labels', '%d labels but %d channels in %s', numel(labels), size(S.xtrain, 1), srcFile);
    end
    if o.check || o.checkOnly
        R = load(fullfile(refDir, sprintf('target_fold%d.mat', i)));
        ok = numel(refRows) == 2;
        for s = {'train','val','test'}
            ok = ok && isequal(S.(['x' s{1}])(refRows,:,:), R.(['x' s{1}])) && isequal(S.(['y' s{1}]), R.(['y' s{1}]));
        end
        fprintf('[layouts] fold %d: targetdomain1 == source rows {Fp1,Fz} = [%s]: %s\n', i, num2str(refRows), mat2str(ok));
        ref2 = fullfile(base, 'targetdomain2', sprintf('target_fold%d.mat', i));
        if ok && isfile(ref2)       % second label check where the F3-F7 folds exist
            R2 = load(ref2, 'xtest');
            r2 = [find(strcmpi(labels, 'F3')), find(strcmpi(labels, 'F7'))];
            ok = numel(r2) == 2 && isequal(S.xtest(r2,:,:), R2.xtest);
            fprintf('[layouts] fold %d: targetdomain2 test == source rows {F3,F7} = [%s]: %s\n', i, num2str(r2), mat2str(ok));
            clear R2
        end
        if ~ok
            error('buildTargetLayoutFolds:check', ['Fold %d: targetdomain1 is not rows {Fp1,Fz} of the source fold. ' ...
                  'Either the label file is wrong or the target folds are not slices; nothing written.'], i);
        end
        clear R
    end
    if o.checkOnly, clear S; continue; end
    sha = fileSha(srcFile);
    for j = 1:size(L, 1)
        outDir = fullfile(base, ['targetlayout_' L{j,1}]);
        if ~isfolder(outDir), mkdir(outDir); end
        outFile = fullfile(outDir, sprintf('target_fold%d.mat', i));
        if isfile(outFile) && ~o.force
            fprintf('[layouts] %s fold %d exists, skipped\n', L{j,1}, i);
        else
            xtrain = S.xtrain(rows(j,:),:,:); xval = S.xval(rows(j,:),:,:); xtest = S.xtest(rows(j,:),:,:); %#ok<NASGU>
            ytrain = S.ytrain; yval = S.yval; ytest = S.ytest; %#ok<NASGU>
            tmp = [outFile '.tmp.mat'];
            save(tmp, 'xtrain', 'xval', 'xtest', 'ytrain', 'yval', 'ytest', '-v7');
            movefile(tmp, outFile, 'f');
            clear xtrain xval xtest ytrain yval ytest
            fprintf('[layouts] %s fold %d -> %s\n', L{j,1}, i, outFile);
        end
        row = table(string(L{j,1}), i, string(L{j,2}{1}), string(L{j,2}{2}), rows(j,1), rows(j,2), string(sha), ...
            'VariableNames', {'layout','fold','ch1','ch2','row1','row2','sourceSha256'});
        info = [info; row]; %#ok<AGROW>
    end
    clear S
end
for j = 1:size(L, 1)
    outDir = fullfile(base, ['targetlayout_' L{j,1}]);
    if isfolder(outDir)
        writetable(info(info.layout == L{j,1}, :), fullfile(outDir, 'layout_info.csv'));
    end
end
end

function labels = readLabels(file)
if ~isfile(file)
    error('buildTargetLayoutFolds:labels', 'No label file %s. Pass ''labels'' or create it (one label per line, source row order).', file);
end
t = strtrim(splitlines(fileread(file)));
t = t(~cellfun(@isempty, t) & ~startsWith(t, '%'));
labels = t;
end

function ramGuard(gb)
if ~ispc, return; end
[~, sys] = memory;
freeGB = sys.PhysicalMemory.Available / 2^30;
if freeGB < gb
    error('buildTargetLayoutFolds:ram', 'Only %.2f GB RAM free (guard %.2f GB); stopping.', freeGB, gb);
end
end

function h = fileSha(file)
md = java.security.MessageDigest.getInstance('SHA-256');
fid = fopen(file, 'r'); c = onCleanup(@() fclose(fid));
while true
    b = fread(fid, 2^24, '*uint8');
    if isempty(b), break; end
    md.update(b);
end
h = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'))', 1, []));
end
