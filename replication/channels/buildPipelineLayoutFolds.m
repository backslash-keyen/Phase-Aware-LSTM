function info = buildPipelineLayoutFolds(varargin)
%BUILDPIPELINELAYOUTFOLDS  Build CWL target folds for other channel pairs with the pipeline's own generator.
%
% Extension study only. The paper's target folds (targetdomain1, Fp1-Fz) were
% made by M:\Earth\datasets\helpers\data_perfoldExtraction_NEC21.m, which
% calls dataFeeder_NEC21 with domain 'target-domain'. For fold i that class
% runs rng(i); p = randperm(13) and uses
%   source subjects p(1:7)  (train p(1:4),  val p(5:7))
%   target subjects p(8:13) (train p(8:11), val p(12:13))
%   test subjects   14, 15  (both domains)
% Each subject is z-scored and winsorized per channel over all 61 channels,
% then the selected channels are kept. The current M: copy of the class
% (1 Nov 2025) then downsamples by 5, but the paper's targetdomain1 folds
% (14 Oct 2025) have all 500 samples, so this function replays the
% generator's steps without that downsampling (feedFolds below), using the
% class's own static preprocess_givenData (read-only from M:). The new
% folds then have the target subjects, split and preprocessing of the
% paper's target folds, and the Fp1FzT isequal gate proves it.
%
% Order and safety:
%   1. prints the subject table for folds 1..ncval and asserts that the
%      source, target and test subjects are disjoint;
%   2. builds Fp1FzT ([1 2]) and checks with isequal that every variable of
%      every fold equals targetdomain1/target_fold<i>.mat. If any fold
%      differs it stops with an error and nothing else is built;
%   3. builds the other layouts in one pass over the union of their rows
%      (exact, since normalisation and winsorizing are per channel).
% Output: datasets/folds/CWL/NEC21_C2/targetlayout_<name>/target_fold<i>.mat
% (xtrain xval xtest ytrain yval ytest, -v7.3 like the pipeline, written via
% a tmp file + rename) and layout_info.csv with the subjects per fold.
%
%   addpath replication/channels
%   info = buildPipelineLayoutFolds();
%   info = buildPipelineLayoutFolds('layouts', {'FzPOzT', [2 46]});

p = inputParser;
addParameter(p, 'projectLoc', fileparts(fileparts(fileparts(mfilename('fullpath')))));
addParameter(p, 'earthLoc', 'M:\Earth');
addParameter(p, 'layouts', {'FzPOzT', [2 46]; 'AF3AF4T', [32 60]});
addParameter(p, 'ncats', 2);
addParameter(p, 'ncval', 10);
addParameter(p, 'force', false);
addParameter(p, 'prepOrder', 'A');   % A..F, see feedFolds (A = per-session prep over 61 ch, as in the M: class)
addParameter(p, 'tol', 0);           % gate tolerance on x (max abs diff); 0 = isequal. Labels are always isequal.
parse(p, varargin{:}); o = p.Results;

helpers = fullfile(o.earthLoc, 'datasets', 'helpers');
prep    = fullfile(o.earthLoc, 'datasets', 'preprocessing');
addpath(helpers, prep);                    % read-only use; nothing is written to M:
assert(exist('dataFeeder_NEC21', 'file') == 2, 'dataFeeder_NEC21 not found in %s', helpers);

config.projectLoc = o.earthLoc;
config.dataLoc = fullfile(o.earthLoc, 'datasets', 'matfiles', 'CWL', 'NEC21');
ec.ncval = o.ncval;
ec.sourceChannel = 1:61;
ec.dataset = 'CWL';
ec.ncategories = o.ncats;
ec.domain = 'target-domain';
ec.prepOrder = upper(o.prepOrder);
fprintf('[pipeline] prepOrder %s, gate tol %g\n', ec.prepOrder, o.tol);

base = fullfile(o.projectLoc, 'datasets', 'folds', 'CWL', sprintf('NEC21_C%d', o.ncats));
refDir = fullfile(base, 'targetdomain1');

% 1. subject table (same code as dataFeeder_NEC21.get_subjectIndices)
subj = subjectTable(o.ncval);
fprintf('[pipeline] fold | source train | source val | target train | target val | test\n');
for i = 1:o.ncval
    s = subj(i);
    fprintf('[pipeline] %4d | %-12s | %-10s | %-12s | %-10s | %s\n', i, num2str(s.srcTrain), num2str(s.srcVal), ...
        num2str(s.tgtTrain), num2str(s.tgtVal), num2str(s.test));
    src = [s.srcTrain s.srcVal]; tgt = [s.tgtTrain s.tgtVal];
    assert(isempty(intersect(src, tgt)) && isempty(intersect(src, s.test)) && isempty(intersect(tgt, s.test)), ...
        'buildPipelineLayoutFolds:subjects', 'Fold %d: source, target and test subjects overlap.', i);
end
fprintf('[pipeline] source, target and test subjects are disjoint in all %d folds\n', o.ncval);

% 2. Fp1FzT, checked against the paper's target folds
info = table();
info = [info; buildLayouts(config, ec, base, {'Fp1FzT', [1 2]}, subj, true)];   % always rebuilt, so the gate never checks stale files;
allOk = true; maxDiff = nan(o.ncval, 1);
for i = 1:o.ncval
    A = load(fullfile(base, 'targetlayout_Fp1FzT', sprintf('target_fold%d.mat', i)));
    R = load(fullfile(refDir, sprintf('target_fold%d.mat', i)));
    vars = {'xtrain','xval','xtest','ytrain','yval','ytest'};
    ok = true; md = 0;
    for v = vars
        a = A.(v{1}); r = R.(v{1});
        same = isequal(a, r) && strcmp(class(a), class(r));
        if isequal(size(a), size(r)) && strcmp(class(a), class(r))
            dv = max(abs(double(a(:)) - double(r(:)))); if isempty(dv), dv = 0; end
            if startsWith(v{1}, 'x'), md = max(md, dv); end
            if ~same && o.tol > 0 && startsWith(v{1}, 'x') && dv <= o.tol, same = true; end
            fprintf('[pipeline] fold %d: %s max abs diff %.3g\n', i, v{1}, dv);
        end
        clear a r
        if ~same
            fprintf('[pipeline] fold %d: %s differs (size %s vs %s, class %s vs %s)\n', i, v{1}, ...
                mat2str(size(A.(v{1}))), mat2str(size(R.(v{1}))), class(A.(v{1})), class(R.(v{1})));
        end
        ok = ok && same;
    end
    maxDiff(i) = md;
    fprintf('[pipeline] fold %d: Fp1FzT == targetdomain1 (all 6 variables, x tol %g): %s\n', i, o.tol, mat2str(ok));
    allOk = allOk && ok;
    clear A R
end
if ~allOk
    error('buildPipelineLayoutFolds:check', 'Fp1FzT does not reproduce targetdomain1; no other layout built.');
end
fprintf('[pipeline] CHECK PASSED: Fp1FzT reproduces targetdomain1 in all %d folds (x tol %g, max x diff %.3g)\n', o.ncval, o.tol, max(maxDiff));
csv = fullfile(base, 'targetlayout_Fp1FzT', 'layout_info.csv');
T = readtable(csv);
T.maxAbsDiffVsTargetdomain1 = maxDiff(T.fold); T.gateTol = repmat(o.tol, height(T), 1);
writetable(T, csv);

% 3. other layouts, one pass over the union of their rows
if ~isempty(o.layouts)
    info = [info; buildLayouts(config, ec, base, o.layouts, subj, o.force)];
end
fprintf('[pipeline] done\n');
end

function info = buildLayouts(config, ec, base, L, subj, force)
outDirs = cellfun(@(n) fullfile(base, ['targetlayout_' n]), L(:,1), 'UniformOutput', false);
todo = false(size(L,1), 1);
for j = 1:size(L,1)
    for i = 1:ec.ncval
        todo(j) = todo(j) || force || ~isfile(fullfile(outDirs{j}, sprintf('target_fold%d.mat', i)));
    end
end
rowsAll = unique([L{:,2}], 'stable');
if any(todo)
    fprintf('[pipeline] target-domain folds (Oct-2025 generator, no downsampling), channels %s\n', mat2str(rowsAll));
    feeds = feedFolds(config, ec, rowsAll, subj);
end
info = table();
for j = 1:size(L,1)
    if ~isfolder(outDirs{j}), mkdir(outDirs{j}); end
    [~, k] = ismember(L{j,2}, rowsAll);
    for i = 1:ec.ncval
        outFile = fullfile(outDirs{j}, sprintf('target_fold%d.mat', i));
        if todo(j) && (force || ~isfile(outFile))
            % same conversions as data_perfold_extractANDsave.m
            xtrain = feeds(i).xtrain(k,:,:); xval = feeds(i).xval(k,:,:); xtest = feeds(i).xtest(k,:,:); %#ok<NASGU>
            ytrain = double(feeds(i).ytrain); yval = double(feeds(i).yval); ytest = double(feeds(i).ytest); %#ok<NASGU>
            tmp = [outFile '.tmp.mat'];
            save(tmp, 'xtrain', 'xval', 'xtest', 'ytrain', 'yval', 'ytest', '-v7.3');
            movefile(tmp, outFile, 'f');
            clear xtrain xval xtest ytrain yval ytest
            fprintf('[pipeline] %s fold %d -> %s\n', L{j,1}, i, outFile);
        else
            fprintf('[pipeline] %s fold %d exists, skipped\n', L{j,1}, i);
        end
        s = subj(i);
        row = table(string(L{j,1}), i, L{j,2}(1), L{j,2}(2), string(num2str(s.tgtTrain)), string(num2str(s.tgtVal)), ...
            string(num2str(s.test)), string(num2str([s.srcTrain s.srcVal])), string(ec.prepOrder), ...
            'VariableNames', {'layout','fold','row1','row2','targetTrainSubjects','targetValSubjects','testSubjects','sourceSubjects','prepOrder'});
        info = [info; row]; %#ok<AGROW>
    end
    writetable(info(info.layout == L{j,1}, :), fullfile(outDirs{j}, 'layout_info.csv'));
end
clear feeds
end

function feeds = feedFolds(config, ec, rows, subj)
% Same steps as dataFeeder_NEC21.get_dataset / load_singleSubject, as the
% generator was when targetdomain1 was written (14 Oct 2025): the current
% M: copy (1 Nov 2025) adds "x = pagetranspose(downsample(pagetranspose(x),5))",
% which the 500-sample paper folds do not have. Per-subject preprocessing
% uses the class's own static preprocess_givenData. Each subject file is
% loaded once and cached (selected rows only); the isequal check against
% targetdomain1 proves the replica.
cache = containers.Map('KeyType', 'double', 'ValueType', 'any');
    function [x, y] = subjectData(s)
        if ~isKey(cache, s)
            fprintf('[pipeline] loading subject %d\n', s);
            d = load(fullfile(config.dataLoc, sprintf('subject_%d', s)), 't_trainingset', 't_testset', 'trainingbin', 'testbin');
            P = @(x) dataFeeder_NEC21.preprocess_givenData(x);
            ya = d.trainingbin; yb = d.testbin; y0 = cat(1, ya, yb);
            drop = ec.ncategories == 2;
            switch ec.prepOrder
                case 'A'   % per-session prep over 61 ch, select rows, pool, drop class 1 (M: class)
                    a = P(d.t_trainingset); b = P(d.t_testset); xs = cat(3, a(rows,:,:), b(rows,:,:));
                    if drop, xs(:,:,y0==1) = []; end
                case 'B'   % pool sessions, prep, select, drop
                    ab = P(cat(3, d.t_trainingset, d.t_testset)); xs = ab(rows,:,:);
                    if drop, xs(:,:,y0==1) = []; end
                case 'C'   % per session: drop class 1, prep; then pool and select
                    a = d.t_trainingset; b = d.t_testset;
                    if drop, a = a(:,:,ya~=1); b = b(:,:,yb~=1); end
                    xs = cat(3, P(a), P(b)); xs = xs(rows,:,:);
                case 'D'   % pool, drop class 1, prep, select
                    ab = cat(3, d.t_trainingset, d.t_testset);
                    if drop, ab = ab(:,:,y0~=1); end
                    xs = P(ab); xs = xs(rows,:,:);
                case 'E'   % no prep
                    xs = cat(3, d.t_trainingset(rows,:,:), d.t_testset(rows,:,:));
                    if drop, xs(:,:,y0==1) = []; end
                case 'F'   % A, then prep again on the pooled selected rows
                    a = P(d.t_trainingset); b = P(d.t_testset); xs = cat(3, a(rows,:,:), b(rows,:,:));
                    if drop, xs(:,:,y0==1) = []; end
                    xs = P(xs);
                otherwise
                    error('buildPipelineLayoutFolds:prepOrder', 'Unknown prepOrder %s', ec.prepOrder);
            end
            if drop, y0(y0==1) = []; end
            clear d a b ab
            cache(s) = struct('x', xs, 'y', categorical(y0));
        end
        c = cache(s); x = c.x; y = c.y;
    end
    function [x, y] = pooled(subs)
        x = []; y = [];
        for s = subs
            [xt, yt] = subjectData(s);
            x = cat(3, x, xt); y = cat(1, y, yt);
        end
    end
for i = 1:ec.ncval
    [feeds(i).xtrain, feeds(i).ytrain] = pooled(subj(i).tgtTrain); %#ok<AGROW>
    [feeds(i).xval,   feeds(i).yval]   = pooled(subj(i).tgtVal);   %#ok<AGROW>
    [feeds(i).xtest,  feeds(i).ytest]  = pooled(subj(i).test);     %#ok<AGROW>
end
end

function subj = subjectTable(ncval)
% copy of dataFeeder_NEC21.get_subjectIndices (13 pooled subjects, 14-15 reserved for test)
totalSubs = 13;
for i = 1:ncval
    rng(i);
    pp = randperm(totalSubs, totalSubs);
    subj(i).srcTrain = pp(1:4); subj(i).srcVal = pp(5:7);   %#ok<AGROW> floor(7*0.7)=4, ceil(7*0.3)=3
    subj(i).tgtTrain = pp(8:11); subj(i).tgtVal = pp(12:13); %#ok<AGROW> floor(6*0.7)=4, ceil(6*0.3)=2
    subj(i).test = [14 15]; %#ok<AGROW>
end
end
