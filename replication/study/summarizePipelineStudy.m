function T = summarizePipelineStudy(varargin)
%SUMMARIZEPIPELINESTUDY  Table of the 5-fold pipeline study from fold_log.csv.
%
%   T = summarizePipelineStudy()                 % prints markdown, returns table
%   summarizePipelineStudy('out', 'replication/results/pipeline_study.md')
%
% Per job: mean +- SD of val and test accuracy over the folds present (1-5),
% number of collapsed folds (val and test both within 1 point of 50%,
% i.e. one class), and the paired test difference to cell 'base' with the
% number of folds it wins. The last row for a (job, fold) in the log counts.
p = inputParser;
addParameter(p, 'log', fullfile(fileparts(fileparts(mfilename('fullpath'))), 'results', 'fold_log.csv'));
addParameter(p, 'folds', 1:5);
addParameter(p, 'experiment', 'cwl_N2_target-domain_DeepRNN_phase-lstm');
addParameter(p, 'out', '');
parse(p, varargin{:}); o = p.Results;

addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'versions', 'fast', 'common'));
L = readtable(o.log, 'TextType', 'string', 'Delimiter', ',');
L = L(ismember(L.fold, o.folds) & startsWith(L.id, o.experiment + "__"), :);
L.job = extractAfter(L.id, o.experiment + "__");
[~, last] = unique(L(:, {'job','fold'}), 'last');
L = L(last, :);

C = pipelineStudyCells();
jobs   = [strcat("pv_ps_", string({C.id})), "pv_piSign_none", "pv_piSign_sum", "pv_piSign_realProj", "pv_piSign_tanhRe", "notrigFastPhaseLSTM"];
factor = [string({C.factor}), repmat("output sign", 1, 4), "paper layer"];
label  = [string({C.label}), "unsigned (paper eq. 7)", "sign(Re+Im)", "real projection cos(phi)", "tanh(Re c)", "notrig as in Nov 2025"];

B = L(L.job == "pv_ps_base", :);
rows = {};
for j = 1:numel(jobs)
    R = L(L.job == jobs(j), :);
    if isempty(R), continue; end
    col = abs(R.accVal - 0.5) < 0.01 & abs(R.accTest - 0.5) < 0.01;
    [tf, ib] = ismember(R.fold, B.fold);
    d = 100 * (R.accTest(tf) - B.accTest(ib(tf)));
    rows(end+1, :) = {factor(j), jobs(j), label(j), height(R), ...
        100*mean(R.accVal), 100*std(R.accVal), 100*mean(R.accTest), 100*std(R.accTest), ...
        nnz(col), mean(d), nnz(d > 0), numel(d), sum(R.seconds)/60/height(R)}; %#ok<AGROW>
end
T = cell2table(rows, 'VariableNames', {'factor','job','setting','nFolds','valMean','valSD', ...
    'testMean','testSD','collapsed','dTestVsBase','winsVsBase','nPaired','minPerFold'});

lines = ["| Factor | Setting | Job | Folds | Val | Test | Collapsed | Test vs base (wins) |", ...
         "|---|---|---|---|---|---|---|---|"];
for i = 1:height(T)
    lines(end+1) = sprintf('| %s | %s | %s | %d | %.1f ± %.1f | %.1f ± %.1f | %d/%d | %+.1f (%d/%d) |', ...
        T.factor(i), T.setting(i), T.job(i), T.nFolds(i), T.valMean(i), T.valSD(i), ...
        T.testMean(i), T.testSD(i), T.collapsed(i), T.nFolds(i), T.dTestVsBase(i), ...
        T.winsVsBase(i), T.nPaired(i)); %#ok<AGROW>
end
fprintf('%s\n', lines{:});
fprintf('\nPer-fold test accuracy (%%):\n');
P = unstack(L(:, {'job','fold','accTest'}), 'accTest', 'fold');
disp(P);
if ~isempty(o.out)
    fid = fopen(o.out, 'w'); fprintf(fid, '%s\n', lines{:}); fclose(fid);
    writetable(L(:, {'job','fold','accTrain','accVal','accTest','seconds','finishedAt'}), replace(o.out, '.md', '_folds.csv'));
end
end
