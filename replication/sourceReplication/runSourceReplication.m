function T = runSourceReplication(stage, varargin)
%RUNSOURCEREPLICATION  Retrain the CWL C2 DeepRNN source models the way the paper's runs did.
%
%   T = runSourceReplication(1)            % stage 1: fold 1, trained twice, determinism check
%   T = runSourceReplication(2)            % stage 2: all 10 folds (resumes after stage 1)
%   T = runSourceReplication(1, 'id', 'cwl_N2_source-domain-2_DeepRNN_deeprnn', 'stem', 'sourceDomain2Classifier_N2_DeepRNN')
%                                          % the same for the dual-channel source
%   Name-value options
%     id        'cwl_N2_source-domain_DeepRNN_deeprnn'   experiment id (runReplication 'listOnly')
%     stem      'sourceDomainClassifier_N2_DeepRNN'      saved model file name stem
%     fold      1        stage-1 fold
%     rngType   'threefry'  generator of the Nov 2025 paper runs (see README)
%     refDir    ''       folder with reference nets to compare against, e.g. the
%                        HPC nets from extractHPCSourceNets.m or the authors'
%                        L:\plstm\pretrainedModels\cwl\cwl1\DeepRNN (read only)
%
%   Run from the repository root of a scratch copy: models are written to
%   pretrainedModels/cwl/cwl1/DeepRNN and results to cvalresults/, as in the
%   original code. Stage 1 keeps both stage-1 models in _runs/stage1_{a,b}/.

p = inputParser;
addParameter(p, 'id',      'cwl_N2_source-domain_DeepRNN_deeprnn');
addParameter(p, 'stem',    'sourceDomainClassifier_N2_DeepRNN');
addParameter(p, 'fold',    1);
addParameter(p, 'rngType', 'threefry');
addParameter(p, 'refDir',  '');
parse(p, varargin{:});
o = p.Results;

modelDir = fullfile('pretrainedModels', 'cwl', 'cwl1', 'DeepRNN');
T = table();
switch stage
    case 1
        f = o.fold;
        runs = {'a','b'};
        for r = 1:2
            runReplication('ids', {o.id}, 'folds', f, 'force', true, 'rngType', o.rngType);
            d = fullfile('_runs', ['stage1_' runs{r}]);
            if ~exist(d, 'dir'), mkdir(d); end
            copyfile(fullfile(modelDir, sprintf('%s_%d.mat', o.stem, f)), d);
        end
        a = fullfile('_runs', 'stage1_a', sprintf('%s_%d.mat', o.stem, f));
        b = fullfile('_runs', 'stage1_b', sprintf('%s_%d.mat', o.stem, f));
        C = compareSourceNets(a, b);
        fprintf('[stage 1] fold %d, run a vs run b: %d of %d parameters bit-identical, max |diff| %.3g\n', ...
            f, nnz(C.identical), height(C), max(C.maxAbs));
        T = summarise({a}, o.refDir, o.stem, f);
    case 2
        runReplication('ids', {o.id}, 'folds', 1:10, 'rngType', o.rngType);
        files = arrayfun(@(i) fullfile(modelDir, sprintf('%s_%d.mat', o.stem, i)), 1:10, 'UniformOutput', false);
        T = summarise(files, o.refDir, o.stem, 1:10);
    otherwise
        error('runSourceReplication:stage', 'stage must be 1 or 2.');
end
end

function T = summarise(files, refDir, stem, folds)
fold = folds(:); identicalToRef = nan(numel(folds),1); lstmCosine = nan(numel(folds),1);
for k = 1:numel(folds)
    if isempty(refDir), continue; end
    ref = fullfile(refDir, sprintf('%s_%d.mat', stem, folds(k)));
    if ~isfile(ref), continue; end
    C = compareSourceNets(files{k}, ref);
    identicalToRef(k) = all(C.identical);
    lstmCosine(k) = C.cosine(find(contains(C.parameter, 'InputWeights'), 1));
end
T = table(fold, identicalToRef, lstmCosine);
disp(T)
end
