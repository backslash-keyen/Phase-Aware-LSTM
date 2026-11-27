function T = checkShuffleOrder(foldFile, refNetFile)
%CHECKSHUFFLEORDER  Which CPU generator reproduces a reference run's trial order?
%
%   T = checkShuffleOrder(foldFile, refNetFile)
%
%   makeDatastore.m calls rng(1) and then shuffles the train, validation and
%   test datastores. rng(1) keeps the current generator type, so the order
%   depends on the generator that was active (runReplication's rngType). The
%   reference net (an HPC net from extractHPCSourceNets.m, or any source net
%   saved by the pipeline) stores its shuffled validation set in
%   options.ValidationData. This rebuilds the shuffle for several generators
%   and reports which one reproduces that order exactly.
%
%   Example:
%     checkShuffleOrder('datasets/folds/CWL/NEC21_C2/fullchannels/source_fold1.mat', ...
%                       'pretrainedModels_hpc/cwl/cwl1/DeepRNN/sourceDomainClassifier_N2_DeepRNN_1.mat')
%   The Nov 2025 paper runs match 'threefry' only.

D = load(foldFile, 'xtrain', 'xval');
R = load(refNetFile, 'options');
ref = readall(R.options.ValidationData);
refX = cat(3, ref{:,1});                         % [chan x time x N] in the reference order
nTrain = size(D.xtrain, 3); nVal = size(D.xval, 3);
[~, refIdx] = ismember(trialKeys(refX), trialKeys(D.xval), 'rows');
assert(all(refIdx > 0), 'checkShuffleOrder:trials', 'The reference validation trials are not the fold file''s validation trials.');

gens = {'twister','threefry','philox','combRecursive','simdTwister','multFibonacci'};
match = false(numel(gens),1);
for g = 1:numel(gens)
    rng(1, gens{g});
    shuffle(indexStore(nTrain));                 % makeDatastore shuffles the training set first
    s = shuffle(indexStore(nVal));
    c = readall(s);
    order = double(squeeze(cat(3, c{:,1})));
    match(g) = isequal(order(:), refIdx(:));
end
T = table(gens(:), match, 'VariableNames', {'generator','reproducesReferenceOrder'});
disp(T)
end

function ds = indexStore(N)
ds = combine(arrayDatastore(reshape(single(1:N),1,1,N), 'IterationDimension', 3, 'ReadSize', 1), ...
             arrayDatastore(categorical(mod(1:N,2))', 'IterationDimension', 1));
end

function k = trialKeys(x)
% a few summary numbers per trial, enough to tell trials apart exactly
x = double(x);
k = [squeeze(sum(sum(x,1),2)), squeeze(x(1,1,:)), squeeze(x(end,end,:)), squeeze(sum(sum(x.^2,1),2))];
end
