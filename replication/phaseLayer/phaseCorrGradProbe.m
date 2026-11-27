function R = phaseCorrGradProbe(varargin)
%PHASECORRGRADPROBE  Raw phase-layer gradient norms vs sequence length, per version (trains nothing).
%
%   addpath('replication/phaseLayer'); R = phaseCorrGradProbe();
%   R = phaseCorrGradProbe('fold', 2, 'T', [50 100 200 500], 'ids', {'port','complexH'})
%
% Run from the repository root. For each version: the layer alone at its
% initialisation (seed = fold, as the fold's training starts), 32 target
% training trials of the fold, single precision (as trainNetwork), loss =
% sum(Y .* w) with fixed random w. Prints the L2 norm (computed in double)
% of each learnable's gradient before any cap and whether it is finite.
% A norm above ~1.8e19 cannot be clipped by trainNetwork's single-precision
% l2norm clip (its square overflows), which zeroes the update.
p = inputParser;
addParameter(p, 'fold', 2);
addParameter(p, 'T', [50 100 200 500]);
addParameter(p, 'ids', {});
addParameter(p, 'nTrials', 32);
parse(p, varargin{:});
o = p.Results;
root = pwd;
addpath(genpath(fullfile(root, 'utils')));
addpath(fullfile(fileparts(mfilename('fullpath')), 'common'));
list = phaseCorrVersions();
ids = o.ids; if isempty(ids), ids = {list.id}; end
S = load(fullfile(root, 'datasets', 'folds', 'CWL', 'NEC21_C2', 'targetdomain1', sprintf('target_fold%d.mat', o.fold)), 'xtrain');
X0 = single(S.xtrain(:, :, 1:o.nTrials)); clear S
names = {'W1','W2','U','Q','b1','b2'};
R = struct('id', {}, 'T', {}, 'norms', {});
fprintf('%-12s %4s  %s\n', 'version', 'T', strjoin(compose('%9s', string(names)), ' '));
for k = 1:numel(ids)
    cfg = phaseCorrVersions(ids{k}); cfg.gradCap = Inf;
    rng(o.fold);
    net = dlnetwork([sequenceInputLayer(2); phaseAwareLSTMCorr(61, 'Name', 'phaseLSTM', 'Version', ids{k}, 'Cfg', cfg)]);
    if canUseGPU, net = dlupdate(@gpuArray, net); end
    for T = o.T
        X = dlarray(permute(X0(:, 1:T, :), [1 3 2]), 'CBT');
        rng(0); w = single(randn(61, o.nTrials, T));
        if canUseGPU, X = gpuArray(X); w = gpuArray(w); end
        phaseCorrCapGrads('reset');
        dlfeval(@lg, net, X, w);
        n = phaseCorrCapGrads('stats');
        n = n(end, :);
        R(end+1) = struct('id', ids{k}, 'T', T, 'norms', n); %#ok<AGROW>
        fprintf('%-12s %4d  %s\n', ids{k}, T, strjoin(compose('%9.2e', n), ' '));
    end
end
end

function lg(net, X, w)
Y = forward(net, X);
loss = sum(stripdims(Y) .* w, 'all') / size(w, 3);
g = dlgradient(loss, net.Learnables); %#ok<NASGU>
end
