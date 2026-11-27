function T = smokeAssocLSTM(versions, fold, nIter)
%SMOKEASSOCLSTM  Build each version's real network and time a few training steps.
%
%   smokeAssocLSTM                       every version, fold 1, 3 steps
%   smokeAssocLSTM({'paper'}, 1, 5)
%
% Run from the root of a run tree (datasets/ and pretrainedModels/ as for
% runReplication). For each version: the fold's source net and target data
% (CWL C2, Fp1-Fz), the projection network as runReplication builds it, then
% nIter Adam steps on mini-batches of 197 through dlnetwork/dlfeval with the
% layer's hand-written backward. Reports the parameter count of the
% projection, seconds per step, the loss, and the largest L2 norm of each
% Associative LSTM gradient (single-precision clipping overflows above
% ~1.8e19, the reason the phase-aware layer needed gradCap).
if nargin < 1 || isempty(versions), versions = {assocVersions().id}; end
if nargin < 2, fold = 1; end
if nargin < 3, nIter = 3; end
versions = cellstr(versions);
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'common'));
addpath(genpath(fullfile(pwd, 'utils')));
S = load(fullfile('pretrainedModels', 'cwl', 'cwl1', 'DeepRNN', sprintf('sourceDomainClassifier_N2_DeepRNN_%d.mat', fold)));
D = load(fullfile('datasets', 'folds', 'CWL', 'NEC21_C2', 'targetdomain1', sprintf('target_fold%d.mat', fold)), 'xtrain', 'ytrain');
fprintf('[smoke] fold %d: %d training trials, %d x %d\n', fold, size(D.xtrain, 3), size(D.xtrain, 1), size(D.xtrain, 2));
useGPU = canUseGPU;
rows = {};
for k = 1:numel(versions)
    id = versions{k};
    model = struct('pretrainedNet', S.net, 'nchans', 2, 'signalLength', size(D.xtrain, 2), 'typeNet', 'DeepRNN');
    lg = get_pLayer_assocLSTM(model, id);
    lg = removeLayers(lg, lg.Layers(end).Name);            % classification output -> dlnetwork
    net = dlnetwork(lg);
    lrn = net.Learnables;
    isProj = startsWith(lrn.Layer, 'assoc');
    nProj = sum(cellfun(@(v) numel(v), lrn.Value(isProj)));
    cls = categories(categorical(D.ytrain));
    assocCapGrads('reset');
    avg = []; avgSq = []; tt = zeros(1, nIter); losses = zeros(1, nIter);
    for it = 1:nIter
        sel = mod((it-1)*197 + (0:196), size(D.xtrain, 3)) + 1;
        X = dlarray(single(D.xtrain(:,:,sel)), 'CTB');
        Yt = dlarray(single(onehotencode(categorical(D.ytrain(sel), str2double(cls)), 2).'), 'CB');
        if useGPU, X = gpuArray(X); Yt = gpuArray(Yt); end
        tic;
        [loss, grad] = dlfeval(@lossFn, net, X, Yt);
        [net, avg, avgSq] = adamupdate(net, grad, avg, avgSq, it, 0.0029);
        if useGPU, wait(gpuDevice); end
        tt(it) = toc; losses(it) = double(gather(extractdata(loss)));
    end
    N = assocCapGrads('stats');
    mx = max(N, [], 1);
    rows(end+1, :) = {id, nProj, median(tt), losses(1), losses(end), mx(1), mx(2), mx(3), mx(4)}; %#ok<AGROW>
    fprintf('[smoke] %-13s params %7d  %.2f s/step  loss %.3f -> %.3f  max|g| W %.2e U %.2e b %.2e omega %.2e\n', rows{end, :});
end
T = cell2table(rows, 'VariableNames', {'version','projParams','secPerStep','loss1','lossEnd','gW','gU','gb','gOmega'});
end

function [loss, grad] = lossFn(net, X, Yt)
Y = forward(net, X);
loss = crossentropy(Y, Yt);
grad = dlgradient(loss, net.Learnables);
end
