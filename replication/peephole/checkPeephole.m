function nFail = checkPeephole(varargin)
%CHECKPEEPHOLE  Correctness and speed check of peepholeLSTMLayer (trains almost nothing, a few minutes).
%
%   addpath('replication/peephole'); nFail = checkPeephole();
%   checkPeephole('gpu', false, 'net', false)
%
% Run from the repository root.
% 1. Every version against an independent plain-dlarray implementation of the
%    same equations differentiated by dlgradient (double, CPU): output and every
%    learnable gradient, OutputMode sequence and last. Truncated versions:
%    the reference cuts the graph exactly where the paper truncates (h_{t-1}
%    into the gates and cell input, c into the peepholes); only the CEC path
%    f .* c_{t-1} stays connected.
% 2. 'ideal' without peepholes and with lstmLayer's weights = lstmLayer:
%    output and gradients (built-in layer through dlgradient).
% 3. Initial values of each version (biases, peepholes, CIFG rows); whether
%    'glorot' draws the same weights as lstmLayer from the same seed (info).
% 4. GPU single precision vs CPU double for two versions (if a GPU is there).
% 5. With 'net' (default true, needs pretrainedModels/.../sourceDomainClassifier_N2_DeepRNN_1.mat):
%    the projection network for every version builds through
%    define_layerArchitecture with the variant on the path, and a 2-epoch
%    trainNetwork run on random data at the real size (61 units, T = 500)
%    finishes with finite loss; time per iteration is printed.
p = inputParser;
addParameter(p, 'gpu', canUseGPU);
addParameter(p, 'net', true);
parse(p, varargin{:});
o = p.Results;

root = pwd;
here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root, 'utils')));
addpath(fullfile(here, 'common'));
nFail = 0;
H = 5; C = 2; B = 3; T = 14;
rng(7); Xn = randn(C, B, T); Wn = randn(H, B, T);
tolF = 1e-10; tolG = 1e-8;

% ---- 1. every version vs plain dlarray -------------------------------------
list = peepholeVersions();
for k = 1:numel(list)
    cfg = list(k).cfg;
    for mode = ["sequence", "last"]
        rng(1);
        net = dlnetwork([sequenceInputLayer(C); peepholeLSTMLayer(H, 'Name', 'p', 'OutputMode', mode, 'Version', list(k).id)]);
        net = dlupdate(@double, net);
        rng(3);   % moderate random weights so every term matters (keep fixed zeros fixed)
        Lt = net.Learnables;
        for j = 1:height(Lt)
            v = extractdata(Lt.Value{j});
            mask = true(size(v));
            if strcmp(Lt.Parameter{j}, 'p') && ~cfg.peep, mask(:) = false; end
            if cfg.cifg && any(strcmp(Lt.Parameter{j}, {'W','R','b'})), mask(1:H,:) = false; end
            if cfg.cifg && strcmp(Lt.Parameter{j}, 'p'), mask(1:H) = false; end
            Lt.Value{j} = dlarray(v + 0.4*randn(size(v)).*mask);
        end
        net.Learnables = Lt;
        X = dlarray(Xn, 'CBT');
        w = Wn; if mode == "last", w = Wn(:,:,end); end
        [y1, g1] = dlfeval(@lossLayer, net, X, w);
        [y2, g2] = dlfeval(@lossRef, net.Learnables, X, w, cfg, H, mode);
        dY = maxRel(y1, y2); dG = maxGradRel(g1, g2);
        nFail = nFail + ~report(sprintf('%-12s %-8s vs dlgradient reference: output %.1e, grads %.1e', ...
            list(k).id, mode, dY, dG), dY < tolF && dG < tolG);
    end
end

% ---- 2. ideal without peepholes = lstmLayer ---------------------------------
cfg = peepholeVersions('ideal'); cfg.peep = false;
for mode = ["sequence", "last"]
    rng(5);
    netL = dlnetwork([sequenceInputLayer(C); lstmLayer(H, 'Name', 'p', 'OutputMode', mode)]);
    netL = dlupdate(@double, netL);
    netL = dlupdate(@(v) v + 0.3*randn(size(v)), netL);
    lay = netL.Layers(2);
    P = peepholeLSTMLayer(H, 'Name', 'p', 'OutputMode', mode, 'Version', 'ideal-nopeep', 'Cfg', cfg);
    P.W = lay.InputWeights; P.R = lay.RecurrentWeights; P.b = lay.Bias; P.p = zeros(3*H, 1);
    netP = dlnetwork([sequenceInputLayer(C); P]);
    netP = dlupdate(@double, netP);
    X = dlarray(Xn, 'CBT');
    w = Wn; if mode == "last", w = Wn(:,:,end); end
    [yL, gL] = dlfeval(@lossLayer, netL, X, w);
    [yP, gP] = dlfeval(@lossLayer, netP, X, w);
    gP = gP(~strcmp(netP.Learnables.Parameter, 'p'));
    dY = maxRel(yL, yP); dG = maxGradRel(gL, gP);
    % the built-in layer agrees to single precision only (it rounds internally)
    nFail = nFail + ~report(sprintf('ideal, peepholes off = lstmLayer (%s): output %.1e, grads %.1e', mode, dY, dG), ...
        dY < 1e-5 && dG < 1e-5);
end

% ---- 3. initial values ------------------------------------------------------
for k = 1:numel(list)
    cfg = list(k).cfg;
    rng(1);
    net = dlnetwork([sequenceInputLayer(C); peepholeLSTMLayer(H, 'Name', 'p', 'Version', list(k).id)]);
    L = net.Layers(2);
    b = double(extractdataIf(L.b)); pp = double(extractdataIf(L.p));
    W = double(extractdataIf(L.W)); R = double(extractdataIf(L.R));
    switch cfg.init
        case 'gers'
            eb = [0; -2; 2];
            okW = all(abs([W(:); R(:); b(2*H+1:3*H)]) <= 0.1) && (~cfg.peep || all(abs(pp) <= 0.1 & pp ~= 0));
        case 'glorot'
            eb = [0; 1; 0];
            okW = all(pp == 0);
    end
    if ~isnan(cfg.forgetBias), eb(2) = cfg.forgetBias; end
    if cfg.cifg, eb(1) = 0; end
    okB = all(b(1:H) == eb(1)) && all(b(H+1:2*H) == eb(2)) && all(b(3*H+1:4*H) == eb(3));
    okP = cfg.peep || all(pp == 0);
    okC = ~cfg.cifg || (all(W(1:H,:) == 0, 'all') && all(R(1:H,:) == 0, 'all') && all(pp(1:H) == 0));
    nFail = nFail + ~report(sprintf('%-12s init: biases i/f/o %g/%g/%g, weights, peepholes, CIFG rows', ...
        list(k).id, eb), okB && okW && okP && okC);
end
rng(1); nL = dlnetwork([sequenceInputLayer(C); lstmLayer(H)]);
rng(1); nP = dlnetwork([sequenceInputLayer(C); peepholeLSTMLayer(H, 'Version', 'ideal')]);
same = isequal(extractdataIf(nL.Layers(2).InputWeights), extractdataIf(nP.Layers(2).W)) && ...
       isequal(extractdataIf(nL.Layers(2).RecurrentWeights), extractdataIf(nP.Layers(2).R));
fprintf('[info] ideal (glorot) draws the same W and R as lstmLayer from the same seed: %d\n', same);

% ---- 4. GPU single vs CPU double --------------------------------------------
if o.gpu
    for id = ["gers2002", "ideal"]
        rng(1);
        net = dlnetwork([sequenceInputLayer(C); peepholeLSTMLayer(H, 'Name', 'p', 'Version', id)]);
        X = dlarray(Xn, 'CBT');
        [yc, gc] = dlfeval(@lossLayer, dlupdate(@double, net), X, Wn);
        [yg, gg] = dlfeval(@lossLayer, dlupdate(@(v) gpuArray(single(v)), net), dlarray(gpuArray(single(Xn)), 'CBT'), gpuArray(single(Wn)));
        dY = maxRel(yc, gather(yg)); dG = maxGradRel(gc, cellfun(@(v) gather(v), gg, 'UniformOutput', false));
        nFail = nFail + ~report(sprintf('%s GPU single vs CPU double: output %.1e, grads %.1e', id, dY, dG), dY < 1e-4 && dG < 1e-3);
    end
end

% ---- 5. real network, trainNetwork -------------------------------------------
if o.net
    src = fullfile(root, 'pretrainedModels', 'cwl', 'cwl1', 'DeepRNN', 'sourceDomainClassifier_N2_DeepRNN_1.mat');
    if ~isfile(src)
        fprintf('[skip] no %s: network check skipped\n', src);
    else
        S = load(src, 'net');
        model = struct('pretrainedNet', S.net, 'nchans', 2, 'signalLength', 500, 'typeNet', 'DeepRNN');
        config = struct('domain', 'target-domain', 'layerName', 'lstm', 'typeNet', 'DeepRNN', 'datasetType', 'CWL');
        rng(2); Xtr = arrayfun(@(~) single(randn(2, 500)), 1:40, 'UniformOutput', false)'; Ytr = categorical(randi(2, 40, 1));
        for k = 1:numel(list)
            vdir = fullfile(here, 'versions', list(k).id);
            addpath(vdir, '-begin');
            try
                lg = define_layerArchitecture(model, config);
                L = lg.Layers(strcmp({lg.Layers.Name}, 'phaseLSTM'));
                okB = isa(L, 'peepholeLSTMLayer') && strcmp(L.VersionId, list(k).id) && L.NumHiddenUnits == S.net.Layers(1).InputSize;
                msg = '';
                if k == 1 || k == numel(list)   % trainNetwork run for the faithful and the ideal version
                    opts = trainingOptions('adam', 'MaxEpochs', 2, 'MiniBatchSize', 20, 'Verbose', false, ...
                        'Plots', 'none', 'GradientThreshold', 82.407, 'ExecutionEnvironment', ternary(o.gpu, 'gpu', 'cpu'));
                    tic; [~, info] = trainNetwork(Xtr, Ytr, lg, opts); tt = toc;
                    okB = okB && all(isfinite(info.TrainingLoss));
                    msg = sprintf(', trainNetwork 4 iterations (B = 20) %.1f s, loss %.3f -> %.3f', tt, info.TrainingLoss(1), info.TrainingLoss(end));
                end
            catch err
                okB = false; msg = [' ERROR ' err.message];
            end
            rmpath(vdir);
            nFail = nFail + ~report(sprintf('%-12s builds in the LSTM projection (H = %d)%s', list(k).id, S.net.Layers(1).InputSize, msg), okB);
        end
        % one full-size iteration (B = 197, T = 500) forward + backward
        HH = S.net.Layers(1).InputSize;
        for id = ["gers2002", "ideal"]
            rng(1);
            net = dlnetwork([sequenceInputLayer(2); peepholeLSTMLayer(HH, 'Name', 'p', 'Version', id)]);
            Xb = dlarray(single(randn(2, 197, 500)), 'CBT'); Wb = single(randn(HH, 197, 500));
            if o.gpu, Xb = gpuArray(Xb); Wb = gpuArray(Wb); net = dlupdate(@gpuArray, net); end
            dlfeval(@lossLayer, net, Xb, Wb);        % warm-up
            tic; [~, g] = dlfeval(@lossLayer, net, Xb, Wb); if o.gpu, wait(gpuDevice); end; tt = toc;
            fin = all(cellfun(@(v) all(isfinite(gather(v)), 'all'), g));
            nFail = nFail + ~report(sprintf('%s full size (H %d, B 197, T 500): forward + backward %.2f s, finite grads', id, HH, tt), fin);
        end
    end
end
fprintf('[checkPeephole] %d failure(s)\n', nFail);
end

% =============================================================================
function [y, g] = lossLayer(net, X, w)
y = forward(net, X);
loss = sum(stripdims(y) .* w, 'all');
g = dlgradient(loss, net.Learnables.Value);
y = extractdata(y);
end

function [y, g] = lossRef(Lt, X, w, cfg, H, mode)
val = @(n) Lt.Value{strcmp(Lt.Parameter, n)};
W = val('W'); R = val('R'); b = val('b'); p = val('p');
x = stripdims(X);
[~, B, T] = size(x, 1:3);
h = dlarray(zeros(H, B)); c = h;
ri = 1:H; rf = H+1:2*H; rz = 2*H+1:3*H; ro = 3*H+1:4*H;
tr = strcmp(cfg.grad, 'truncated');
cut = @(v) ternary(tr, dlarray(extractdata(v)), v);
act = @(n, v) ternary(strcmp(n, 'tanh'), tanh(v), v);
ys = cell(1, T);
for t = 1:T
    hin = cut(h); cpk = cut(c);
    a = W * x(:,:,t) + R * hin + b;
    if cfg.peep
        ai = a(ri,:) + p(ri) .* cpk; af = a(rf,:) + p(rf) .* cpk;
    else
        ai = a(ri,:); af = a(rf,:);
    end
    f = sigmoid(af);
    if cfg.cifg, ig = 1 - f; else, ig = sigmoid(ai); end
    c = f .* c + ig .* act(cfg.gFun, a(rz,:));
    ao = a(ro,:);
    if cfg.peep
        if strcmp(cfg.outPeep, 'current'), ao = ao + p(2*H+1:3*H) .* cut(c);
        else, ao = ao + p(2*H+1:3*H) .* cpk; end
    end
    h = sigmoid(ao) .* act(cfg.hFun, c);
    ys{t} = h;
end
if mode == "sequence", y = cat(3, ys{:}); else, y = h; end
loss = sum(y .* w, 'all');
g = dlgradient(loss, Lt.Value);
y = extractdata(y);
end

function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end

function d = maxRel(a, b)
a = double(gather(extractdataIf(a))); b = double(gather(extractdataIf(b)));
d = max(abs(a(:) - b(:))) / max(1, max(abs(b(:))));
end

function d = maxGradRel(g1, g2)
d = 0;
for k = 1:numel(g1)
    d = max(d, maxRel(g1{k}, g2{k}));
end
end

function v = extractdataIf(v)
if isa(v, 'dlarray'), v = extractdata(v); end
end

function ok = report(msg, ok)
if ok, tag = 'ok  '; else, tag = 'FAIL'; end
fprintf('[%s] %s\n', tag, msg);
end
