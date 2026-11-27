function nFail = checkPhased(sourceNetFile)
%CHECKPHASED  Tests of the Phased LSTM versions before any training run.
%
%   nFail = checkPhased('pretrainedModels/cwl/cwl1/DeepRNN/sourceDomainClassifier_N2_DeepRNN_1.mat')
%
% Run from the root of a run tree (replication/ and utils/ next to each other).
%  1  phasedLSTMCfg with the phasedBase settings draws the same initial weights as
%     the ported phasedLSTM.m (same seed), and gives the same output and the same
%     gradients (autodiff through phasedLSTM.m) for W, U, b, the input, and zero
%     gradients for tau and s
%  2  for every version, the hand-written backward pass equals autodiff of a
%     reference forward written with dlarray operations (double precision, the
%     timing parameters set so that every gate phase occurs)
%  3  every version builds in front of the source net and trains three iterations
%     with trainNetwork; tau, s and r_on move only when the version says so
%  4  speed: seconds per forward + backward at the CWL size (H 61, B 234, T 500)
% Returns the number of failed tests (the run chain stops on any failure).
here = fileparts(mfilename('fullpath'));
rep = fileparts(here);
addpath(fullfile(rep, 'variants', 'phased', 'common'));
cleanup = onCleanup(@() rmpath(fullfile(rep, 'variants', 'phased', 'common')));
addpath(genpath(fullfile(fileparts(rep), 'utils')));
nFail = 0;
V = phasedVersions();
report = @(name, ok, msg) fprintf('[checkPhased] %-44s %s %s\n', name, ternary(ok, 'PASS', 'FAIL'), msg);

%% 1  port equality
try
    H = 8; C = 2; B = 5; T = 90;
    rng(11, 'threefry'); n0 = dlnetwork([sequenceInputLayer(C, 'Name', 'in'); phasedLSTM(H, 'Name', 'p')]);
    rng(11, 'threefry'); n1 = dlnetwork([sequenceInputLayer(C, 'Name', 'in'); phasedLSTMCfg(H, phasedVersions('phasedBase'), 'Name', 'p')]);
    L0 = n0.Learnables; L1 = n1.Learnables;
    same = true;
    for p = {'W1','U','b1','tau','shi'}
        a = gather(extractdata(L0.Value{strcmp(L0.Parameter, p{1})}));
        b = gather(extractdata(L1.Value{strcmp(L1.Parameter, p{1})}));
        same = same && isequal(a, b);
    end
    report('1a same initial weights as phasedLSTM.m', same, '');
    nFail = nFail + ~same;
    X = dlarray(single(randn(C, B, T)), 'CBT');
    if canUseGPU, X = gpuArray(X); end
    Y0 = predict(n0, X); Y1 = predict(n1, X);
    e = relErr(Y0, Y1);
    report('1b same output as phasedLSTM.m', e < 1e-5, sprintf('(rel err %.2g)', e));
    nFail = nFail + ~(e < 1e-5);
    R = dlarray(single(randn(H, B, T)), 'CBT'); if canUseGPU, R = gpuArray(R); end
    [g0, gx0] = dlfeval(@lossGrad, n0, X, R);
    [g1, gx1] = dlfeval(@lossGrad, n1, X, R);
    worst = relErr(gx0, gx1);
    for p = {'W1','U','b1'}
        worst = max(worst, relErr(g0.Value{strcmp(g0.Parameter, p{1})}, g1.Value{strcmp(g1.Parameter, p{1})}));
    end
    zt = 0;
    for p = {'tau','shi'}
        zt = zt + nnz(gather(extractdata(g0.Value{strcmp(g0.Parameter, p{1})}))) ...
                + nnz(gather(extractdata(g1.Value{strcmp(g1.Parameter, p{1})})));
    end
    ok = worst < 1e-4 && zt == 0;
    report('1c same gradients as phasedLSTM.m (autodiff)', ok, sprintf('(worst rel err %.2g, nonzero tau/s grads %d)', worst, zt));
    nFail = nFail + ~ok;
catch ME
    report('1  port equality', false, ME.message); nFail = nFail + 1;
end

%% 2  hand backward = autodiff, every version
for v = 1:numel(V)
    s = V(v);
    if s.port, continue; end
    try
        H = 6; C = 2; B = 4; T = 70;
        rng(3, 'threefry');
        lay = phasedLSTMCfg(H, s, 'Name', 'p');
        lay = initialize(lay, networkDataLayout([C NaN NaN], 'CBT'));
        P = struct();
        for p = {'W1','U','b1','tau','shi','ron','Pi','Pf','Po'}, P.(p{1}) = double(gather(lay.(p{1}))); end
        % timing that visits every gate phase in 70 steps
        P.tau = linspace(3, 17, H)'; P.shi = 0.3 * P.tau; P.ron = linspace(0.25, 0.6, H)';
        if s.peephole, P.Pi = 0.3*randn(H,1); P.Pf = 0.3*randn(H,1); P.Po = 0.3*randn(H,1); end
        lay.Cfg.leakTrain = max(s.leakTrain, 0.05);    % a visible leak, so its gradient is tested
        for p = fieldnames(P)', lay.(p{1}) = P.(p{1}); end
        Xd = randn(C, B, T); Rd = randn(H, B, T);
        X = dlarray(Xd, 'CBT');
        [Z, mem] = forward(lay, X);
        gh = cell(1, 10);
        [gh{:}] = backward(lay, X, Z, dlarray(Rd, 'CBT'), mem);
        names = {'X','W1','U','b1','tau','shi','ron','Pi','Pf','Po'};
        Pd = structfun(@(a) dlarray(a), P, 'UniformOutput', false);
        ga = cell(1, 10);
        [ga{:}] = dlfeval(@refGrad, Pd, dlarray(Xd), Rd, lay.Cfg);
        worst = 0; msg = '';
        for q = 1:10
            a = double(extractdata(gh{q})); b = double(extractdata(ga{q}));
            e = norm(a(:) - b(:)) / max(norm(b(:)), 1e-12);
            if norm(b(:)) == 0, e = norm(a(:)); end
            if e > worst, worst = e; msg = names{q}; end
        end
        Zr = refForward(Pd, dlarray(Xd), lay.Cfg, lay.Cfg.leakTrain);
        ef = relErr(Z, Zr);
        ok = worst < 1e-8 && ef < 1e-12;
        report(sprintf('2  %s backward = autodiff', s.name), ok, sprintf('(forward %.2g, worst grad %.2g on %s)', ef, worst, msg));
        nFail = nFail + ~ok;
    catch ME
        report(sprintf('2  %s backward = autodiff', s.name), false, ME.message); nFail = nFail + 1;
    end
end

%% 3  build in front of the source net and train three iterations
haveSource = nargin > 0 && isfile(sourceNetFile);
if ~haveSource
    report('3  build + train (no source net file given)', false, ''); nFail = nFail + 1;
else
    S = load(sourceNetFile);
    model.pretrainedNet = S.net; model.nchans = 2; model.signalLength = 500; model.typeNet = 'DeepRNN';
    cls = S.net.Layers(end).Classes;
    rng(5, 'threefry');
    N = 24; Xs = cell(N, 1);
    for j = 1:N, Xs{j} = single(randn(2, 500)); end
    Ys = cls(1 + mod((1:N)', numel(cls)));
    cfg = struct('domain', 'target-domain', 'layerName', 'phased-lstm', 'typeNet', 'DeepRNN', 'datasetType', 'CWL', 'ncats', 2);
    for v = 1:numel(V)
        s = V(v);
        try
            lg = buildVersion(model, s);
            % draw the projection's weights now, so the trained values can be compared
            pl = layerByName(lg, 'phaseLSTM');
            rng(5, 'threefry');
            pl = initialize(pl, networkDataLayout([2 NaN NaN], 'CBT'));
            lg = replaceLayer(lg, 'phaseLSTM', pl);
            opts = get_trainingOptions(cfg);          % the phased routing (phased/common)
            opts.MaxEpochs = 1; opts.MiniBatchSize = 8; opts.Verbose = false;
            opts.ValidationData = []; opts.OutputNetwork = 'last-iteration';
            rng(5, 'threefry');
            tn = trainNetwork(Xs, Ys, lg, opts);
            tl = layerByName(tn.Layers, 'phaseLSTM');
            mv = @(p) ~isequal(gather(pl.(p)), gather(tl.(p)));
            if s.port
                expect = struct('tau', true, 'shi', true);      % L2 decay only
            else
                expect = struct('tau', s.timingGrad || s.timingL2 > 0, 'shi', s.timingGrad || s.timingL2 > 0, 'ron', s.ronLearn);
            end
            ok = true; msg = '';
            for p = fieldnames(expect)'
                m = mv(p{1});
                msg = [msg sprintf('%s %s; ', p{1}, ternary(m, 'moved', 'fixed'))]; %#ok<AGROW>
                ok = ok && (m == expect.(p{1}));
            end
            ok = ok && mv('W1');                       % the projection trained at all
            yp = predict(tn, Xs(1:2));
            ok = ok && all(isfinite(yp(:)));
            report(sprintf('3  %s build + 3 iterations', s.name), ok, msg);
            nFail = nFail + ~ok;
        catch ME
            report(sprintf('3  %s build + 3 iterations', s.name), false, ME.message); nFail = nFail + 1;
        end
    end
end

%% 4  speed at the CWL size
try
    H = 61; C = 2; B = 234; T = 500;
    X = dlarray(single(randn(C, B, T)), 'CBT'); R = dlarray(single(randn(H, B, T)), 'CBT');
    if canUseGPU, X = gpuArray(X); R = gpuArray(R); end
    rng(1, 'threefry'); n0 = dlnetwork([sequenceInputLayer(C); phasedLSTM(H, 'Name', 'p')]);
    rng(1, 'threefry'); n1 = dlnetwork([sequenceInputLayer(C); phasedLSTMCfg(H, phasedVersions('phasedNeil'), 'Name', 'p')]);
    tic; [~, ~] = dlfeval(@lossGrad, n0, X, R); if canUseGPU, wait(gpuDevice); end; t0 = toc;
    tic; [~, ~] = dlfeval(@lossGrad, n1, X, R); if canUseGPU, wait(gpuDevice); end; t1 = toc;
    fprintf('[checkPhased] 4  speed, one forward + backward at H 61, B 234, T 500 (%s): phasedLSTM.m (autodiff) %.1f s, phasedLSTMCfg %.1f s\n', ...
        ternary(canUseGPU, 'GPU', 'CPU'), t0, t1);
catch ME
    fprintf('[checkPhased] 4  speed test failed: %s\n', ME.message);
end
fprintf('[checkPhased] %d failed\n', nFail);
end

% ---------------------------------------------------------------------
function lg = buildVersion(model, s)
if s.port
    lg = get_pLayer_phasedLSTM(model);
else
    lg = get_pLayer_phasedVersion(model, s);
end
end
function L = layerByName(layers, name)
if isa(layers, 'nnet.cnn.LayerGraph'), layers = layers.Layers; end
L = layers(strcmp({layers.Name}, name));
end
function [g, gx] = lossGrad(net, X, R)
Y = forward(net, X);
loss = sum(Y .* R, 'all');
[g, gx] = dlgradient(loss, net.Learnables, X);
end
function varargout = refGrad(P, X, R, cfg)
Y = refForward(P, X, cfg, cfg.leakTrain);
loss = sum(Y .* R, 'all');
names = {'W1','U','b1','tau','shi','ron','Pi','Pf','Po'};
args = cellfun(@(n) P.(n), names, 'UniformOutput', false);
g = cell(1, 9);
[gx, g{1:9}] = dlgradient(loss, X, args{:});
varargout = [{gx}, g];
end
function Y = refForward(P, X, c, alpha)
% Phased LSTM written with dlarray operations only (autodiff reference)
H = size(P.U, 2); [C, B, T] = size(X, 1:3); %#ok<ASGLU>
tau = P.tau; s = P.shi; r = P.ron;
if ~c.timingGrad, tau = dlarray(extractdata(tau)); s = dlarray(extractdata(s)); end
if c.timingGrad, tau = max(tau, 0.5); end
if c.ronLearn, r = min(max(r, 1e-3), 1); else, r = dlarray(extractdata(r)); end
switch c.gate
    case 'nov2025', a = 0.1; c1 = 1; c2 = 3; c3 = 1;
    case 'neil',    a = 0.5; c1 = 2; c2 = 2; c3 = 2;
end
h = zeros(H, B); cc = zeros(H, B);
Y = dlarray(zeros(H, B, T));
for t = 1:T
    u = (t - s) ./ tau;
    phi = u - floor(extractdata(u));
    pv = extractdata(phi); rv = extractdata(r);
    m1 = pv < a * rv; m2 = ~m1 & pv < rv; m3 = ~(m1 | m2);
    k = m1 .* (c1 * phi ./ r) + m2 .* (c2 - c3 * phi ./ r) + m3 .* (alpha * phi);
    G = P.W1 * X(:, :, t) + P.U * h + P.b1;
    ai = G(1:H, :); af = G(H+1:2*H, :); ag = G(2*H+1:3*H, :); ao = G(3*H+1:4*H, :);
    if c.peephole, ai = ai + P.Pi .* cc; af = af + P.Pf .* cc; end
    i = sigmoid(ai); f = sigmoid(af); g = tanh(ag);
    ct = f .* cc + i .* g;
    if c.peephole, ao = ao + P.Po .* ct; end
    o = sigmoid(ao);
    ht = o .* tanh(ct);
    cc = k .* ct + (1 - k) .* cc;
    h  = k .* ht + (1 - k) .* h;
    Y(:, :, t) = h;
end
end
function e = relErr(a, b)
if isa(a, 'dlarray'), a = extractdata(a); end
if isa(b, 'dlarray'), b = extractdata(b); end
a = double(gather(a)); b = double(gather(b));
e = norm(a(:) - b(:)) / max(norm(b(:)), 1e-30);
end
function out = ternary(c, a, b)
if c, out = a; else, out = b; end
end
