function report = checkCrossFeed(varargin)
%CHECKCROSSFEED  Verify the fused crossFeed layer against an autodiff reference.
%
% Short test (a few minutes). Trains nothing, reads no data, writes nothing.
% From the repository root:
%   addpath('replication/versions');
%   report = checkCrossFeed();
%
% 1. Zero cross weights: phaseAwareLSTMCrossFeed (both modes) built with the
%    same RNG seed as phaseAwareLSTMHalfAngMatched (d4) has the same
%    learnables and gives the same output and gradients, i.e. at
%    initialisation crossFeed IS d4.
% 2. Random cross weights (Vm, Vp ~ 0.3 randn), double precision, CPU and
%    GPU: output, final states and every learnable gradient of the fused
%    layer (hand-written backward; fused kernels on the GPU) against
%    crossFeedRef below, a plain dlarray implementation differentiated by
%    dlgradient. Both CrossMode "both" and "magOnly", OutputMode sequence and
%    last. For "magOnly" the dVp gradient must be exactly zero.
% 3. Timing of one forward+backward at full size (H 30, B 161, T 500,
%    single, GPU if available), crossFeed vs d4.
% Expected: rel. differences below about 1e-6 (T = 20).
p = inputParser;
addParameter(p, 'H', 12);
addParameter(p, 'B', 5);
addParameter(p, 'C', 2);
addParameter(p, 'T', 20);
addParameter(p, 'timeFull', true);
addParameter(p, 'Hfull', 30);
addParameter(p, 'Bfull', 161);
addParameter(p, 'Tfull', 500);
addParameter(p, 'seed', 0);
parse(p, varargin{:}); o = p.Results;

verDir  = fileparts(mfilename('fullpath'));
repDir  = fileparts(verDir);
rootDir = fileparts(repDir);
oldPath = path; restorePath = onCleanup(@() path(oldPath)); %#ok<NASGU>
addpath(genpath(fullfile(rootDir, 'utils')));
addpath(fullfile(verDir, 'fast', 'common'));
addpath(fullfile(repDir, 'variants', 'halfAngMatchedPhaseLSTM'));
useGPU = canUseGPU;
envs = {'cpu'}; if useGPU, envs{end+1} = 'gpu'; end
H = o.H; C = o.C; B = o.B; T = o.T;
rd = @(a, b) max(abs(a(:) - b(:))) / max(max(abs(b(:))), realmin);
report = struct('test', {}, 'value', {});
worst = 0;

% ---- 1. zero cross weights == d4 --------------------------------------
rng(o.seed); netD = dlnetwork([sequenceInputLayer(C); phaseAwareLSTMHalfAngMatched(H, 'Name', 'p')]);
for mode = ["both", "magOnly"]
    rng(o.seed); netX = dlnetwork([sequenceInputLayer(C); phaseAwareLSTMCrossFeed(H, 'Name', 'p', 'CrossMode', mode)]);
    for k = 1:height(netD.Learnables)
        same = isequal(ex(netD.Learnables.Value{k}), ex(getL(netX, netD.Learnables.Parameter{k})));
        assert(same, 'zero-init: learnable %s differs from d4', netD.Learnables.Parameter{k});
    end
    assert(~any(ex(getL(netX, 'Vm')), 'all') && ~any(ex(getL(netX, 'Vp')), 'all'), 'cross weights not zero at init');
    for env = envs
        [nD, Xd, w] = prep(netD, C, B, T, env{1}, o.seed);
        nX = prep(netX, C, B, T, env{1}, o.seed);
        [YD, GD] = dlfeval(@netLossGrad, nD, Xd, w);
        [YX, GX] = dlfeval(@netLossGrad, nX, Xd, w);
        e = rd(YX, YD);
        for k = 1:height(GD)
            e = max(e, rd(ex(getG(GX, GD.Parameter{k})), ex(GD.Value{k})));
        end
        fprintf('zero cross weights, %-7s %s: max rel diff vs d4 (Y and grads) %.3g\n', mode, env{1}, e);
        report(end+1) = struct('test', sprintf('zeroInit_%s_%s', mode, env{1}), 'value', e); %#ok<AGROW>
        worst = max(worst, e);
    end
end

% ---- 2. random cross weights vs autodiff reference ---------------------
for mode = ["both", "magOnly"]
    for om = ["sequence", "last"]
        rng(o.seed + 1);
        net = dlnetwork([sequenceInputLayer(C); phaseAwareLSTMCrossFeed(H, 'Name', 'p', 'CrossMode', mode, 'OutputMode', om)]);
        L = net.Learnables;
        for k = 1:height(L)
            v = ex(L.Value{k});
            switch L.Parameter{k}
                case {'Vm', 'Vp'}, v = 0.3 * randn(size(v));
                case {'b1', 'b2'}, v = v + 0.2 * randn(size(v));
            end
            L.Value{k} = dlarray(v);
        end
        net.Learnables = L;
        for env = envs
            [n, X, w] = prep(net, C, B, T, env{1}, o.seed + 2);
            if om == "last", w = w(:,:,end); end
            [YF, GF, SF] = dlfeval(@netLossGrad, n, X, w);
            P = struct();
            for k = 1:height(n.Learnables), P.(n.Learnables.Parameter{k}) = n.Learnables.Value{k}; end
            [YR, GR, SR] = dlfeval(@refLossGrad, P, X, w, H, mode, om);
            e = rd(YF, YR);
            msg = sprintf('Y %.2g', e);
            for s = 1:4
                es = rd(SF{s}, SR{s}); e = max(e, es);
            end
            msg = [msg sprintf(' states %.2g', es)]; %#ok<AGROW>
            for k = 1:height(GF)
                nm = GF.Parameter{k};
                a = ex(GF.Value{k}); b = ex(GR.(nm));
                if mode == "magOnly" && strcmp(nm, 'Vp')
                    assert(~any(a, 'all'), 'magOnly: dVp is not zero');
                    ek = 0;
                else
                    ek = rd(a, b);
                end
                e = max(e, ek);
                msg = [msg sprintf(' d%s %.2g', nm, ek)]; %#ok<AGROW>
            end
            fprintf('crossFeed %-7s %-8s %s: max rel diff vs autodiff %.3g  [%s]\n', mode, om, env{1}, e, msg);
            report(end+1) = struct('test', sprintf('autodiff_%s_%s_%s', mode, om, env{1}), 'value', e); %#ok<AGROW>
            worst = max(worst, e);
        end
    end
end
fprintf('WORST rel diff %.3g -> %s\n', worst, passFail(worst < 1e-6));

% ---- 3. timing ----------------------------------------------------------
if o.timeFull
    Hf = o.Hfull;
    env = 'cpu'; if useGPU, env = 'gpu'; end
    rng(o.seed);
    nets = {dlnetwork([sequenceInputLayer(C); phaseAwareLSTMHalfAngMatched(Hf, 'Name', 'p')]), ...
            dlnetwork([sequenceInputLayer(C); phaseAwareLSTMCrossFeed(Hf, 'Name', 'p', 'CrossMode', "both")]), ...
            dlnetwork([sequenceInputLayer(C); phaseAwareLSTMCrossFeed(Hf, 'Name', 'p', 'CrossMode', "magOnly")])};
    names = {'d4', 'crossFeed', 'crossFeedMag'};
    for i = 1:3
        [n, X, w] = prep(nets{i}, C, o.Bfull, o.Tfull, env, o.seed, 'single');
        dlfeval(@netLossGrad, n, X, w);                    % warm-up
        if useGPU, wait(gpuDevice); end
        t = tic;
        for r = 1:3, dlfeval(@netLossGrad, n, X, w); end
        if useGPU, wait(gpuDevice); end
        s = toc(t) / 3;
        fprintf('timing %-12s H %d B %d T %d %s: %.2f s per forward+backward\n', names{i}, Hf, o.Bfull, o.Tfull, env, s);
        report(end+1) = struct('test', ['seconds_' names{i}], 'value', s); %#ok<AGROW>
    end
end
end

% =====================================================================
function [n, X, w] = prep(net, C, B, T, env, seed, cls)
if nargin < 7, cls = 'double'; end
cast_ = @(v) cast(v, cls);
if strcmp(env, 'gpu'), mv = @(v) gpuArray(cast_(v)); else, mv = cast_; end
n = dlupdate(@(v) dlarray(mv(ex(v))), net);
rng(seed);
X = dlarray(mv(randn(C, B, T)), 'CBT');
w = mv(randn(n.Layers(end).NumHiddenUnits, B, T));
end

function [Y, G, S] = netLossGrad(net, X, w)
[Y, state] = forward(net, X, 'Acceleration', 'none');
loss = sum(stripdims(Y) .* w, 'all');
G = dlgradient(loss, net.Learnables);
Y = ex(stripdims(Y));
S = cell(1, 4);
for s = 1:4, S{s} = ex(state.Value{s}); end
end

function [Y, G, S] = refLossGrad(P, X, w, H, mode, om)
[Y, S] = crossFeedRef(P, X, H, mode, om);
loss = sum(Y .* w, 'all');
G = dlgradient(loss, P);
Y = ex(Y);
S = cellfun(@ex, S, 'UniformOutput', false);
end

function [Y, S] = crossFeedRef(P, X, H, mode, om)
% Plain implementation of the crossFeed recurrence; differentiated by
% dlgradient (no custom backward). Same maths as phaseAwareLSTMCrossFeed:
%   G = clip5(W1 x + U h + Vm [cos phi; sin phi] + b1)        f i g o
%   P = clip5(W2 x + Q phi + Vp h + b2)    rows 1:3H           theta_f theta_i g_im
%   theta via half angle: t = tanh(P/2), cos = (1-t^2)/(1+t^2+eps), sin = 2t/(1+t^2+eps)
%   c = f e^{i theta_f} c_prev + i e^{i theta_i} (tanh g + i tanh g_im)
%   h = o tanh(max(|c|, eps)),  phi = atan2(Im c, Re c)
ep = 1e-9;
Xd = stripdims(X);                                  % C x B x T
[~, B, T] = size(Xd);
W2 = P.W2(1:3*H,:); Q3 = P.Q(1:3*H,:); b2 = P.b2(1:3*H);
Vp = P.Vp; if mode ~= "both", Vp = 0 * Vp; end
z  = zeros(H, B, 'like', extractdata(Xd));
h = dlarray(z); ph = dlarray(z); cR = dlarray(z); cI = dlarray(z);
clip = @(v) max(min(v, 5), -5);
Ys = cell(1, T);
for t = 1:T
    xt = Xd(:,:,t);
    G  = clip(P.W1 * xt + P.U * h + P.Vm * [cos(ph); sin(ph)] + P.b1);
    Pp = clip(W2 * xt + Q3 * ph + Vp * h + b2);
    f  = sigmoid(G(1:H,:)); ig = sigmoid(G(H+1:2*H,:));
    gR = tanh(G(2*H+1:3*H,:)); o = sigmoid(G(3*H+1:4*H,:));
    gI = tanh(Pp(2*H+1:3*H,:));
    [cf, sf] = halfAngle(Pp(1:H,:), ep);
    [cn, sn] = halfAngle(Pp(H+1:2*H,:), ep);
    nR = f .* (cR .* cf - cI .* sf) + ig .* (gR .* cn - gI .* sn);
    nI = f .* (cR .* sf + cI .* cf) + ig .* (gR .* sn + gI .* cn);
    cR = nR; cI = nI;
    h  = o .* tanh(max(sqrt(cR.^2 + cI.^2), ep));
    ph = atan2(cI, cR);
    Ys{t} = h;
end
if om == "sequence", Y = cat(3, Ys{:}); else, Y = h; end
S = {h, ph, cR, cI};
end

function [c, s] = halfAngle(p, ep)
u = tanh(0.5 * p); a = u .* u;
c = (1 - a) ./ (a + 1 + ep);
s = (2 * u) ./ (a + 1 + ep);
end

function v = getL(net, name)
v = net.Learnables.Value{strcmp(net.Learnables.Parameter, name)};
end

function v = getG(G, name)
v = G.Value{strcmp(G.Parameter, name)};
end

function v = ex(v)
if isa(v, 'dlarray'), v = extractdata(v); end
v = double(gather(v));
end

function s = passFail(ok)
if ok, s = 'PASS'; else, s = 'FAIL'; end
end
