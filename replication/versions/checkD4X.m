function report = checkD4X(varargin)
%CHECKD4X  Verify phaseAwareLSTMD4X (d4lowInit, d4specUQ, d4normMem) against autodiff.
%
% Short test, trains nothing. From the repository root:
%   addpath('replication/versions'); checkD4X();
% 1. All options off: same learnables, output and gradients as
%    phaseAwareLSTMHalfAngMatched (d4). InitScale 0.5: U and Q = 0.5 x d4's.
% 2. SpecNorm and MemNorm (separately and together), double, CPU and GPU:
%    output, final states and every learnable gradient of the fused layer
%    against a plain dlarray implementation differentiated by dlgradient
%    (spectral norm in the reference by 300 power iterations). U and Q are
%    scaled x1.6 so the spectral normalisation is active, and the biases and
%    input weights are enlarged so |c| > 1 occurs (the fraction is printed).
% Expected: rel. differences below about 1e-6.
p = inputParser;
addParameter(p, 'H', 10);
addParameter(p, 'B', 5);
addParameter(p, 'C', 2);
addParameter(p, 'T', 20);
addParameter(p, 'seed', 0);
parse(p, varargin{:}); o = p.Results;
verDir  = fileparts(mfilename('fullpath'));
repDir  = fileparts(verDir);
rootDir = fileparts(repDir);
oldPath = path; restorePath = onCleanup(@() path(oldPath)); %#ok<NASGU>
addpath(genpath(fullfile(rootDir, 'utils')));
addpath(fullfile(verDir, 'fast', 'common'));
addpath(fullfile(repDir, 'variants', 'halfAngMatchedPhaseLSTM'));
envs = {'cpu'}; if canUseGPU, envs{end+1} = 'gpu'; end
H = o.H; C = o.C; B = o.B; T = o.T;
rd = @(a, b) max(abs(a(:) - b(:))) / max(max(abs(b(:))), realmin);
report = struct('test', {}, 'value', {});
worst = 0;

% ---- 1. options off == d4; InitScale
rng(o.seed); netD = dlnetwork([sequenceInputLayer(C); phaseAwareLSTMHalfAngMatched(H, 'Name', 'p')]);
rng(o.seed); netX = dlnetwork([sequenceInputLayer(C); phaseAwareLSTMD4X(H, 'Name', 'p')]);
rng(o.seed); netL = dlnetwork([sequenceInputLayer(C); phaseAwareLSTMD4X(H, 'Name', 'p', 'InitScale', 0.5)]);
for k = 1:height(netD.Learnables)
    nm = netD.Learnables.Parameter{k};
    a = ex(netD.Learnables.Value{k});
    assert(isequal(a, ex(netX.Learnables.Value{k})), 'options off: %s differs from d4', nm);
    l = ex(netL.Learnables.Value{k});
    if any(strcmp(nm, {'U', 'Q'})), assert(max(abs(l(:) - 0.5 * a(:))) < 1e-7, 'InitScale: %s', nm);
    else, assert(isequal(l, a), 'InitScale changed %s', nm); end
end
for env = envs
    [nD, X, w] = prep(netD, C, B, T, env{1}, o.seed);
    nX = prep(netX, C, B, T, env{1}, o.seed);
    [YD, GD] = dlfeval(@netLossGrad, nD, X, w);
    [YX, GX] = dlfeval(@netLossGrad, nX, X, w);
    e = rd(YX, YD);
    for k = 1:height(GD), e = max(e, rd(ex(GX.Value{k}), ex(GD.Value{k}))); end
    fprintf('options off, %s: max rel diff vs d4 %.3g\n', env{1}, e);
    report(end+1) = struct('test', ['off_' env{1}], 'value', e); %#ok<AGROW>
    worst = max(worst, e);
end

% ---- 2. SpecNorm / MemNorm vs autodiff
modes = {struct('SpecNorm', true, 'MemNorm', false), struct('SpecNorm', false, 'MemNorm', true), ...
         struct('SpecNorm', true, 'MemNorm', true)};
for mi = 1:numel(modes)
    md = modes{mi};
    for om = ["sequence", "last"]
        rng(o.seed + 1);
        net = dlnetwork([sequenceInputLayer(C); phaseAwareLSTMD4X(H, 'Name', 'p', 'OutputMode', om, ...
            'SpecNorm', md.SpecNorm, 'MemNorm', md.MemNorm)]);
        L = net.Learnables;
        for k = 1:height(L)
            v = ex(L.Value{k});
            switch L.Parameter{k}
                case {'U', 'Q'}, v = 1.6 * v;
                case {'W1', 'W2'}, v = 3 * v;
                case {'b1', 'b2'}, v = v + 0.8 * randn(size(v));
            end
            L.Value{k} = dlarray(v);
        end
        net.Learnables = L;
        for env = envs
            [n, X, w] = prep(net, C, B, T, env{1}, o.seed + 2);
            X = 3 * X;
            if om == "last", w = w(:,:,end); end
            [YF, GF, SF] = dlfeval(@netLossGrad, n, X, w);
            P = struct();
            for k = 1:height(n.Learnables), P.(n.Learnables.Parameter{k}) = n.Learnables.Value{k}; end
            [YR, GR, SR, fracBig] = dlfeval(@refLossGrad, P, X, w, H, md, om);
            e = rd(YF, YR); msg = sprintf('Y %.2g', e);
            for s = 1:4, e = max(e, rd(SF{s}, SR{s})); end
            for k = 1:height(GF)
                nm = GF.Parameter{k};
                ek = rd(ex(GF.Value{k}), ex(GR.(nm)));
                e = max(e, ek); msg = [msg sprintf(' d%s %.2g', nm, ek)]; %#ok<AGROW>
            end
            fprintf('spec %d mem %d %-8s %s: max rel diff vs autodiff %.3g  (|c|>1 in %.0f%% of unit-steps) [%s]\n', ...
                md.SpecNorm, md.MemNorm, om, env{1}, e, 100 * fracBig, msg);
            report(end+1) = struct('test', sprintf('spec%d_mem%d_%s_%s', md.SpecNorm, md.MemNorm, om, env{1}), 'value', e); %#ok<AGROW>
            worst = max(worst, e);
        end
    end
end
fprintf('WORST rel diff %.3g -> %s\n', worst, passFail(worst < 1e-6));
end

% =====================================================================
function [n, X, w] = prep(net, C, B, T, env, seed)
if strcmp(env, 'gpu'), mv = @(v) gpuArray(double(v)); else, mv = @double; end
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

function [Y, G, S, fracBig] = refLossGrad(P, X, w, H, md, om)
[Y, S, fracBig] = d4xRef(P, X, H, md, om);
loss = sum(Y .* w, 'all');
G = dlgradient(loss, P);
Y = ex(Y);
S = cellfun(@ex, S, 'UniformOutput', false);
end

function [Y, S, fracBig] = d4xRef(P, X, H, md, om)
ep = 1e-9; dl2 = 0.05^2;
Xd = stripdims(X);
[~, B, T] = size(Xd);
U = P.U; Q = P.Q;
if md.SpecNorm, U = U / max(1, sigmaMax(U)); Q = Q / max(1, sigmaMax(Q)); end
W2 = P.W2(1:3*H,:); Q3 = Q(1:3*H,:); b2 = P.b2(1:3*H);
z = zeros(H, B, 'like', extractdata(Xd));
h = dlarray(z); ph = dlarray(z); cR = dlarray(z); cI = dlarray(z);
clip = @(v) max(min(v, 5), -5);
Ys = cell(1, T); nBig = 0;
for t = 1:T
    xt = Xd(:,:,t);
    G  = clip(P.W1 * xt + U * h + P.b1);
    Pp = clip(W2 * xt + Q3 * ph + b2);
    f  = sigmoid(G(1:H,:)); ig = sigmoid(G(H+1:2*H,:));
    gR = tanh(G(2*H+1:3*H,:)); o = sigmoid(G(3*H+1:4*H,:));
    gI = tanh(Pp(2*H+1:3*H,:));
    [cf, sf] = halfAngle(Pp(1:H,:), ep);
    [cn, sn] = halfAngle(Pp(H+1:2*H,:), ep);
    nR = f .* (cR .* cf - cI .* sf) + ig .* (gR .* cn - gI .* sn);
    nI = f .* (cR .* sf + cI .* cf) + ig .* (gR .* sn + gI .* cn);
    if md.MemNorm
        rho = sqrt(nR.^2 + nI.^2);
        nBig = nBig + nnz(extractdata(rho) > 1);
        k = 1 ./ max(rho, 1);
        cR = nR .* k; cI = nI .* k;
        r2 = cR.^2 + cI.^2;
        h  = o .* tanh(max(sqrt(r2), ep));
        ph = atan2(cI, cR) .* (r2 ./ (r2 + dl2));
    else
        cR = nR; cI = nI;
        h  = o .* tanh(max(sqrt(cR.^2 + cI.^2), ep));
        ph = atan2(cI, cR);
    end
    Ys{t} = h;
end
fracBig = nBig / (H * B * T);
if om == "sequence", Y = cat(3, Ys{:}); else, Y = h; end
S = {h, ph, cR, cI};
end

function s = sigmaMax(W)
v = dlarray(ones(size(W, 2), 1, 'like', extractdata(W)) / sqrt(size(W, 2)));
for k = 1:300
    v = W.' * (W * v);
    v = v / sqrt(sum(v.^2));
end
s = sqrt(sum((W * v).^2));
end

function [c, s] = halfAngle(p, ep)
u = tanh(0.5 * p); a = u .* u;
c = (1 - a) ./ (a + 1 + ep);
s = (2 * u) ./ (a + 1 + ep);
end

function v = ex(v)
if isa(v, 'dlarray'), v = extractdata(v); end
v = double(gather(v));
end

function s = passFail(ok)
if ok, s = 'PASS'; else, s = 'FAIL'; end
end
