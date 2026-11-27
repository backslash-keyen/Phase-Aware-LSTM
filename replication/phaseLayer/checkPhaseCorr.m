function ok = checkPhaseCorr(varargin)
%CHECKPHASECORR  Correctness check of phaseAwareLSTMCorr (trains nothing, about a minute).
%
%   addpath('replication/phaseLayer'); checkPhaseCorr();
%   checkPhaseCorr('gpu', false)
%
% Run from the repository root.
% 1. 'port' (defaults) against the paper layer phaseAwareLSTM_notrig
%    (variants/notrigPhaseLSTM): same initial learnables from the same seed,
%    same output and same gradients (dlgradient through the original,
%    hand-written backward here), double precision.
% 2. Every version against an independent plain-dlarray implementation of
%    the same configuration, differentiated by dlgradient: output and every
%    learnable gradient, double, CPU (and GPU in single if available).
% 3. Initial biases and Q size of each version.
% 4. gradCap: the capped gradients have norm <= cap and the direction is kept.
p = inputParser;
addParameter(p, 'gpu', canUseGPU);
parse(p, varargin{:});
useGpu = p.Results.gpu;

root = pwd;
here = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root, 'utils')));
addpath(fullfile(here, 'common'));
ok = true;
H = 5; C = 2; B = 3; T = 12;
rng(7); Xn = randn(C, B, T); Wn = randn(H, B, T);
tolF = 1e-10; tolG = 1e-8;

% ---- 1. port vs the paper layer --------------------------------------------
addpath(fullfile(root, 'replication', 'variants', 'notrigPhaseLSTM'));
cleanup = onCleanup(@() rmpath(fullfile(root, 'replication', 'variants', 'notrigPhaseLSTM')));
for mode = ["sequence", "last"]
    rng(1); netO = dlnetwork([sequenceInputLayer(C); phaseAwareLSTM_notrig(H, 'Name', 'p', 'OutputMode', mode)]);
    rng(1); netC = dlnetwork([sequenceInputLayer(C); phaseAwareLSTMCorr(H, 'Name', 'p', 'OutputMode', mode, 'Version', 'port')]);
    dInit = maxLearnDiff(netO, netC);
    netO = dlupdate(@double, netO); netC = dlupdate(@double, netC);
    X = dlarray(Xn, 'CBT');
    w = Wn; if mode == "last", w = Wn(:,:,end); end
    [yO, gO] = dlfeval(@lossGrad, netO, X, w);
    [yC, gC] = dlfeval(@lossGrad, netC, X, w);
    dY = maxRel(yO, yC); dG = maxGradRel(gO, gC);
    ok = report(sprintf('port = paper layer (%s): init %.1e, output %.1e, grads %.1e', mode, dInit, dY, dG), ...
        dInit == 0 && dY < tolF && dG < tolG) && ok;
end
clear cleanup

% ---- 2./3. every version vs plain dlarray ----------------------------------
list = phaseCorrVersions();
for k = 1:numel(list)
    id = list(k).id;
    cfg = phaseCorrVersions(id);
    cfgNoCap = cfg; cfgNoCap.gradCap = Inf;
    isIdeal = startsWith(id, 'ideal');
    rng(1);
    if isIdeal
        net = dlnetwork([sequenceInputLayer(C); phaseAwareLSTMIdeal(H, 'Name', 'p', 'Version', id, 'Cfg', cfgNoCap)]);
        L = net.Layers(2);
        eb = zeros(6*H,1); eb(1:H) = 1;
        initOk = isequal(double(extractdata(L.b)), eb) && isequal(size(L.Uz), [6*H, 2*H]) && ...
            (cfg.omega == any(extractdata(L.omega) ~= 0));
    else
        net = dlnetwork([sequenceInputLayer(C); phaseAwareLSTMCorr(H, 'Name', 'p', 'Version', id, 'Cfg', cfgNoCap)]);
        L = net.Layers(2);
        b1 = extractdata(L.b1); b2 = extractdata(L.b2);
        eb1 = zeros(4*H,1); if strcmp(cfg.forgetRow, 'f'), eb1(1:H) = 1; else, eb1(H+1:2*H) = 1; end
        eb2 = zeros(4*H,1); if strcmp(cfg.phaseBias, 'unitForget'), eb2(H+1:2*H) = 1; end
        qw = H * (1 + ~strcmp(cfg.feedback, 'atan2'));
        initOk = isequal(double(b1), eb1) && isequal(double(b2), eb2) && isequal(size(L.Q), [4*H, qw]);
    end
    net = dlupdate(@double, net);
    % moderate random weights so every branch is exercised (no clipping at +-5 mostly)
    rng(3);
    net = dlupdate(@(v) v + 0.3*randn(size(v)), net);
    if isIdeal && ~cfg.omega      % omega is fixed at 0 in this version
        Lt = net.Learnables;
        Lt.Value{strcmp(Lt.Parameter, 'omega')} = dlarray(zeros(H, 1));
        net.Learnables = Lt;
    end
    X = dlarray(Xn, 'CBT');
    [yC, gC] = dlfeval(@lossGrad, net, X, Wn);
    [yR, gR] = dlfeval(@lossGradRef, net.Learnables, X, Wn, cfg, H, isIdeal);
    if isIdeal && ~cfg.omega
        gR.Value{strcmp(gR.Parameter, 'omega')} = 0 * gR.Value{strcmp(gR.Parameter, 'omega')};
    end
    if isIdeal && strcmp(cfg.output, 'mag')     % Wo, bo unused
        for nm = {'Wo','bo'}, gR.Value{strcmp(gR.Parameter, nm{1})} = 0 * gR.Value{strcmp(gR.Parameter, nm{1})}; end
    end
    dY = maxRel(yR, yC); dG = maxGradRel(gR, gC);
    ok = report(sprintf('%-12s vs dlarray: init %d, output %.1e, grads %.1e', id, initOk, dY, dG), ...
        initOk && dY < tolF && dG < 1e-7) && ok;
    if useGpu
        netG = dlupdate(@(v) gpuArray(single(v)), net);
        XG = dlarray(gpuArray(single(Xn)), 'CBT');
        [yG, gG] = dlfeval(@lossGrad, netG, XG, gpuArray(single(Wn)));
        dYg = maxRel(yR, yG); dGg = maxGradRel(gR, gG);
        ok = report(sprintf('%-12s GPU single: output %.1e, grads %.1e', id, dYg, dGg), ...
            dYg < 1e-4 && dGg < 1e-2) && ok;
    end
end

% ---- 4. gradCap ------------------------------------------------------------
g = {randn(4,3)*1e3, randn(2,1)*1e-3, [1 Inf], single(randn(5)*1e20)};
c = phaseCorrCapGrads(g, 1.25);
capOk = abs(norm(c{1}(:)) - 1.25) < 1e-9 && isequal(c{2}, g{2}) && all(c{3} == 0) ...
    && abs(norm(double(c{4}(:))) - 1.25) < 1e-5 && norm(c{1}(:)/norm(c{1}(:)) - g{1}(:)/norm(g{1}(:))) < 1e-12;
ok = report('gradCap: norm <= cap, direction kept, Inf -> 0, single 1e20 ok', capOk) && ok;

if ok, fprintf('checkPhaseCorr: PASS\n'); else, fprintf('checkPhaseCorr: FAIL\n'); end
end

% =====================================================================
function [y, g] = lossGrad(net, X, w)
Y = forward(net, X);
y = extractdata(stripdims(Y));
loss = sum(stripdims(Y) .* w, 'all');
g = dlgradient(loss, net.Learnables);
end

function [y, g] = lossGradRef(learn, X, w, cfg, H, isIdeal)
P = struct();
for i = 1:height(learn), P.(learn.Parameter{i}) = learn.Value{i}; end
if isIdeal
    Y = refIdeal(P, stripdims(X), cfg, H);
else
    Y = refForward(P, stripdims(X), cfg, H);
end
y = extractdata(Y);
loss = sum(Y .* w, 'all');
g = learn;
g.Value = cell(height(learn), 1);
grads = cell(1, height(learn));
[grads{:}] = dlgradient(loss, learn.Value{:});
g.Value = grads(:);
end

function Y = refForward(P, X, cfg, H)
% Independent implementation: explicit angles, cos/sin, complex algebra in
% real and imaginary parts; autodiff gives the gradients.
[~, B, T] = size(X);
h = zeros(H, B, 'like', X); cR = h; cI = h;
zw = H * (1 + ~strcmp(cfg.feedback, 'atan2'));
z = zeros(zw, B, 'like', X);
Y = [];
for t = 1:T
    x = X(:,:,t);
    G = P.W1 * x + P.b1 + P.U * h;
    Pp = P.W2 * x + P.b2 + P.Q * z;
    G = min(max(G, -cfg.clip), cfg.clip);
    Pp = min(max(Pp, -cfg.clip), cfg.clip);
    f = sigmoid(G(1:H,:)); ig = sigmoid(G(H+1:2*H,:)); o = sigmoid(G(3*H+1:4*H,:));
    gR = tanh(G(2*H+1:3*H,:)); gI = tanh(Pp(2*H+1:3*H,:));
    thf = atan(Pp(1:H,:)); thi = atan(Pp(H+1:2*H,:));
    % f e^{i thf} (cR + i cI) + i_g e^{i thi} (gR + i gI)
    nR = f .* (cos(thf).*cR - sin(thf).*cI) + ig .* (cos(thi).*gR - sin(thi).*gI);
    nI = f .* (sin(thf).*cR + cos(thf).*cI) + ig .* (sin(thi).*gR + cos(thi).*gI);
    cR = nR; cI = nI;
    rho = sqrt(cR.^2 + cI.^2);
    h = o .* tanh(max(rho, 1e-9));
    switch cfg.feedback
        case 'atan2',    z = atan2(cI, cR);
        case 'cosSin',   z = [cR ./ rho; cI ./ rho];
        case 'complexH', z = [o .* tanh(rho) ./ rho .* cR; o .* tanh(rho) ./ rho .* cI];
    end
    Y = cat(3, Y, h);
end
end

function Y = refIdeal(P, X, cfg, H)
% Independent implementation of phaseAwareLSTMIdeal: polar form for bound(),
% explicit cos/sin rotation; autodiff gives the gradients.
[~, B, T] = size(X);
z = zeros(2*H, B, 'like', X); cR = zeros(H, B, 'like', X); cI = cR;
Y = [];
for t = 1:T
    A = P.Wx * X(:,:,t) + P.Uz * z + P.b;
    f = sigmoid(A(1:H,:)); ig = sigmoid(A(H+1:2*H,:)); o = sigmoid(A(2*H+1:3*H,:));
    vR = A(3*H+1:4*H,:); vI = A(4*H+1:5*H,:);
    mv = sqrt(vR.^2 + vI.^2);
    uR = tanh(mv) .* vR ./ mv; uI = tanh(mv) .* vI ./ mv;
    if strcmp(cfg.rotMap, 'piTanh'), th = P.omega + pi * tanh(A(5*H+1:6*H,:)); else, th = P.omega + atan(A(5*H+1:6*H,:)); end
    if strcmp(cfg.inputScale, 'coupled'), s = 1 - f; else, s = 1; end
    nR = f .* (cos(th) .* cR - sin(th) .* cI) + s .* ig .* uR;
    nI = f .* (sin(th) .* cR + cos(th) .* cI) + s .* ig .* uI;
    cR = nR; cI = nI;
    rho = sqrt(cR.^2 + cI.^2);
    z = [o .* tanh(rho) .* cR ./ rho; o .* tanh(rho) .* cI ./ rho];
    if strcmp(cfg.output, 'readout'), y = P.Wo * z + P.bo; else, y = o .* tanh(rho); end
    Y = cat(3, Y, y);
end
end

function d = maxLearnDiff(a, b)
d = 0;
for i = 1:height(a.Learnables)
    va = extractdata(a.Learnables.Value{i}); vb = extractdata(b.Learnables.Value{i});
    if ~isequal(size(va), size(vb)), d = Inf; return; end
    d = max(d, max(abs(double(va(:)) - double(vb(:)))));
end
end

function r = maxRel(a, b)
a = double(gather(extractIf(a))); b = double(gather(extractIf(b)));
r = max(abs(a(:) - b(:))) / max(1, max(abs(a(:))));
end

function r = maxGradRel(ga, gb)
r = 0;
for i = 1:height(ga)
    a = double(gather(extractIf(ga.Value{i}))); b = double(gather(extractIf(gb.Value{i})));
    if ~isequal(size(a), size(b)), r = Inf; return; end
    r = max(r, max(abs(a(:) - b(:))) / max(1e-12, max(abs(a(:)))));
end
end

function v = extractIf(v)
if isa(v, 'dlarray'), v = extractdata(v); end
end

function ok = report(msg, ok)
if ok, tag = 'ok  '; else, tag = 'FAIL'; end
fprintf('[%s] %s\n', tag, msg);
end
