function ok = checkAssocLSTM(versions)
%CHECKASSOCLSTM  Checks of associativeLSTMLayer, CPU, seconds.
%
%   checkAssocLSTM                  every version in assocVersions
%   checkAssocLSTM({'paper'})
%
% For each version, on a small problem in double precision (N = 5 complex
% cells, 2 inputs, batch 3, 9 steps; weights scaled so that bound() and the
% keys are active on both sides of |z| = 1):
%   1. forward = a plain dlarray implementation written per copy, from the
%      paper's equations (independent of the layer's stacked code)
%   2. hand-written backward = dlgradient of that implementation, for the
%      input and every learnable
%   3. Eq. 36 (paper Sec. 6.2.4): keys fixed to 1, one copy, no omega ->
%      the memory follows the LSTM cell update c = gf c + gi u
% Run from the repository root (needs utils/ for the initialisers).
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'common'));
root = fileparts(fileparts(here));
addpath(genpath(fullfile(root, 'utils', 'modelstuff', 'helpers')));
if nargin < 1 || isempty(versions), versions = {assocVersions().id}; end
versions = cellstr(versions);
ok = true;
for k = 1:numel(versions)
    ok = checkOne(versions{k}) && ok;
end
ok = checkEq36() && ok;
if ok, fprintf('[checkAssocLSTM] all checks passed\n'); else, fprintf('[checkAssocLSTM] FAILED\n'); end
end

function ok = checkOne(id)
N = 5; C = 2; B = 3; T = 9;
cfg = assocVersions(id);
cfg.numComplex = N;
L = associativeLSTMLayer(N, 'Version', id, 'Cfg', cfg);
L = initialize(L, networkDataLayout([C NaN NaN], 'CBT'));
rng(7, 'twister');
L.W = 2.0 * randn(size(L.W));
L.U = 0.8 * randn(size(L.U));
if ~cfg.whu, L.U(7*N+1:end, :) = 0; end
L.b = 0.5 * randn(size(L.b));
if cfg.omega, L.omega = 0.3 * randn(N, 1); else, L.omega = zeros(N, 1); end
X = dlarray(randn(C, B, T), 'CBT');
[Y, mem] = forward(L, X);
R = randn(size(Y));

p = struct('W', dlarray(double(L.W)), 'U', dlarray(double(L.U)), 'b', dlarray(double(L.b)), 'omega', dlarray(double(L.omega)));
[lossRef, gRef, Yref] = dlfeval(@refLoss, p, dlarray(extractdata(X)), R, cfg, L.Perm);
fwdErr = max(abs(extractdata(stripdims(Y)) - extractdata(Yref)), [], 'all');

assocCapGrads('reset');
g = cell(1, 5);
[g{:}] = backward(L, X, Y, dlarray(R, 'CBT'), mem);
names = {'X','W','U','b','omega'};
ref = {gRef.X, gRef.W, gRef.U, gRef.b, gRef.omega};
ok = fwdErr < 1e-10;
msg = sprintf('fwd %.1e', fwdErr);
for i = 1:5
    a = double(extractdata(stripdims(g{i}))); r = double(extractdata(ref{i}));
    e = max(abs(a(:) - r(:))) / max(1e-12, max(abs(r(:))));
    if i == 3 && ~cfg.whu                              % W_hu = 0: its rows get no gradient
        e = max(max(abs(a(7*N+1:end, :)), [], 'all'), ...
                max(abs(a(1:7*N,:) - r(1:7*N,:)), [], 'all') / max(abs(r(:))));
    end
    if i == 5 && ~cfg.omega, e = max(abs(a(:))); end   % omega frozen: gradient must be 0
    ok = ok && e < 1e-8;
    msg = [msg sprintf(', d%s %.1e', names{i}, e)]; %#ok<AGROW>
end
fprintf('[checkAssocLSTM] %-13s loss %+.4f  %s  -> %s\n', id, extractdata(lossRef), msg, tf(ok));
end

function [loss, g, Y] = refLoss(p, X, R, cfg, Perm)
Y = refForward(p, X, cfg, Perm);
loss = sum(Y .* R, 'all');
[g.X, g.W, g.U, g.b, g.omega] = dlgradient(loss, X, p.W, p.U, p.b, p.omega);
end

function Y = refForward(p, X, cfg, Perm)
% Danihelka et al. 2016, Eqs. 17-30, one copy at a time.
[C, B, T] = size(X); %#ok<ASGLU>
N = cfg.numComplex; S = cfg.numCopies;
h = dlarray(zeros(2*N, B));
c = cell(S, 2);
for s = 1:S, c{s,1} = dlarray(zeros(N, B)); c{s,2} = dlarray(zeros(N, B)); end
Y = [];
for t = 1:T
    z  = p.W * X(:,:,t) + p.b + p.U * h;
    gf = sigmoid(z(1:N,:)); gi = sigmoid(z(N+1:2*N,:)); go = sigmoid(z(2*N+1:3*N,:));
    [riR, riI] = key(z(3*N+1:4*N,:), z(4*N+1:5*N,:), cfg.keys);
    [roR, roI] = key(z(5*N+1:6*N,:), z(6*N+1:7*N,:), cfg.keys);
    [uR, uI]   = bnd(z(7*N+1:8*N,:), z(8*N+1:9*N,:));
    vR = gi .* uR; vI = gi .* uI;
    mR = 0; mI = 0;
    for s = 1:S
        P = Perm(:, s);
        cR = c{s,1}; cI = c{s,2};
        if cfg.omega
            [cR, cI] = deal(cos(p.omega) .* cR - sin(p.omega) .* cI, sin(p.omega) .* cR + cos(p.omega) .* cI);
        end
        kR = riR(P,:); kI = riI(P,:);
        cR = gf .* cR + (kR .* vR - kI .* vI);
        cI = gf .* cI + (kR .* vI + kI .* vR);
        c{s,1} = cR; c{s,2} = cI;
        oR = roR(P,:); oI = roI(P,:);
        mR = mR + (oR .* cR - oI .* cI) / S;
        mI = mI + (oR .* cI + oI .* cR) / S;
    end
    [bR, bI] = bnd(mR, mI);
    h = [go .* bR; go .* bI];
    switch cfg.output
        case 'h',    y = h;
        case 'real', y = go .* bR;
        case 'mod',  y = go .* sqrt(bR.^2 + bI.^2 + 1e-12);
    end
    Y = cat(3, Y, y);
end
end

function [a, b] = bnd(a, b)
d = max(1, sqrt(a.^2 + b.^2));
a = a ./ d; b = b ./ d;
end

function [a, b] = key(a, b, mode)
if strcmp(mode, 'bound'), [a, b] = bnd(a, b); return; end
d = max(sqrt(a.^2 + b.^2), 1e-6);
a = a ./ d; b = b ./ d;
end

function ok = checkEq36()
% Keys = 1 (r_i = r_o = 1 + 0i), one copy: c = gf c + gi u and, while
% |c| <= 1, h = go c, exactly the LSTM with tanh replaced by bound().
N = 4; C = 2; B = 2; T = 6;
cfg = assocVersions('copies1'); cfg.numComplex = N;
L = associativeLSTMLayer(N, 'Cfg', cfg, 'Version', 'copies1');
L = initialize(L, networkDataLayout([C NaN NaN], 'CBT'));
rng(3, 'twister');
L.W = 0.3 * randn(size(L.W)); L.U = 0.2 * randn(size(L.U)); L.b = zeros(size(L.b));
L.W(3*N+1:7*N, :) = 0; L.U(3*N+1:7*N, :) = 0;      % key pre-activations fixed ...
L.b(3*N+1:4*N) = 1; L.b(5*N+1:6*N) = 1;              % ... at 1 + 0i
X = randn(C, B, T) * 0.3;
Y = extractdata(predict(L, dlarray(X, 'CBT')));
h = zeros(2*N, B); cR = zeros(N, B); cI = cR; err = 0;
for t = 1:T
    z = L.W * X(:,:,t) + L.b + L.U * h;
    f = 1./(1+exp(-z(1:N,:))); i = 1./(1+exp(-z(N+1:2*N,:))); o = 1./(1+exp(-z(2*N+1:3*N,:)));
    [uR, uI] = bnd(z(7*N+1:8*N,:), z(8*N+1:9*N,:));
    cR = f .* cR + i .* uR; cI = f .* cI + i .* uI;
    [bR, bI] = bnd(cR, cI);
    h = [o .* bR; o .* bI];
    err = max(err, max(abs(h - Y(:,:,t)), [], 'all'));
end
ok = err < 1e-5;
fprintf('[checkAssocLSTM] Eq. 36 (unit keys, one copy = LSTM cell update): max |diff| %.1e -> %s\n', err, tf(ok));
end

function s = tf(ok)
if ok, s = 'ok'; else, s = 'FAIL'; end
end
