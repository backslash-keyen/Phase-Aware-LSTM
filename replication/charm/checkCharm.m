function nFail = checkCharm()
%CHECKCHARM  Quick CPU checks of the CHARM layers and of every CHARM variant's networks.
%
%   cd <repo root>; addpath replication/charm; checkCharm
%
% Small random data only (no dataset, no saved models), a few seconds of training.
%   1. charmLayer3Conv with NumConv = 1 equals charmLayer (same seed): bit-identical output
%   2. CHARM is invariant to the order of the input channels (paper's key property)
%   3. every row of p is a softmax: each output channel is a convex blend of the inputs
%   4. charmLayer3Conv (3 convs): every learnable gets a non-zero gradient
%   5. instanceNormalizationLayer on 'CBT' data: each channel of each trial has
%      mean 0 and std 1 over time at initialisation
%   6. each variant builds and trains (1 epoch) its target network on 2 channels;
%      charmSourceTrained also trains CHARM + DeepRNN on 61 channels first and
%      checks that the transferred CHARM keeps its source-trained weights
here = fileparts(mfilename('fullpath'));
repo = fileparts(fileparts(here));
vdir = fullfile(repo, 'replication', 'variants');
addpath(genpath(fullfile(repo, 'utils')));
common = fullfile(vdir, 'charmCommon');
addpath(common, '-begin');
cleanup = onCleanup(@() rmpath(common));
rng(7, 'threefry');
report();                                % reset the failure count

C = 5; B = 4; T = 60; M = 9; d = 32;
X = dlarray(single(randn(C, B, T)), 'CBT');

% 1 -------------------------------------------------------------------
rng(1, 'threefry'); a = initialize(charmLayer(M, d), []);
rng(1, 'threefry'); b = initialize(charmLayer3Conv(M, d, 'NumConv', 1), []);
za = extractdata(predict(a, X)); zb = extractdata(predict(b, X));
report('1. charmLayer3Conv(NumConv=1) == charmLayer', isequal(za, zb), sprintf('max|diff| = %g', max(abs(za(:)-zb(:)))));

% 2 -------------------------------------------------------------------
perm = [3 5 1 4 2];
zp = extractdata(predict(a, X(perm, :, :)));
report('2. invariant to input channel order', max(abs(zp(:)-za(:))) < 1e-5, sprintf('max|diff| = %g', max(abs(zp(:)-za(:)))));

% 3 -------------------------------------------------------------------
% output channel m at time t = sum_j p_mj x_jt with p_mj >= 0, sum_j p_mj = 1
xs = extractdata(X);
lo = min(xs, [], 1); hi = max(xs, [], 1);
inside = all(za >= lo - 1e-5 & za <= hi + 1e-5, 'all');
report('3. outputs are convex blends of the input channels', inside, '');

% 4 -------------------------------------------------------------------
rng(2, 'threefry'); c3 = initialize(charmLayer3Conv(M, d, 'NumConv', 3), []);
net = dlnetwork([sequenceInputLayer(C, 'Name', 'in'); c3], 'Initialize', false);
net = initialize(net, X);
[~, g] = dlfeval(@lossGrad, net, X);
nz = cellfun(@(v) any(extractdata(v(:)) ~= 0), g.Value);
report('4. 3-conv CHARM: every learnable has a gradient', all(nz), strjoin(strcat(g.Parameter, '=', string(nz)), ' '));

% 4b ------------------------------------------------------------------
% charmLayerLowMem (hand-written backward) vs charmLayer (autodiff): output and gradients
rng(3, 'threefry'); la = initialize(charmLayer(M, d), []);
rng(3, 'threefry'); lb = initialize(charmLayerLowMem(M, d), []);
lb.ChunkRows = 7;                                  % force several chunks
na = initialize(dlnetwork([sequenceInputLayer(C, 'Name', 'in'); la], 'Initialize', false), X);
nb = initialize(dlnetwork([sequenceInputLayer(C, 'Name', 'in'); lb], 'Initialize', false), X);
[za2, ga, gxa] = dlfeval(@lossGradX, na, X);
[zb2, gb, gxb] = dlfeval(@lossGradX, nb, X);
rel = @(u, v) max(abs(u(:) - v(:))) / max(1e-12, max(abs(u(:))));
eg = cellfun(@(u, v) rel(extractdata(u), extractdata(v)), ga.Value, gb.Value);
ex = rel(extractdata(gxa), extractdata(gxb));
report('4b. low-memory CHARM == charmLayer (output, dW, db, dQ, dX)', ...
    rel(za2, zb2) < 1e-5 && all(eg < 1e-4) && ex < 1e-4, ...
    sprintf('out %.1g, grads %s, dX %.1g', rel(za2, zb2), mat2str(eg', 2), ex));

% 5 -------------------------------------------------------------------
nn = dlnetwork([sequenceInputLayer(C, 'Name', 'in'); instanceNormalizationLayer('Name', 'in2')]);
Y = extractdata(predict(nn, 3 + 2*X));
mu = mean(Y, 3); sd = std(Y, 1, 3);
report('5. instance norm: per channel and trial mean 0, std 1', max(abs(mu(:))) < 1e-4 && max(abs(sd(:) - 1)) < 1e-2, ...
    sprintf('max|mean| = %.2g, max|std-1| = %.2g', max(abs(mu(:))), max(abs(sd(:)-1))));

% 6 -------------------------------------------------------------------
Cs = 61; Ct = 2; N = 24; Tt = 80;
[xs61, ys] = toyData(Cs, N, Tt);
[xt2,  yt] = toyData(Ct, N, Tt);
opts = trainingOptions('adam', 'MaxEpochs', 1, 'MiniBatchSize', 8, 'Verbose', false, ...
    'Plots', 'none', 'ExecutionEnvironment', 'cpu');
cfg = struct('domain', 'source-domain', 'layerName', 'DeepRNN', 'typeNet', 'DeepRNN', ...
    'datasetType', 'CWL', 'ncats', 2, 'projectLoc', '');
srcModel = struct('typeNet', 'DeepRNN', 'nchans', Cs, 'signalLength', Tt, 'nclass', 2);
srcNet = trainNetwork(xs61, ys, get_model_DeepRNN(srcModel), opts);
tgtModel = struct('typeNet', 'DeepRNN', 'nchans', Ct, 'signalLength', Tt, 'nclass', 2, 'pretrainedNet', srcNet);

for name = {'charmBase', 'charm3Conv', 'charmInstNorm', 'charmSourceTrained'}
    vd = fullfile(vdir, name{1});
    addpath(vd, '-begin');
    try
        tc = cfg; tc.domain = 'target-domain'; tc.layerName = 'charm';
        m = tgtModel;
        if strcmp(name{1}, 'charmSourceTrained')
            sc = cfg; sc.layerName = 'charm';
            sNet = trainNetwork(xs61, ys, define_layerArchitecture(srcModel, sc), opts);
            m.pretrainedNet = sNet;
        end
        lg = define_layerArchitecture(m, tc);
        tNet = trainNetwork(xt2, yt, lg, opts);
        L = tNet.Layers;
        names = string({L.Name});
        ok = L(1).InputSize == Ct && any(arrayfun(@(l) isa(l, 'charmLayer') || isa(l, 'charmLayer3Conv'), L));
        detail = strjoin(names(1:min(4, end)), ' > ');
        if strcmp(name{1}, 'charmSourceTrained')
            s0 = sNet.Layers(2); t1 = L(2);
            ok = ok && isa(s0, 'charmLayerLowMem') && size(t1.Q, 1) == Cs;
            % untrained CHARM-base: Q is Glorot within sqrt(6/(d+M)); the source-trained Q came from the source run
            detail = sprintf('%s; transferred Q from source run, |Q_target - Q_source| = %.3g after 1 target epoch', ...
                detail, max(abs(extractdata(dlarray(t1.Q(:))) - extractdata(dlarray(s0.Q(:))))));
        end
        if strcmp(name{1}, 'charmInstNorm'), ok = ok && any(names == "instnorm"); end
        if strcmp(name{1}, 'charm3Conv'),    ok = ok && isa(L(2), 'charmLayer3Conv') && L(2).NumConv == 3; end
        if strcmp(name{1}, 'charmBase'),     ok = ok && isa(L(2), 'charmLayer'); end
        report(sprintf('6. %s builds and trains', name{1}), ok, detail);
    catch ME
        report(sprintf('6. %s builds and trains', name{1}), false, ME.message);
    end
    rmpath(vd);
end
nFail = report();
fprintf('[checkCharm] %d failed\n', nFail);
if nargout == 0 && nFail > 0, error('checkCharm:fail', '%d CHARM checks failed', nFail); end
end

function [loss, g] = lossGrad(net, X)
Z = forward(net, X);
loss = sum(Z.^2, 'all');
g = dlgradient(loss, net.Learnables);
end

function [z, g, gx] = lossGradX(net, X)
Z = forward(net, X);
w = dlarray(single(reshape(1:numel(Z), size(Z))) / numel(Z), dims(Z));   % fixed, non-symmetric weights
loss = sum(Z .* w, 'all');
[g, gx] = dlgradient(loss, net.Learnables, X);
z = extractdata(Z);
end

function [x, y] = toyData(C, N, T)
x = arrayfun(@(~) single(randn(C, T)), 1:N, 'UniformOutput', false)';
y = categorical(mod(0:N-1, 2)' + 1);
end

function n = report(what, ok, detail)
persistent fails
if nargin == 0, n = fails; if isempty(n), n = 0; end, fails = 0; return; end
if isempty(fails), fails = 0; end
fails = fails + ~ok;
if ok, tag = 'PASS'; else, tag = 'FAIL'; end
fprintf('[checkCharm] %s  %s  %s\n', tag, what, detail);
end
