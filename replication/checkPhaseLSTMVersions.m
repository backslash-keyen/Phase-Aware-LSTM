function report = checkPhaseLSTMVersions(varargin)
%CHECKPHASELSTMVERSIONS  Compare the phase-aware LSTM speed versions with the original layer.
%
% Short test (a few minutes at most). Trains nothing, reads no data, writes
% no results. Run from the repository root:
%   addpath('replication');
%   report = checkPhaseLSTMVersions();              % H 30, B 197, T 500, GPU if available
%   report = checkPhaseLSTMVersions('H', 30, 'env', 'cpu', 'reps', 3);
%
% Versions
%   1 original   utils/.../phaseAwareLSTM.m              (reference)
%   2 fast       variants/fastPhaseLSTM                  should be bit-identical
%   3 fastest    variants/fastestPhaseLSTM               same model, float rounding
%   4 backward   variants/backwardPhaseLSTM              v3 forward + hand-written backward
%   5 fused      variants/fusedPhaseLSTM                 v4 with one fused GPU kernel per step
%
% 'profile', true also runs the MATLAB profiler on one forward+backward of
% the original layer and prints the 15 most expensive functions.
%
% Each layer sits in a dlnetwork after a sequenceInputLayer, as in training,
% so version 4's backward() is the one dlgradient uses. All versions get the
% same weights and input. For every output (Y, the four final states) and
% every gradient (W1 W2 U Q b1 b2) it reports
%   identical@T   bit-for-bit equal to the original at full length (single)
%   diff@Tcheck   largest difference relative to the largest value over the
%                 first Tcheck steps in double precision (see note in code)
% and it times one forward+backward pass (one training iteration).
%
% H is the projection size = channel count of the pretrained source model.
p = inputParser;
addParameter(p, 'H', 30);
addParameter(p, 'B', 197);      % MiniBatchSize, get_trainingOptions_DeepRNN_CWL
addParameter(p, 'T', 500);      % CWL signal length
addParameter(p, 'C', 2);        % target channels (Fp1, Fz)
addParameter(p, 'reps', 3);
addParameter(p, 'env', 'auto'); % 'auto' | 'gpu' | 'cpu'
addParameter(p, 'seed', 0);
addParameter(p, 'versions', 1:5);
addParameter(p, 'profile', false);
addParameter(p, 'trainSmoke', true); % 2-iteration trainNetwork run per version (training path)
addParameter(p, 'Tcheck', 20);  % steps used for the accuracy comparison
parse(p, varargin{:}); o = p.Results;

repDir  = fileparts(mfilename('fullpath'));
rootDir = fileparts(repDir);
oldPath = path; restorePath = onCleanup(@() path(oldPath)); %#ok<NASGU>
addpath(genpath(fullfile(rootDir, 'utils')));
for v = {'fastPhaseLSTM','fastestPhaseLSTM','backwardPhaseLSTM','fusedPhaseLSTM'}
    addpath(fullfile(repDir, 'variants', v{1}));
end

useGPU = strcmp(o.env, 'gpu') || (strcmp(o.env, 'auto') && canUseGPU);
names  = {'original','fast','fastest','backward','fused'};
make   = {@phaseAwareLSTM, @phaseAwareLSTMFast, @phaseAwareLSTMFastest, @phaseAwareLSTMBackward, @phaseAwareLSTMFused};
vers   = unique([1, o.versions(:)']);        % the original is always the reference

rng(o.seed, 'twister');
input  = sequenceInputLayer(o.C, 'MinLength', o.T, 'Name', 'input');
netRef = dlnetwork([input; phaseAwareLSTM(o.H, 'Name', 'phaseLSTM', 'OutputMode', 'sequence')]);
X = single(randn(o.C, o.B, o.T));
w = reshape(sin(single(1:o.H*o.B*o.T)), o.H, o.B, o.T);
if useGPU, X = gpuArray(X); w = gpuArray(w); end
% X stays a plain array and becomes a dlarray inside lossAndGrads, so it is
% NOT traced: in training the projection layer's input is data, never a
% variable being differentiated (tracing it breaks the original layer).

labels = {'Y','h','phase','memRe','memIm','dW1','dW2','dU','dQ','db1','db2'};
res = struct('version', {}, 'name', {}, 'seconds', {}, 'speedup', {}, 'identical', {}, ...
             'maxRelDiff', {}, 'detail', {}, 'vals', {});

% ---- 1) speed at full length, single precision (as in training) --------
for k = vers
    net = makeNet(make{k}, o.H, o.C, o.T, netRef, useGPU, 'single');
    ts = zeros(1, o.reps + 1);
    for r = 1:o.reps + 1                       % first call is warm-up
        tic;
        out = cell(1, 3);
        [out{:}] = dlfeval(@lossAndGrads, net, X, w);
        if useGPU, wait(gpuDevice); end
        ts(r) = toc;
    end
    if o.profile && k == 1
        profile('on');
        dlfeval(@lossAndGrads, net, X, w);
        if useGPU, wait(gpuDevice); end
        profile('off');
        showProfile(profile('info'), 15);
    end
    res(end+1).version = k; %#ok<AGROW>
    res(end).name    = names{k};
    res(end).seconds = median(ts(2:end));
    res(end).vals    = flatten(out);
end

% ---- 2) accuracy on a short sequence, double precision ------------------
% The original recurrence is very sensitive: two runs of the SAME original
% code in single vs double precision already differ by O(1) after ~50 of
% the 500 steps (rounding differences grow ~10x every few steps, and the
% atan2 phase can jump by 2*pi). Comparing versions at T = 500 therefore
% only measures that sensitivity. Accuracy is checked on the first Tcheck
% steps in double precision, where rounding is far below the differences a
% real bug would cause; the original's own single-vs-double difference
% over the same steps is printed as the rounding floor.
Xc = double(X(:,:,1:o.Tcheck)); wc = double(w(:,:,1:o.Tcheck));
acc = cell(1, numel(res));
for i = 1:numel(res)
    net = makeNet(make{res(i).version}, o.H, o.C, o.Tcheck, netRef, useGPU, 'double');
    out = cell(1, 3);
    [out{:}] = dlfeval(@lossAndGrads, net, Xc, wc);
    acc{i} = flatten(out);
end
net = makeNet(make{1}, o.H, o.C, o.Tcheck, netRef, useGPU, 'single');
out = cell(1, 3);
[out{:}] = dlfeval(@lossAndGrads, net, single(Xc), single(wc));
floorVals = flatten(out);
floorDiff = relDiffs(acc{1}, floorVals);

% the same sensitivity at full length, for the record
net = makeNet(make{1}, o.H, o.C, o.T, netRef, useGPU, 'double');
out = cell(1, 3);
[out{:}] = dlfeval(@lossAndGrads, net, double(X), double(w));
fullDiff = relDiffs(flatten(out), res(1).vals);

fprintf('\nphase-aware LSTM, %s, H=%d B=%d T=%d, forward+backward, median of %d\n', ...
        ternary(useGPU, 'GPU', 'CPU'), o.H, o.B, o.T, o.reps);
fprintf('%-3s %-10s %9s %8s %16s %22s\n', 'v', 'name', 'seconds', 'speedup', ...
        'identical@T', sprintf('max rel diff@T=%d', o.Tcheck));
for k = 1:numel(res)
    same = true;
    for j = 1:numel(labels), same = same && isequal(res(1).vals{j}, res(k).vals{j}); end
    d = relDiffs(acc{1}, acc{k});
    res(k).speedup    = res(1).seconds / res(k).seconds;
    res(k).identical  = same;
    res(k).maxRelDiff = max(d);
    res(k).detail     = cell2struct(num2cell(d), labels, 2);
    fprintf('%-3d %-10s %9.3f %7.1fx %16d %22.2e\n', res(k).version, res(k).name, ...
            res(k).seconds, res(k).speedup, same, res(k).maxRelDiff);
end
fprintf(['\nRounding floor: the ORIGINAL in single vs double precision differs by %.2e over the ' ...
         'first %d steps\nand by %.2e over all %d steps (Y %.2e).\n'], max(floorDiff), o.Tcheck, ...
         max(fullDiff), o.T, fullDiff(1));
fprintf(['Expected: fast identical@T = 1; every version''s diff@T=%d well below the rounding floor ' ...
         '(double precision).\nPer-output differences: report(k).detail\n\n'], o.Tcheck);
report = rmfield(res, 'vals');
if o.trainSmoke
    smoke = smokeTrain(make, names, [res.version], o, useGPU);
    [report.trainNetworkOK]  = smoke.ok;
    [report.trainNetworkMsg] = smoke.msg;
end
end

function smoke = smokeTrain(make, names, vers, o, useGPU)
% trainNetwork uses a different code path from dlnetwork/dlgradient (the
% legacy custom-layer strategy). Train each version for 2 iterations in a
% tiny network to make sure it works there too.
Ts = 50; N = 64;
rng(o.seed + 1, 'twister');
Xs = arrayfun(@(i) single(randn(o.C, Ts)), 1:N, 'UniformOutput', false)';
Ys = categorical(randi(2, N, 1));
opts = trainingOptions('adam', 'MaxEpochs', 1, 'MiniBatchSize', 32, 'Verbose', false, ...
    'Plots', 'none', 'ExecutionEnvironment', ternary(useGPU, 'gpu', 'cpu'));
smoke.ok = cell(1, numel(vers)); smoke.msg = cell(1, numel(vers));
fprintf('trainNetwork smoke test (2 iterations, T=%d):\n', Ts);
for i = 1:numel(vers)
    k = vers(i);
    layers = [sequenceInputLayer(o.C, 'MinLength', Ts, 'Name', 'input'); ...
              make{k}(o.H, 'Name', 'phaseLSTM', 'OutputMode', 'sequence'); ...
              lstmLayer(8, 'OutputMode', 'last'); fullyConnectedLayer(2); softmaxLayer; classificationLayer];
    try
        tic; trainNetwork(Xs, Ys, layers, opts); t = toc;
        smoke.ok{i} = true; smoke.msg{i} = sprintf('ok (%.1f s)', t);
    catch err
        smoke.ok{i} = false; smoke.msg{i} = getReport(err, 'extended', 'hyperlinks', 'off');
    end
    fprintf('  %-10s %s\n', names{k}, smoke.msg{i});
end
fprintf('\n');
end

function net = makeNet(ctor, H, C, T, netRef, useGPU, precision)
net = dlnetwork([sequenceInputLayer(C, 'MinLength', T, 'Name', 'input'); ...
                 ctor(H, 'Name', 'phaseLSTM', 'OutputMode', 'sequence')]);
L = net.Learnables;
assert(isequal(L.Parameter, netRef.Learnables.Parameter), 'learnables differ');
L.Value = netRef.Learnables.Value;
net.Learnables = L;
cv = str2func(precision);       % @single or @double, both work on dlarray
net = dlupdate(cv, net);
if useGPU, net = dlupdate(@gpuArray, net); end
end

function d = relDiffs(A, Bv)
d = zeros(1, numel(A));
for j = 1:numel(A)
    a = A{j}; b = Bv{j};
    d(j) = max(abs(a(:) - b(:))) / max(max(abs(a(:))), realmin);
end
end

function [Y, S, gL] = lossAndGrads(net, Xraw, w)
X = dlarray(Xraw, 'CBT');            % created here, so untraced (as in training)
% Acceleration 'none': dlnetwork's cached (accelerated) version of an
% Acceleratable custom layer breaks under dlfeval in R2025b, and the four
% versions should be compared on their own code.
[Y, S] = forward(net, X, 'Acceleration', 'none');
Y = stripdims(Y);
loss = sum(Y .* w, 'all');
gL = dlgradient(loss, net.Learnables);
end

function v = flatten(out)
% {Y, state table, learnable-gradient table} -> 11 plain arrays
[Y, S, gL] = out{:};
st = S.Value;                       % hiddenMagnitude, hiddenPhase, memoryReal, memoryImaginary
gv = gL.Value;                      % W1 W2 U Q b1 b2
v = [{Y}, st(:)', gv(:)'];
v = cellfun(@(a) gather(double(extractdata(dlarray(a)))), v, 'UniformOutput', false);
end

function showProfile(info, n)
ft = info.FunctionTable;
[~, order] = sort([ft.TotalTime], 'descend');
fprintf('\nProfiler, original layer, one forward+backward (top %d by total time):\n', n);
fprintf('%10s %10s %8s  %s\n', 'total s', 'self s', 'calls', 'function');
for i = order(1:min(n, numel(order)))
    fprintf('%10.3f %10.3f %8d  %s\n', ft(i).TotalTime, ft(i).TotalTime - sum([ft(i).Children.TotalTime]), ...
            ft(i).NumCalls, ft(i).FunctionName);
end
end

function s = ternary(c, a, b)
if c, s = a; else, s = b; end
end
