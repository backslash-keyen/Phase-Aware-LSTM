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
% every gradient (W1 W2 U Q b1 b2) it reports the largest
% difference relative to the largest value, and it times one
% forward+backward pass (one training iteration of this layer).
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

res = struct('version', {}, 'name', {}, 'seconds', {}, 'speedup', {}, ...
             'maxRelDiff', {}, 'identical', {}, 'detail', {});
for k = vers
    net = dlnetwork([input; make{k}(o.H, 'Name', 'phaseLSTM', 'OutputMode', 'sequence')]);
    L = net.Learnables;
    assert(isequal(L.Parameter, netRef.Learnables.Parameter), 'learnables differ for %s', names{k});
    L.Value = netRef.Learnables.Value;
    net.Learnables = L;
    if useGPU, net = dlupdate(@gpuArray, net); end

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
    vals = flatten(out);
    res(end+1).version = k; %#ok<AGROW>
    res(end).name    = names{k};
    res(end).seconds = median(ts(2:end));
    res(end).vals    = vals;
end

ref = res(1);
fprintf('\nphase-aware LSTM, %s, H=%d B=%d T=%d, forward+backward, median of %d\n', ...
        ternary(useGPU, 'GPU', 'CPU'), o.H, o.B, o.T, o.reps);
fprintf('%-3s %-10s %9s %8s %13s %10s\n', 'v', 'name', 'seconds', 'speedup', 'max rel diff', 'identical');
labels = {'Y','h','phase','memRe','memIm','dW1','dW2','dU','dQ','db1','db2'};
for k = 1:numel(res)
    d = zeros(1, numel(labels)); same = true;
    for j = 1:numel(labels)
        a = ref.vals{j}; b = res(k).vals{j};
        d(j) = max(abs(a(:) - b(:))) / max(max(abs(a(:))), realmin('single'));
        same = same && isequal(a, b);
    end
    res(k).speedup    = ref.seconds / res(k).seconds;
    res(k).maxRelDiff = max(d);
    res(k).identical  = same;
    res(k).detail     = cell2struct(num2cell(d), labels, 2);
    fprintf('%-3d %-10s %9.3f %7.1fx %13.2e %10d\n', res(k).version, res(k).name, ...
            res(k).seconds, res(k).speedup, res(k).maxRelDiff, same);
end
fprintf(['\nExpected: fast identical = 1; fastest, backward and fused max rel diff around 1e-6 ' ...
         'or below (single precision).\nPer-output differences: report(k).detail\n\n']);
report = rmfield(res, 'vals');
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
