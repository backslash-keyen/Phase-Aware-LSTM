function report = checkFastPhaseLSTM(varargin)
%CHECKFASTPHASELSTM  Is phaseAwareLSTMFast bit-identical to phaseAwareLSTM, and how much faster?
%
% Short test (a minute or two). Trains nothing and writes no results.
% Run from the repository root:
%   addpath(genpath('utils')); addpath('replication/variants/fastPhaseLSTM');
%   report = checkFastPhaseLSTM();                 % CWL-sized: H 30, B 197, T 500
%   report = checkFastPhaseLSTM('H',30,'B',197,'T',500,'reps',3,'env','gpu');
%
% For each layer it runs forward + dlgradient of a fixed scalar loss with
% respect to every learnable and the input, on the same weights and data,
% then compares every output with isequal. It also times one
% forward+backward pass (the per-iteration cost inside trainNetwork).
p = inputParser;
addParameter(p, 'H', 30);       % NumHiddenUnits = source channel count
addParameter(p, 'B', 197);      % MiniBatchSize in get_trainingOptions_DeepRNN_CWL
addParameter(p, 'T', 500);      % CWL signal length
addParameter(p, 'C', 2);        % target channels (Fp1, Fz)
addParameter(p, 'reps', 3);
addParameter(p, 'env', 'auto'); % 'auto' | 'gpu' | 'cpu'
addParameter(p, 'seed', 0);
parse(p, varargin{:}); o = p.Results;

useGPU = strcmp(o.env, 'gpu') || (strcmp(o.env, 'auto') && canUseGPU);
rng(o.seed, 'twister');

ref  = phaseAwareLSTM(o.H, 'Name', 'phaseLSTM', 'OutputMode', 'sequence');
fast = phaseAwareLSTMFast(o.H, 'Name', 'phaseLSTM', 'OutputMode', 'sequence');
ref  = initialize(ref, networkDataLayout([o.C NaN NaN], 'CBT'));
P = struct();
for n = {'W1','W2','U','Q','b1','b2'}
    v = single(extractdata(dlarray(ref.(n{1}))));
    if useGPU, v = gpuArray(v); end
    P.(n{1}) = dlarray(v);
end
X = single(randn(o.C, o.B, o.T));
if useGPU, X = gpuArray(X); end
X = dlarray(X, 'CBT');

names = {'Y','hFinal','phaseFinal','memRe','memIm','dW1','dW2','dU','dQ','db1','db2','dX'};
[outR, tR] = runOnce(ref,  P, X, o.reps, useGPU);
[outF, tF] = runOnce(fast, P, X, o.reps, useGPU);

same = true;
fprintf('\n%-11s %-9s %s\n', 'output', 'isequal', 'max |diff|');
for k = 1:numel(names)
    a = gather(extractdata(outR{k})); b = gather(extractdata(outF{k}));
    eq = isequal(a, b);
    same = same && eq;
    fprintf('%-11s %-9d %.3g\n', names{k}, eq, max(abs(a(:) - b(:))));
end
fprintf('\nforward+backward, %s, H=%d B=%d T=%d (median of %d)\n', ...
    ternary(useGPU,'GPU','CPU'), o.H, o.B, o.T, o.reps);
fprintf('  original : %.2f s\n  fast     : %.2f s\n  speedup  : %.1fx\n', tR, tF, tR / tF);
fprintf('  bit-identical: %d\n\n', same);
report = struct('identical', same, 'tOriginal', tR, 'tFast', tF, 'speedup', tR / tF, ...
                'gpu', useGPU, 'H', o.H, 'B', o.B, 'T', o.T);
end

function [out, t] = runOnce(layer, P, X, reps, useGPU)
ts = zeros(1, reps);
for r = 1:reps
    tic;
    out = cell(1, 12);
    [out{:}] = dlfeval(@lossAndGrads, layer, P, X);
    if useGPU, wait(gpuDevice); end
    ts(r) = toc;
end
t = median(ts);
end

function varargout = lossAndGrads(layer, P, X)
for n = fieldnames(P)'
    layer.(n{1}) = P.(n{1});
end
[Y, h, ph, mr, mi] = forward(layer, X);
Yd = stripdims(Y);
w  = reshape(sin(single(1:numel(Yd))), size(Yd));   % fixed weights so every time step matters
loss = sum(Yd .* w, 'all') + sum(h, 'all') + 0.5 * sum(ph, 'all') ...
     + sum(mr .^ 2, 'all') + sum(mi, 'all');
[dW1, dW2, dU, dQ, db1, db2, dX] = dlgradient(loss, P.W1, P.W2, P.U, P.Q, P.b1, P.b2, X);
varargout = {Yd, h, ph, mr, mi, dW1, dW2, dU, dQ, db1, db2, stripdims(dX)};
end

function s = ternary(c, a, b)
if c, s = a; else, s = b; end
end
