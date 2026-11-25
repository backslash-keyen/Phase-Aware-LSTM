% Read-only: gradient norm of the released phaseAwareLSTM (autodiff) vs sequence length T,
% single and double precision, at initialisation, on fold-2 target data (32 samples).
warning('off', 'all');
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));   % repo root (file in replication/diagnostics/)
cd(root);
addpath(genpath(fullfile(root, 'utils')));
S = load(fullfile(root, 'datasets', 'folds', 'CWL', 'NEC21_C2', 'targetdomain1', 'target_fold2.mat'), 'xtrain');
X0 = S.xtrain(:, :, 1:32); clear S
rng(2);
net = dlnetwork([sequenceInputLayer(2, 'Name', 'input'); ...
                 phaseAwareLSTM(61, 'Name', 'phaseLSTM', 'OutputMode', 'sequence')]);
for prec = {'single', 'double'}
    for T = [10 25 50 100 200 300 500]
        X = dlarray(cast(permute(X0(:, 1:T, :), [1 3 2]), prec{1}), 'CBT');
        n = dlupdate(@(v) cast(v, prec{1}), net);
        rng(0); w = cast(randn(61, 32, T), prec{1});
        [g, gq, frac] = dlfeval(@lg, n, X, w);
        fprintf('%-6s T=%3d  |dL/dParams| %.3g   |dL/dQ| %.3g   nonfinite frac %.3f\n', prec{1}, T, g, gq, frac);
    end
end

function [g, gq, frac] = lg(net, X, w)
Y = forward(net, X, 'Acceleration', 'none');
loss = sum(stripdims(Y) .* w, 'all') / size(w, 3);
G = dlgradient(loss, net.Learnables);
v = cellfun(@(x) extractdata(x(:)), G.Value, 'UniformOutput', false);
v = cat(1, v{:});
frac = mean(~isfinite(v));
g = norm(v(isfinite(v)));
gq = norm(extractdata(G.Value{strcmp(G.Parameter, 'Q')}(:)));
end
