% Read-only: matched setting, layer alone (as checkPhaseLSTMVersions does). Original autodiff
% phaseAwareLSTM (needs a TEMPORARY local edit of utils/.../phaseAwareLSTM.m: clip +-5 on G and H,
% b1 without the b1(2H:end)=0 line, b2 = unit-forget; restore it afterwards)
% vs fused phaseAwareLSTMHalfAngMatched. Same parameters, double precision, T = 20, same input.
warning('off', 'all');
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));   % repo root (file in replication/diagnostics/)
cd(root);
addpath(genpath(fullfile(root, 'utils')));
addpath(fullfile(root, 'replication', 'variants', 'halfAngMatchedPhaseLSTM'), '-end');
H = 61; C = 2; B = 32; T = 20;
S = load(fullfile(root, 'datasets', 'folds', 'CWL', 'NEC21_C2', 'targetdomain1', 'target_fold2.mat'), 'xtrain');
X0 = S.xtrain(:, 1:T, 1:B); clear S
rng(5);
netO = dlnetwork([sequenceInputLayer(C, 'Name', 'input'); phaseAwareLSTM(H, 'Name', 'p', 'OutputMode', 'sequence')]);
netF = dlnetwork([sequenceInputLayer(C, 'Name', 'input'); phaseAwareLSTMHalfAngMatched(H, 'Name', 'p', 'OutputMode', 'sequence')]);
fprintf('classes: %s | %s\n', class(netO.Layers(2)), class(netF.Layers(2)));
netO = dlupdate(@(v) gpuArray(double(v)), netO);
L = netF.Learnables; L.Value = netO.Learnables.Value; netF.Learnables = L;   % identical parameters
fprintf('b1 ones %d, b2 ones %d (original init, copied to fused)\n', nnz(ex(netO.Learnables.Value{strcmp(netO.Learnables.Parameter,'b1')}) == 1), nnz(ex(netO.Learnables.Value{strcmp(netO.Learnables.Parameter,'b2')}) == 1));
X = dlarray(gpuArray(double(permute(X0, [1 3 2]))), 'CBT');
rng(0); w = gpuArray(randn(H, B, T));
[YO, GO] = dlfeval(@lg, netO, X, w);
[YF, GF] = dlfeval(@lg, netF, X, w);
rd = @(a, b) max(abs(a(:) - b(:))) / max(max(abs(a(:))), realmin);
fprintf('Y      max rel diff %.3g (max |Y| %.3g)\n', rd(YO, YF), max(abs(YO(:))));
for k = 1:height(GO)
    a = ex(GO.Value{k}); b = ex(GF.Value{k});
    fprintf('d%-3s   max rel diff %.3g   |g| orig %.4g fused %.4g\n', GO.Parameter{k}, rd(a, b), norm(a(:)), norm(b(:)));
end

function [Y, G] = lg(net, X, w)
Y = forward(net, X, 'Acceleration', 'none');
loss = sum(stripdims(Y) .* w, 'all');
G = dlgradient(loss, net.Learnables);
Y = gather(extractdata(stripdims(Y)));
end

function v = ex(v)
if isa(v, 'dlarray'), v = extractdata(v); end
v = double(gather(v));
end
