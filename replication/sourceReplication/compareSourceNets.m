function T = compareSourceNets(fileA, fileB)
%COMPARESOURCENETS  Parameter-by-parameter comparison of two saved source nets.
%
%   T = compareSourceNets(fileA, fileB)
%
%   Each file holds a variable 'net' (as saved by perfold_saveModel.m or
%   extractHPCSourceNets.m). For every learnable and batch-norm statistic, T
%   gives: identical (bit for bit), maximum absolute difference, relative norm
%   ||a-b||/||b|| and cosine similarity. Two runs with the same seed, generator
%   and data order on one machine should be identical; nets that only share the
%   initialisation (same seed, different platform) keep cosine of about 0.6 on
%   the LSTM weights; independent initialisations give about 0.

A = load(fileA, 'net'); B = load(fileB, 'net');
pa = params(A.net); pb = params(B.net);
names = intersect(fieldnames(pa), fieldnames(pb), 'stable');
n = numel(names);
identical = false(n,1); maxAbs = nan(n,1); relNorm = nan(n,1); cosine = nan(n,1);
for k = 1:n
    a = double(pa.(names{k})(:)); b = double(pb.(names{k})(:));
    if numel(a) ~= numel(b), continue; end
    identical(k) = isequal(a, b);
    maxAbs(k)  = max(abs(a - b));
    relNorm(k) = norm(a - b) / max(norm(b), eps);
    cosine(k)  = (a' * b) / max(norm(a) * norm(b), eps);
end
T = table(names, identical, maxAbs, relNorm, cosine, ...
    'VariableNames', {'parameter','identical','maxAbs','relNorm','cosine'});
end

function p = params(net)
p = struct();
L = net.Layers;
props = {'InputWeights','RecurrentWeights','Bias','Weights','Offset','Scale','TrainedMean','TrainedVariance'};
for i = 1:numel(L)
    for j = 1:numel(props)
        if isprop(L(i), props{j}) && ~isempty(L(i).(props{j}))
            p.(matlab.lang.makeValidName([L(i).Name '_' props{j}])) = L(i).(props{j});
        end
    end
end
end
