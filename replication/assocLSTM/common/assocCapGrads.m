function grads = assocCapGrads(grads, cap)
%ASSOCCAPGRADS  Overflow-safe per-learnable gradient cap for associativeLSTMLayer.
%
%   grads = assocCapGrads(grads, cap)   rescale each cell to L2 norm <= cap (Inf: no change)
%   N = assocCapGrads('stats')          norms seen so far [calls x learnables]
%   assocCapGrads('reset')
%
% trainNetwork clips with GradientThreshold in single precision; a squared
% norm above ~3e38 overflows and zeroes the update (see the phase-layer
% gradCap correction). The norm here is taken in double. The norms are
% always recorded, so a check can see whether a version ever needs a cap.
persistent normsSeen
if ischar(grads) || isstring(grads)
    if strcmp(grads, 'reset'), normsSeen = []; end
    grads = normsSeen;
    return
end
row = nan(1, numel(grads));
for k = 1:numel(grads)
    v = grads{k};
    if isempty(v), continue; end
    n = norm(double(gather(v(:))));
    row(k) = n;
    if ~isfinite(cap), continue; end
    if ~isfinite(n)
        grads{k} = zeros(size(v), 'like', v);
    elseif n > cap
        grads{k} = v * cast(cap / n, 'like', v);
    end
end
if size(normsSeen, 1) < 50000, normsSeen(end+1, 1:numel(row)) = row; end
end
