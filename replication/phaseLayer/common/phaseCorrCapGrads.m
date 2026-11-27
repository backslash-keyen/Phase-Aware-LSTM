function grads = phaseCorrCapGrads(grads, cap)
%PHASECORRCAPGRADS  Overflow-safe per-learnable gradient cap (correction 'gradCap').
%
%   grads = phaseCorrCapGrads(grads, cap)   rescale each cell to L2 norm <= cap
%   phaseCorrCapGrads(grads, Inf)           record the norms only
%   N = phaseCorrCapGrads('stats')          norms seen so far [calls x learnables]
%   phaseCorrCapGrads('reset')
%
% Why: trainNetwork clips each learnable to GradientThreshold (1.25 in the
% tuned CWL C2 PhaseAware options) with the 'l2norm' method in single
% precision. The phase layer's gradients reach 1e21 to 1e30 at T = 500, the
% squared norm overflows to Inf, the clip factor 1.25/Inf is 0 and every
% update of the layer is zeroed. Here the norm is taken in double (norm()
% rescales internally), so the cap acts on a finite number; afterwards
% trainNetwork's own clip sees a norm <= cap and the combined effect is the
% clip the options ask for. A gradient containing Inf or NaN is zeroed.
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
