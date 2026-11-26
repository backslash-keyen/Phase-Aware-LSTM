function out = capPhaseGrads(out)
%CAPPHASEGRADS  Overflow-safe gradient cap for the fast phase-aware LSTM layers.
%
% Why: the phase projection's gradients explode with sequence length (about
% 1e25-1e27 at T = 500 on CWL fold 2). trainNetwork's GradientThreshold
% 'l2norm' clip is computed in single precision, the squared norm overflows
% to Inf, the clip factor threshold/Inf is 0, and every update to the layer is
% zeroed: the layer stays at its initial values while the rest of the network
% trains (found 26 Nov 2026, replication/versions/checkCrossFeed + cf_probe).
%
% What: when the 'gradCap' variant is on the path (phaseGradCap returns a
% finite cap), every learnable gradient out{2:end} of a layer's backward() is
% rescaled, in double, to L2 norm <= cap. Direction is kept; gradients below
% the cap are untouched. With cap >= the options' GradientThreshold (1.25),
% trainNetwork's own l2norm clip then acts on a finite norm, so the net effect
% is exactly the clip the options ask for. A gradient containing Inf or NaN is
% set to zero (no direction can be recovered). out{1} (dL/dX) is not changed.
% Without the variant (no phaseGradCap on the path) this does nothing.
% capPhaseGrads('stats') returns the pre-cap norms seen so far ([calls x
% learnables], Inf/NaN kept) and capPhaseGrads('reset') clears them.
persistent normsSeen
if ischar(out) || isstring(out)
    if strcmp(out, 'reset'), normsSeen = []; end
    out = normsSeen;
    return
end
if exist('phaseGradCap', 'file') ~= 2, return; end
cap = phaseGradCap();
if ~isfinite(cap), return; end
row = nan(1, numel(out) - 1);
for k = 2:numel(out)
    g = out{k};
    if isempty(g), continue; end
    wasDl = isa(g, 'dlarray');
    if wasDl, v = extractdata(g); else, v = g; end
    n = norm(double(gather(v(:))));          % norm() rescales internally: no overflow
    row(k-1) = n;
    if ~isfinite(n)
        v = zeros(size(v), 'like', v);
    elseif n > cap
        v = v * cast(cap / n, 'like', v);
    else
        continue
    end
    if wasDl, out{k} = dlarray(v); else, out{k} = v; end
end
if size(normsSeen, 1) < 20000, normsSeen(end+1, 1:numel(row)) = row; end
end
