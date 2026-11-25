function out = phaseVersionFinishGrads(out, g, X, n)
%PHASEVERSIONFINISHGRADS  Wrap backward() outputs as dlarrays (trainNetwork's
% legacy custom-layer path calls extractdata on each), give dLdX the format
% of X, and add zeros for any state-input derivatives the caller asks for
% (the states start from zero at every call).
H = size(g.U, 2);
B = size(g.X, 2);
for k = numel(out)+1:n
    out{k} = zeros(H, B, 'like', g.X);
end
out = out(1:n);
for k = 1:numel(out)
    if isa(out{k}, 'dlarray'), continue; end
    if k == 1 && isa(X, 'dlarray') && ~isempty(dims(X))
        out{k} = dlarray(out{k}, dims(X));
    else
        out{k} = dlarray(out{k});
    end
end
end
