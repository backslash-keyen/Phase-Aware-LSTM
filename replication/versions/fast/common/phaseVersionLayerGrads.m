function g = phaseVersionLayerGrads(layer, X, args)
%PHASEVERSIONLAYERGRADS  Backward pass shared by the fast phase-layer version classes.
% args = the varargin of backward(layer, X, ...); the documented syntaxes for a
% layer with one output and four state outputs:
%   backward(layer, X, Y, dLdY, memory)
%   backward(layer, X, Y, dLdY, dLdS1..dLdS4, memory)
%   backward(layer, X, Y, S1..S4, dLdY, dLdS1..dLdS4, memory)
memory = args{end};
rest   = args(1:end-1);
dS     = {[], [], [], []};
switch numel(rest)
    case 2,  dLdY = rest{2};
    case 6,  dLdY = rest{2}; dS = rest(3:6);
    case 10, dLdY = rest{6}; dS = rest(7:10);
    otherwise
        error('phaseVersion:backwardSyntax', 'Unexpected backward call with %d arguments.', numel(args) + 2);
end
for k = 1:4
    if ~isempty(dS{k}), dS{k} = raw(stripdims(dlarray(dS{k}))); end
end
Xd = raw(stripdims(dlarray(X, 'CBT')));
dY = raw(stripdims(dlarray(dLdY)));
g  = phaseVersionEngine.grads(layer.config(), string(layer.OutputMode), Xd, dY, dS, memory);
end

function v = raw(v)
if isa(v, 'dlarray'), v = extractdata(v); end
end
