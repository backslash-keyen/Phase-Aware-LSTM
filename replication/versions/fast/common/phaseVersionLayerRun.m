function [Y, hS, phS, cRS, cIS, memory] = phaseVersionLayerRun(layer, X, keep)
%PHASEVERSIONLAYERRUN  Forward pass shared by the fast phase-layer version classes.
cfg = layer.config();
Xd  = raw(stripdims(dlarray(X, 'CBT')));
[W3, b3] = deal([]);
if isprop(layer, 'W3'), W3 = raw(layer.W3); end
if isprop(layer, 'b3'), b3 = raw(layer.b3); end
[Yd, h, ph, cR, cI, memory] = phaseVersionEngine.run(cfg, string(layer.OutputMode), Xd, ...
    raw(layer.W1), raw(layer.W2), W3, raw(layer.b1), raw(layer.b2), b3, ...
    raw(layer.U), raw(layer.Q), keep);
if string(layer.OutputMode) == "sequence"
    Y = dlarray(Yd, 'CBT');
else
    Y = dlarray(Yd, 'CB');
end
hS = dlarray(h); phS = dlarray(ph); cRS = dlarray(cR); cIS = dlarray(cI);
end

function v = raw(v)
if isa(v, 'dlarray'), v = extractdata(v); end
end
