function layers = swapPhaseCorr(model, callerFile, versionId)
%SWAPPHASECORR  Build the network with the repository's own get_pLayer_phaseLSTM,
% then replace the layer named 'phaseLSTM' by phaseAwareLSTMCorr(versionId).
d = fileparts(callerFile);
rmpath(d);
restore = onCleanup(@() addpath(d, '-begin'));
layers = get_pLayer_phaseLSTM(model);
clear restore
old = layers.Layers(strcmp({layers.Layers.Name}, 'phaseLSTM'));
assert(isscalar(old) && isa(old, 'phaseAwareLSTM'), ...
    'swapPhaseCorr: expected one phaseAwareLSTM layer named ''phaseLSTM''.');
if startsWith(versionId, 'ideal'), ctor = @phaseAwareLSTMIdeal; else, ctor = @phaseAwareLSTMCorr; end
fprintf('[pc] phase layer = %s version %s\n', func2str(ctor), versionId);
layers = replaceLayer(layers, 'phaseLSTM', ctor(old.NumHiddenUnits, ...
    'Name', old.Name, 'OutputMode', old.OutputMode, 'Version', versionId));
end
