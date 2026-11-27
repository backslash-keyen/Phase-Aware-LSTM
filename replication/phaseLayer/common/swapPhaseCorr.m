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
fprintf('[pc] phase layer = phaseAwareLSTMCorr version %s\n', versionId);
layers = replaceLayer(layers, 'phaseLSTM', phaseAwareLSTMCorr(old.NumHiddenUnits, ...
    'Name', old.Name, 'OutputMode', old.OutputMode, 'Version', versionId));
end
