function layers = swapPhaseLayer(model, callerFile, ctor)
%SWAPPHASELAYER  Build the network with the repository's own get_pLayer_phaseLSTM,
% then replace the layer named 'phaseLSTM' by ctor(H, Name, OutputMode).
% callerFile is the shadowing get_pLayer_phaseLSTM.m; its folder is taken
% off the path while the original runs, then put back on top.
d = fileparts(callerFile);
rmpath(d);
restore = onCleanup(@() addpath(d, '-begin'));
layers = get_pLayer_phaseLSTM(model);
clear restore
old = layers.Layers(strcmp({layers.Layers.Name}, 'phaseLSTM'));
assert(isscalar(old) && isa(old, 'phaseAwareLSTM'), ...
    'swapPhaseLayer: expected one phaseAwareLSTM layer named ''phaseLSTM''.');
layers = replaceLayer(layers, 'phaseLSTM', ctor(old.NumHiddenUnits, ...
    'Name', old.Name, 'OutputMode', old.OutputMode));
end
