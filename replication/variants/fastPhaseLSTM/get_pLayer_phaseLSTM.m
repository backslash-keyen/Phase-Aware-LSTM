function layers = get_pLayer_phaseLSTM(model)
%GET_PLAYER_PHASELSTM  (replication variant 'fastPhaseLSTM', opt-in)
% Shadows utils/modelstuff/modeldefinitions/get_pLayer_phaseLSTM.m ONLY when
% runReplication(..., 'variant', 'fastPhaseLSTM') puts this folder first on
% the path. The original file is not changed.
%
% Builds the network with the ORIGINAL function, then swaps the layer named
% 'phaseLSTM' for phaseAwareLSTMFast with the same size, name and
% OutputMode. Everything else in the graph (freezing, EEGNet reshape,
% Transformer re-routing) is exactly what the original builds.
layers = variantCallOriginal('get_pLayer_phaseLSTM', model);
old = layers.Layers(strcmp({layers.Layers.Name}, 'phaseLSTM'));
assert(isscalar(old) && isa(old, 'phaseAwareLSTM'), ...
    'fastPhaseLSTM: expected one phaseAwareLSTM layer named ''phaseLSTM''.');
fast = phaseAwareLSTMFast(old.NumHiddenUnits, ...
    'Name', old.Name, 'OutputMode', old.OutputMode);
layers = replaceLayer(layers, 'phaseLSTM', fast);
end
