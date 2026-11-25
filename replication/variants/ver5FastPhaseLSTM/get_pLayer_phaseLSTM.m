function layers = get_pLayer_phaseLSTM(model)
%GET_PLAYER_PHASELSTM  (replication variant 'ver5FastPhaseLSTM', opt-in)
% Shadows utils/modelstuff/modeldefinitions/get_pLayer_phaseLSTM.m ONLY when
% runReplication(..., 'variant', 'ver5FastPhaseLSTM') puts this folder first on
% the path. The original file is not changed.
%
% Builds the network with the ORIGINAL function, then swaps the layer named
% 'phaseLSTM' for phaseAwareLSTMVer5Fast with the same size, name and
% OutputMode. Everything else in the graph (freezing, EEGNet reshape,
% Transformer re-routing) is exactly what the original builds.
layers = callOriginal_ver5FastPhaseLSTM('get_pLayer_phaseLSTM', model);
old = layers.Layers(strcmp({layers.Layers.Name}, 'phaseLSTM'));
assert(isscalar(old) && isa(old, 'phaseAwareLSTM'), ...
    'ver5FastPhaseLSTM: expected one phaseAwareLSTM layer named ''phaseLSTM''.');
fast = phaseAwareLSTMVer5Fast(old.NumHiddenUnits, ...
    'Name', old.Name, 'OutputMode', old.OutputMode);
layers = replaceLayer(layers, 'phaseLSTM', fast);
end
