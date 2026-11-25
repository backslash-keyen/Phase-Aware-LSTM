function layers = get_pLayer_phaseLSTM(model)
%GET_PLAYER_PHASELSTM  (replication variant 'notrigPhaseLSTM', opt-in)
% Shadows utils/modelstuff/modeldefinitions/get_pLayer_phaseLSTM.m ONLY when
% runReplication(..., 'variant', 'notrigPhaseLSTM') puts this folder first on
% the path. The original file is not changed.
%
% DIAGNOSTIC / reproduction: the Nov 2025 layer behind the paper's CWL
% phase-lstm results (L:\plstm\helperLibrary\modelframe\phaseAwareLSTM_notrig.m,
% copied unchanged). The Nov 2025 get_pLayer_phaseLSTM_notrig.m differs from
% the released get_pLayer_phaseLSTM.m only in the class it constructs.
% Builds the network with the ORIGINAL function, then swaps the layer named
% 'phaseLSTM' for phaseAwareLSTM_notrig with the same size, name and
% OutputMode. Everything else in the graph (freezing, EEGNet reshape,
% Transformer re-routing) is exactly what the original builds.
layers = callOriginal_notrigPhaseLSTM('get_pLayer_phaseLSTM', model);
old = layers.Layers(strcmp({layers.Layers.Name}, 'phaseLSTM'));
assert(isscalar(old) && isa(old, 'phaseAwareLSTM'), ...
    'notrigPhaseLSTM: expected one phaseAwareLSTM layer named ''phaseLSTM''.');
fast = phaseAwareLSTM_notrig(old.NumHiddenUnits, ...
    'Name', old.Name, 'OutputMode', old.OutputMode);
layers = replaceLayer(layers, 'phaseLSTM', fast);
end
