function layers = get_pLayer_phaseLSTM(model)
%GET_PLAYER_PHASELSTM  (replication variant 'pv_pal12Oct25', opt-in)
% Shadows utils/modelstuff/modeldefinitions/get_pLayer_phaseLSTM.m ONLY while
% runReplication(..., 'variant', 'pv_pal12Oct25') has this folder first on the path.
% Builds the network with the ORIGINAL function and swaps the layer named
% 'phaseLSTM' for phaseAwareLSTM_pal12Oct25_fast (same size, name, OutputMode).
layers = swapPhaseLayer(model, mfilename('fullpath'), @phaseAwareLSTM_pal12Oct25_fast);
end
