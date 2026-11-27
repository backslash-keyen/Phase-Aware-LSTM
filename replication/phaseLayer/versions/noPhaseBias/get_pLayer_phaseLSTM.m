function layers = get_pLayer_phaseLSTM(model)
%GET_PLAYER_PHASELSTM  (replication variant 'pc_noPhaseBias', opt-in)
% Builds the network with the original function and swaps the layer named
% 'phaseLSTM' for phaseAwareLSTMCorr version 'noPhaseBias' (phaseCorrVersions.m).
layers = swapPhaseCorr(model, mfilename('fullpath'), 'noPhaseBias');
end
