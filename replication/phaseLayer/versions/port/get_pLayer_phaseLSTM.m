function layers = get_pLayer_phaseLSTM(model)
%GET_PLAYER_PHASELSTM  (replication variant 'pc_port', opt-in)
% Builds the network with the original function and swaps the layer named
% 'phaseLSTM' for phaseAwareLSTMCorr version 'port' (phaseCorrVersions.m).
layers = swapPhaseCorr(model, mfilename('fullpath'), 'port');
end
