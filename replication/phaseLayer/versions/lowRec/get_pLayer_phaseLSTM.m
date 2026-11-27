function layers = get_pLayer_phaseLSTM(model)
%GET_PLAYER_PHASELSTM  (replication variant 'pc_lowRec', opt-in)
% Builds the network with the original function and swaps the layer named
% 'phaseLSTM' for phaseAwareLSTMCorr version 'lowRec' (phaseCorrVersions.m).
layers = swapPhaseCorr(model, mfilename('fullpath'), 'lowRec');
end
