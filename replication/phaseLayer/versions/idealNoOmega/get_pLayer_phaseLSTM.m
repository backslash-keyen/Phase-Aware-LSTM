function layers = get_pLayer_phaseLSTM(model)
%GET_PLAYER_PHASELSTM  (replication variant 'pc_idealNoOmega', opt-in)
% Builds the network with the original function and swaps the layer named
% 'phaseLSTM' for phase layer version 'idealNoOmega' (phaseCorrVersions.m).
layers = swapPhaseCorr(model, mfilename('fullpath'), 'idealNoOmega');
end
