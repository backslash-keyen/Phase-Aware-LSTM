function layers = get_pLayer_phaseLSTM(model)
%GET_PLAYER_PHASELSTM  (replication variant 'pc_ideal', opt-in)
% Builds the network with the original function and swaps the layer named
% 'phaseLSTM' for phase layer version 'ideal' (phaseCorrVersions.m).
layers = swapPhaseCorr(model, mfilename('fullpath'), 'ideal');
end
