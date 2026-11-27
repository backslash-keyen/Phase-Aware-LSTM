function layers = get_pLayer_LSTM(model)
%GET_PLAYER_LSTM  (replication variant 'ph_noPeep', opt-in)
% Builds the LSTM projection with the original function and swaps its lstmLayer
% for peepholeLSTMLayer version 'noPeep' (peepholeVersions.m).
layers = swapPeephole(model, mfilename('fullpath'), 'noPeep');
end
