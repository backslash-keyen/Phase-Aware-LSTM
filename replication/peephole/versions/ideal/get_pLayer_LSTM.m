function layers = get_pLayer_LSTM(model)
%GET_PLAYER_LSTM  (replication variant 'ph_ideal', opt-in)
% Builds the LSTM projection with the original function and swaps its lstmLayer
% for peepholeLSTMLayer version 'ideal' (peepholeVersions.m).
layers = swapPeephole(model, mfilename('fullpath'), 'ideal');
end
