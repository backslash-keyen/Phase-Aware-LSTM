function layers = get_pLayer_LSTM(model)
%GET_PLAYER_LSTM  (replication variant 'ph_gers2002', opt-in)
% Builds the LSTM projection with the original function and swaps its lstmLayer
% for peepholeLSTMLayer version 'gers2002' (peepholeVersions.m).
layers = swapPeephole(model, mfilename('fullpath'), 'gers2002');
end
