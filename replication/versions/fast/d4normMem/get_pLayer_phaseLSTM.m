function layers = get_pLayer_phaseLSTM(model)
%GET_PLAYER_PHASELSTM  (replication variant 'pv_d4normMem', opt-in; use with 'gradCap')
% Shadows utils/modelstuff/modeldefinitions/get_pLayer_phaseLSTM.m ONLY while
% runReplication(..., 'variant', {'pv_d4normMem', 'gradCap'}) has this folder first on the path.
% Swaps the layer named 'phaseLSTM' for phaseAwareLSTMD4X: the matched
% half-angle layer d4 with one change: memory magnitude bounded (c <- c/max(1,|c|)) and fed-back phase damped by |c|^2/(|c|^2+0.05^2).
layers = swapPhaseLayer(model, mfilename('fullpath'), ...
    @(H, varargin) phaseAwareLSTMD4X(H, varargin{:}, 'MemNorm', true));
end
