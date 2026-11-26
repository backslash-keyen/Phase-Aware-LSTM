function layers = get_pLayer_phaseLSTM(model)
%GET_PLAYER_PHASELSTM  (replication variant 'pv_d4lowInit', opt-in; use with 'gradCap')
% Shadows utils/modelstuff/modeldefinitions/get_pLayer_phaseLSTM.m ONLY while
% runReplication(..., 'variant', {'pv_d4lowInit', 'gradCap'}) has this folder first on the path.
% Swaps the layer named 'phaseLSTM' for phaseAwareLSTMD4X: the matched
% half-angle layer d4 with one change: U and Q initialised at 0.5 x the orthogonal init.
layers = swapPhaseLayer(model, mfilename('fullpath'), ...
    @(H, varargin) phaseAwareLSTMD4X(H, varargin{:}, 'InitScale', 0.5));
end
