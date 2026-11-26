function layers = get_pLayer_phaseLSTM(model)
%GET_PLAYER_PHASELSTM  (replication variant 'pv_ps_angAtan', opt-in: pipeline study cell 'angAtan')
% Shadows utils/modelstuff/modeldefinitions/get_pLayer_phaseLSTM.m ONLY while
% runReplication(..., 'variant', 'pv_ps_angAtan') has this folder first on the path.
% Builds the network with the ORIGINAL function and swaps the layer named
% 'phaseLSTM' for phaseAwareLSTM_study_fast with cell 'angAtan'
% (replication/versions/fast/common/pipelineStudyCells.m).
layers = swapPhaseLayer(model, mfilename('fullpath'), @(H, varargin) phaseAwareLSTM_study_fast(H, 'angAtan', varargin{:}));
end
