function layers = get_pLayer_phaseLSTM(model)
%GET_PLAYER_PHASELSTM  (replication variant 'pv_crossFeed', opt-in)
% Shadows utils/modelstuff/modeldefinitions/get_pLayer_phaseLSTM.m ONLY while
% runReplication(..., 'variant', 'pv_crossFeed') has this folder first on the path.
% Builds the network with the ORIGINAL function and swaps the layer named
% 'phaseLSTM' for phaseAwareLSTMCrossFeed (CrossMode "both"): the matched
% half-angle layer d4 whose magnitude gates also see [cos phi; sin phi] and
% whose phase gates also see h.
layers = swapPhaseLayer(model, mfilename('fullpath'), ...
    @(H, varargin) phaseAwareLSTMCrossFeed(H, varargin{:}, 'CrossMode', "both"));
end
