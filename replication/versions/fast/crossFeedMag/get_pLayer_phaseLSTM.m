function layers = get_pLayer_phaseLSTM(model)
%GET_PLAYER_PHASELSTM  (replication variant 'pv_crossFeedMag', opt-in)
% Shadows utils/modelstuff/modeldefinitions/get_pLayer_phaseLSTM.m ONLY while
% runReplication(..., 'variant', 'pv_crossFeedMag') has this folder first on the path.
% Builds the network with the ORIGINAL function and swaps the layer named
% 'phaseLSTM' for phaseAwareLSTMCrossFeed (CrossMode "magOnly"): the matched
% half-angle layer d4 whose magnitude gates also see [cos phi; sin phi]; the
% phase gates are as in d4 (one-way cross-feed).
layers = swapPhaseLayer(model, mfilename('fullpath'), ...
    @(H, varargin) phaseAwareLSTMCrossFeed(H, varargin{:}, 'CrossMode', "magOnly"));
end
