function layers = define_layerArchitecture(model, config)
%DEFINE_LAYERARCHITECTURE  (replication variant 'targetDomain2', opt-in)
% Shadows utils/modelstuff/define_layerArchitecture.m ONLY when
% runReplication(..., 'variant', 'targetDomain2') puts this folder first on
% the path. The original file is not changed.
%
% Treats 'target-domain-2' (F3-F7) exactly like 'target-domain' and calls
% the ORIGINAL function, so the projection builders (get_pLayer_FFNN,
% get_pLayer_LSTM, get_pLayer_GRU, ...) are the same ones the Fp1-Fz runs
% use. freezeFunctionalLayers.m is not touched: no freezing change.
cfg = config;
if strcmpi(config.domain, 'target-domain-2')
    cfg.domain = 'target-domain';
end
layers = variantCallOriginal('define_layerArchitecture', model, cfg);
end
