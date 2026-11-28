function layers = define_layerArchitecture(model, config)
%DEFINE_LAYERARCHITECTURE  (replication variant 'temporalMasking', opt-in)
% Treats 'temporalmasking-1' / 'temporalmasking-2' exactly like 'target-domain'
% and calls the ORIGINAL utils/modelstuff/define_layerArchitecture.m, so the
% projection (or no-transfer) networks are the ones the unmasked runs build.
cfg = config;
if startsWith(lower(config.domain), 'temporalmasking')
    cfg.domain = 'target-domain';
end
layers = callOriginal_temporalMasking('define_layerArchitecture', model, cfg);
end
