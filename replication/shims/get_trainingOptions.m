function hyperParameters = get_trainingOptions(config)
%GET_TRAININGOPTIONS (replication shim, opt-in)
% Shadows utils/modelstuff/trainingoptions/get_trainingOptions.m only when
% runReplication(..., 'useShims', true) puts replication/shims on the path.
%
% The original accepts only 'source-domain' and 'target-domain' and errors
% on every other domain that perfold_dataset.m can load. This shim maps
%   source-domain-2                      -> source-domain rules (30-epoch cap)
%   target-domain-2, temporalmasking-1/2 -> target-domain rules (15-epoch cap)
% and then calls the ORIGINAL function unchanged, so the options are
% exactly what the original would give for that domain family.
cfg = config;
switch lower(config.domain)
    case {'source-domain','source-domain-2'}
        cfg.domain = 'source-domain';
    case {'target-domain','target-domain-2','temporalmasking-1','temporalmasking-2'}
        cfg.domain = 'target-domain';
end
hyperParameters = callOriginal('get_trainingOptions', cfg);
end
