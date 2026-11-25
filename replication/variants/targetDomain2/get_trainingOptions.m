function hyperParameters = get_trainingOptions(config)
%GET_TRAININGOPTIONS  (replication variant 'targetDomain2', opt-in)
% Shadows utils/modelstuff/trainingoptions/get_trainingOptions.m ONLY when
% runReplication(..., 'variant', 'targetDomain2') puts this folder first on
% the path. The original file is not changed.
%
% The original accepts only 'source-domain' and 'target-domain', although
% perfold_dataset.m and retrievePretrainedNet.m both handle
% 'target-domain-2' (the F3-F7 layout, data in targetdomain2/). This variant
% treats 'target-domain-2' exactly like 'target-domain' and then calls the
% ORIGINAL function, so the options are identical to the Fp1-Fz runs.
cfg = config;
if strcmpi(config.domain, 'target-domain-2')
    cfg.domain = 'target-domain';
end
hyperParameters = callOriginal_targetDomain2('get_trainingOptions', cfg);
end
