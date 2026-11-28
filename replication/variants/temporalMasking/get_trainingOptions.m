function hyperParameters = get_trainingOptions(config)
%GET_TRAININGOPTIONS  (replication variant 'temporalMasking', opt-in)
% Shadows utils/modelstuff/trainingoptions/get_trainingOptions.m ONLY when
% runReplication(..., 'variant', 'temporalMasking') puts this folder first on
% the path. The original file is not changed.
%
% perfold_dataset.m and retrievePretrainedNet.m already handle the masked
% target domains 'temporalmasking-1' (Fp1-Fz, data in temporalMasking_Fp1_Fz/)
% and 'temporalmasking-2' (F3-F7, temporalMasking_F3_F7/), but the original
% accepts only 'source-domain' and 'target-domain'. This variant treats both
% masked domains exactly like 'target-domain' and then calls the ORIGINAL
% function, so the options are identical to the unmasked target runs.
cfg = config;
if startsWith(lower(config.domain), 'temporalmasking')
    cfg.domain = 'target-domain';
end
hyperParameters = callOriginal_temporalMasking('get_trainingOptions', cfg);
end
