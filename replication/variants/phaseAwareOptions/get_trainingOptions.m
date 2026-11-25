function hyperParameters = get_trainingOptions(config)
%GET_TRAININGOPTIONS  (replication variant 'phaseAwareOptions', opt-in)
% Shadows utils/modelstuff/trainingoptions/get_trainingOptions.m ONLY when
% runReplication(..., 'variant', 'phaseAwareOptions') puts this folder first
% on the path. The original file is not changed.
%
% The repository ships get_trainingOptions_CWL_C2_PhaseAware.m (Adam, LR
% 4.01e-4, GradientThreshold 1.25, beta2 0.982, L2 0.00273, batch 161,
% 'best-validation'), but the released get_trainingOptions.m picks options
% by dataset and source network only, so phase-lstm is trained with the
% DeepRNN options (LR 0.0029, GradientThreshold 82.4, beta2 0.4727, ...).
%
% For CWL + DeepRNN + phase-lstm in a target domain this variant uses the
% phase-aware file instead, then applies exactly what the original applies
% after choosing a file: the target-domain epoch cap (15) and the common
% settings (Plots none, Shuffle never, GPU if available). Every other
% experiment gets the original function's options unchanged.
hyperParameters = callOriginal_phaseAwareOptions('get_trainingOptions', config);
if ~(strcmpi(config.datasetType, 'cwl') && strcmpi(config.typeNet, 'deeprnn') && ...
     strcmpi(config.layerName, 'phase-lstm') && startsWith(lower(config.domain), 'target'))
    return
end
if ~isempty(config.ncats) && config.ncats ~= 2
    warning('phaseAwareOptions:ncats', ...
        'Using the CWL 2-class phase-aware options for a %d-class run.', config.ncats);
end
po = get_trainingOptions_CWL_C2_PhaseAware();
po.MaxEpochs = min(po.MaxEpochs, 15);          % target-domain cap in the original
po.Plots     = 'none';                         % common settings in the original
po.Shuffle   = 'never';
po.ExecutionEnvironment = hyperParameters.ExecutionEnvironment;
fprintf('[phaseAwareOptions] phase-lstm uses get_trainingOptions_CWL_C2_PhaseAware (MaxEpochs %d)\n', po.MaxEpochs);
hyperParameters = po;
end
