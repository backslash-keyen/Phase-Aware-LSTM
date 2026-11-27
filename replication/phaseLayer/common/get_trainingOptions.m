function hyperParameters = get_trainingOptions(config)
%GET_TRAININGOPTIONS  (every 'pc_<version>' variant, opt-in)
% Shadows utils/modelstuff/trainingoptions/get_trainingOptions.m only while
% runReplication(..., 'variant', 'pc_<version>') has this folder on the path.
% Same routing as the paper's Nov 2025 runs (and the notrigPhaseLSTM
% variant): CWL + DeepRNN + phase-lstm target runs use
% get_trainingOptions_CWL_C2_PhaseAware (Adam, LR 4.01e-4, GradientThreshold
% 1.25 l2norm, beta1 0.781, beta2 0.982, L2 0.00273, batch 161, LR drop 0.37
% every 10 epochs, best-validation output) with MaxEpochs 20 and Shuffle
% never. Every other experiment gets the original options unchanged.
hyperParameters = callOriginalPhaseCorr('get_trainingOptions', config);
if ~(strcmpi(config.datasetType, 'cwl') && strcmpi(config.typeNet, 'deeprnn') && ...
     strcmpi(config.layerName, 'phase-lstm') && startsWith(lower(config.domain), 'target'))
    return
end
po = get_trainingOptions_CWL_C2_PhaseAware();
po.MaxEpochs = 20;
if ~isempty(getenv('PHASECORR_MAXEPOCHS'))      % short checks only
    po.MaxEpochs = str2double(getenv('PHASECORR_MAXEPOCHS'));
end
po.Plots     = 'none';
po.Shuffle   = 'never';
po.ExecutionEnvironment = hyperParameters.ExecutionEnvironment;
fprintf('[pc] phase-lstm uses get_trainingOptions_CWL_C2_PhaseAware (MaxEpochs %d)\n', po.MaxEpochs);
hyperParameters = po;
end
