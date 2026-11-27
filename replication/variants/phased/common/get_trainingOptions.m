function hyperParameters = get_trainingOptions(config)
%GET_TRAININGOPTIONS  (Phased LSTM variants) the Nov 2025 options routing for layer 'phased-lstm'.
%
% In the Nov 2025 pipeline (L:\plstm helperLibrary/modelframe/get_trainingOptions.m)
% CWL C2 DeepRNN runs with layerName 'phased-lstm' take
% get_trainingOptions_CWL_C3_DeepRNN (Adam, LR 0.004687, mini-batch 234,
% best-validation output), then the target-domain rule MaxEpochs = 15, then the
% general settings Plots 'none', Shuffle 'never' and the GPU/CPU choice. The
% released get_trainingOptions.m has no such case (it would give the FFNN / LSTM /
% GRU options). Every Phased LSTM version uses this routing, so only the layer
% differs between versions. Anything else goes to the released function.
if ~strcmpi(config.layerName, 'phased-lstm') || ~strcmpi(config.datasetType, 'CWL') ...
        || ~strcmpi(config.typeNet, 'DeepRNN') || config.ncats ~= 2
    hyperParameters = callOriginal_phased('get_trainingOptions', config);
    return
end
hyperParameters = get_trainingOptions_CWL_C3_DeepRNN();
if any(strcmpi(config.domain, {'target-domain','target-domain-2'}))
    hyperParameters.MaxEpochs = 15;
end
hyperParameters.Plots   = 'none';
hyperParameters.Shuffle = 'never';
if canUseGPU
    hyperParameters.ExecutionEnvironment = 'gpu';
else
    hyperParameters.ExecutionEnvironment = 'cpu';
end
% test hook for the short checks: PHASED_MAXEPOCHS=<n> overrides the epochs
e = str2double(getenv('PHASED_MAXEPOCHS'));
if ~isnan(e) && e > 0, hyperParameters.MaxEpochs = e; end
end
