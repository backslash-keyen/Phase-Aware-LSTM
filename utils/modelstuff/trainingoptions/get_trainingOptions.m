function hyperParameters = get_trainingOptions(config)
%GET_TRAININGOPTIONS
% Clean public version for CWL + MI datasets.
% Provides stable, predictable training options.

% ------------------------------------------------------------
% Select dataset-specific defaults
% ------------------------------------------------------------
switch lower(config.datasetType)

    % =======================
    % CWL DEMO DATASET
    % =======================
    case 'cwl'
        switch lower(config.typeNet)
            case 'deeprnn'
                hyperParameters = get_trainingOptions_DeepRNN_CWL();

            case 'eegnet'
                hyperParameters = get_trainingOptions_NEC21_EEGNet();

            case 'transformer'
                hyperParameters = get_trainingOptions_NEC21_Transformer();

            otherwise
                error('Unsupported network type: %s', config.typeNet);
        end

    % =======================
    % MI DEMO DATASET
    % =======================
    case 'mi'
        switch lower(config.typeNet)
            case 'deeprnn'
                hyperParameters = get_trainingOptions_Graz_DeepRNN();

            case 'eegnet'
                hyperParameters = get_trainingOptions_Graz_EEGNet();

            case 'transformer'
                hyperParameters = get_trainingOptions_Graz_Transformer();

            otherwise
                error('Unsupported network type: %s', config.typeNet);
        end

    otherwise
        error('Unknown datasetType: %s', config.datasetType);
end

% ------------------------------------------------------------
% Domain-level epoch adjustments
% ------------------------------------------------------------
switch lower(config.domain)
    case 'source-domain'
        hyperParameters.MaxEpochs = min(hyperParameters.MaxEpochs, 30);

    case 'target-domain'
        hyperParameters.MaxEpochs = min(hyperParameters.MaxEpochs, 15);

    otherwise
        error('Unknown domain: %s', config.domain);
end

% ------------------------------------------------------------
% Common settings for all runs
% ------------------------------------------------------------
hyperParameters.Plots     = 'none';
hyperParameters.Shuffle   = 'never';

if canUseGPU
    hyperParameters.ExecutionEnvironment = 'gpu';
else
    hyperParameters.ExecutionEnvironment = 'cpu';
end

end
