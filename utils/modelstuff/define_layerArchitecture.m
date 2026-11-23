function layers = define_layerArchitecture(model, config)
%DEFINE_LAYERARCHITECTURE
% Minimal architecture selector for public release.
% Supports:
%   - Source domain models: DeepRNN, EEGNet
%   - Projection baselines: LSTM, GRU, FFNN
%   - Phase-Aware LSTM (proposed method)

switch lower(config.domain)

    % ============================================================
    % 1) SOURCE DOMAIN MODELS
    % ============================================================
    case 'source-domain'
        switch lower(config.layerName)
            case 'deeprnn'
                layers = get_model_DeepRNN(model);
            case 'eegnet'
                layers = get_EEGNet(model);
            case 'transformer'
                layers = get_TransformerSource(model)
            otherwise
                error('Unsupported source model: %s', config.layerName);
        end

    % ============================================================
    % 2) TRANSFER / TARGET DOMAIN: PROJECTION LAYERS
    % ============================================================
    case 'target-domain'
        switch lower(config.layerName)
            case 'deeprnn'
                layers = get_model_DeepRNN(model);
            case 'eegnet'
                layers = get_EEGNet(model);
            case 'transformer'
                layers = get_TransformerSource(model)
            % Baseline projection layers
            case 'lstm'
                layers = get_pLayer_LSTM(model);
            case 'gru'
                layers = get_pLayer_GRU(model);
            case 'ffnn'
                layers = get_pLayer_FFNN(model);

            % Proposed method
            case 'phase-lstm'
                layers = get_pLayer_phaseLSTM(model);

            otherwise
                error('Unknown projection layer: %s', config.layerName);
        end

    otherwise
        error('Unknown domain: %s', config.domain);
end

end
