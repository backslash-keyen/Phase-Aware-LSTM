function model = retrievePretrainedNet(model, config, i)
%RETRIEVEPRETRAINEDNET
% Initializes or loads pretrained network layers depending on the domain.
%
% For source domain: Defines a fresh model.
% For target domains: Loads pretrained encoder and appends a domain projection head.
%
% Inputs:
%   model  - Struct with network metadata (e.g., typeNet, nchans, nclass)
%   config - Struct with experiment settings (including domain name)
%   i      - Fold index (used to load fold-specific source model)
%
% Output:
%   model  - Updated model struct with .layers populated

% ------------------------------------------------------------
% Case 1: SOURCE DOMAIN → Build full model from scratch
% ------------------------------------------------------------
switch lower(config.domain)

    case 'source-domain'
        % Create the full model architecture (e.g., DeepRNN, EEGNet, etc.)
        model.layers = define_layerArchitecture(model, config);
    case 'source-domain-2'
        % Load pretrained source model into model.pretrainedNet
        % model = get_pretrainedLayers(model, config, i);

        % Append domain-specific projection layers (e.g., LSTM, GRU, FFNN)
        model.layers = define_layerArchitecture(model, config);
% ------------------------------------------------------------
% Case 2: TARGET DOMAINS → Load source + append projection layer
% ------------------------------------------------------------
    case {'target-domain', 'target-domain-2', ...
          'temporalmasking-1', 'temporalmasking-2'}

        % Load pretrained source model into model.pretrainedNet
        model = get_pretrainedLayers(model, config, i);

        % Append domain-specific projection layers (e.g., LSTM, GRU, FFNN)
        model.layers = define_layerArchitecture(model, config);

% ------------------------------------------------------------
% Unknown case → Raise error
% ------------------------------------------------------------
    otherwise
        error('Unknown domain type: %s', config.domain);
end

% ------------------------------------------------------------
% Optional verbose mode for target domains
% ------------------------------------------------------------
if isfield(config, 'verbose') && config.verbose && ~strcmpi(config.domain, 'source-domain')
    fprintf('Loaded pretrained encoder for fold %d — domain: %s | dataset: %s | model: %s\n', ...
        i, config.domain, config.datasetType, config.typeNet);
end

end
