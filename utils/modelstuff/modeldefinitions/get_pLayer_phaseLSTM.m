function layers = get_pLayer_phaseLSTM(model)
%GET_PROJECTIONLAYER_PHASELSTM
% Constructs a domain projection network using phase-aware LSTM 
% to map sparse EEG input to the latent space of a pretrained source model.
%
% Input:
%   model - Struct containing pretrainedNet, nchans, signalLength, typeNet
%
% Output:
%   layers - LayerGraph with projection and adapted pretrained layers

% ------------------------------------------------------------
% Determine number of input channels
% ------------------------------------------------------------
if ~isscalar(model.nchans)
    nchans = length(model.nchans);
else
    nchans = model.nchans;
end

% ------------------------------------------------------------
% Load and split pretrained network
% ------------------------------------------------------------
net = model.pretrainedNet;
pretrainedNet = net.Layers;
sourceDimension = pretrainedNet(1).InputSize;
oldLayers = pretrainedNet(2:end);  % remove source input layer

% ------------------------------------------------------------
% Freeze functional layers for transfer learning
% ------------------------------------------------------------
oldLayers = freezeFunctionalLayers(oldLayers, model.typeNet);

% ------------------------------------------------------------
% EEGNet-specific reshaping if convolutional front-end
% ------------------------------------------------------------
if strcmpi(model.typeNet, 'EEGNet') && contains(class(oldLayers(1)), 'conv', 'IgnoreCase', true)
    reshapeLayer = functionLayer(@(X) reshapeOutputForEEGNet(X), ...
        'Name', 'reshape_for_EEGNet', 'Formattable', true);
    oldLayers = [reshapeLayer; oldLayers];
end

% ------------------------------------------------------------
% Domain projection layer using Phase-Aware LSTM (e.g., cstm_arch1)
% ------------------------------------------------------------
projInput = sequenceInputLayer(2, ...
    'MinLength', model.signalLength, ...
    'Name', 'input');

projPhaseLSTM = phaseAwareLSTM(sourceDimension(1), ...
    "OutputMode", "sequence", ...
    'Name', 'phaseLSTM');

domainProjectionLayer = [projInput; projPhaseLSTM];

% ------------------------------------------------------------
% Combine into transfer layer graph
% ------------------------------------------------------------
transferringNet = layerGraph([domainProjectionLayer; oldLayers]);

% ------------------------------------------------------------
% Transformer case: connect projection to attention input
% ------------------------------------------------------------
if strcmpi(model.typeNet, 'transformer')
    % 0) Start from the ORIGINAL pretrained graph (preserves residuals)
    lg = layerGraph(model.pretrainedNet);   % <- NOT net.Layers!

    % 1) Add your projection layers to the graph
    projLayers = [
        sequenceInputLayer(nchans, 'MinLength', model.signalLength, 'Name','input')  % keep name 'input'
phaseAwareLSTM(sourceDimension(1), ...
    "OutputMode", "sequence", ...
    'Name', 'phaseLSTM');
    ];
    lg = replaceLayer(lg, 'input', projLayers(1));       % keep external name 'input'
    lg = addLayers(lg, projLayers(2));                   % add FC as a separate layer

    % 2) Find the FIRST consumer of 'input' (e.g., 'in_proj')
    C = lg.Connections;                                  % table Source, Destination
    r = find(strcmpi(C.Source,'input'), 1);
    assert(~isempty(r), 'No consumer of ''input'' found.');
    firstDest = C.Destination{r};                        % e.g., 'in_proj'

    % 3) Re-route: input -> projectionFFNN -> firstDest
    lg = disconnectLayers(lg, 'input', firstDest);
    lg = connectLayers(lg, 'input', 'phaseLSTM');
    lg = connectLayers(lg, 'phaseLSTM', firstDest);

    % 4) Hand back
    transferringNet = lg;
end


layers = transferringNet;

end
