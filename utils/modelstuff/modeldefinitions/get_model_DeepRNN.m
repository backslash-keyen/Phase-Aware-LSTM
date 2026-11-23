% function layers = get_DeepRNN(model)
% %GET_DEEPRNN
% % Returns a deep LSTM-based classification network.
% % Used as the full model for source-domain training or encoder+projection setup.
% %
% % Input:
% %   model - Struct with fields:
% %           .nchans  : Number of EEG channels (input features)
% %           .nclass  : Number of output classes
% %
% % Output:
% %   layers - Layer array compatible with trainNetwork
% 
% %------------------------------------------------------------
% % Extract input and output sizes
% %------------------------------------------------------------
% nchans = model.nchans;
% 
% % Use model.nclass if available, else default to binary
% if isfield(model, 'nclass')
%     nclass = model.nclass;
% else
%     nclass = 2;
% end
% 
% %------------------------------------------------------------
% % Define layer parameters
% %------------------------------------------------------------
% nhidden1 = 16;      % Units in first LSTM layer
% nhidden2 = 32;       % Units in second LSTM layer
% dropout1 = 0.7;
% dropout2 = 0.7;
% 
% %------------------------------------------------------------
% % Define Deep RNN architecture
% %------------------------------------------------------------
% layers = [
%     sequenceInputLayer(nchans, 'Name', 'input')
% 
%     bilstmLayer(nhidden1, 'OutputMode', 'sequence', 'Name', 'lstm_1')
%     dropoutLayer(dropout1, 'Name', 'dropout_1')
%     batchNormalizationLayer('Name', 'bn_1')
%     eluLayer(0.5, 'Name', 'elu_1')  % Activation after batch norm
% 
%     lstmLayer(nhidden2, 'OutputMode', 'last', 'Name', 'lstm_2')
%     dropoutLayer(dropout2, 'Name', 'dropout_2')
%     batchNormalizationLayer('Name', 'bn_2')
%     eluLayer(0.5, 'Name', 'elu_2')  % Activation after batch norm
% 
%     fullyConnectedLayer(nclass, 'Name', 'fc_class')
%     softmaxLayer('Name', 'softmax')
%     classificationLayer('Name', 'crossentropy')
% ];
% 
% end

function layers = get_model_DeepRNN(model)
%GET_DEEPRNN
% Returns a deep LSTM-based classification network.
% Used as the full model for source-domain training or encoder+projection setup.
%
% Input:
%   model - Struct with fields:
%           .nchans  : Number of EEG channels (input features)
%           .nclass  : Number of output classes
%
% Output:
%   layers - Layer array compatible with trainNetwork

%------------------------------------------------------------
% Extract input and output sizes
%------------------------------------------------------------
nchans = model.nchans;

% Use model.nclass if available, else default to binary
if isfield(model, 'nclass')
    nclass = model.nclass;
else
    nclass = 2;
end

%------------------------------------------------------------
% Define layer parameters
%------------------------------------------------------------
nhidden1 = 128;      % Units in first LSTM layer
nhidden2 = 64;       % Units in second LSTM layer
dropout1 = 0.5;
dropout2 = 0.6;

%------------------------------------------------------------
% Define Deep RNN architecture
%------------------------------------------------------------
layers = [
    sequenceInputLayer(nchans, 'Name', 'input')

    lstmLayer(nhidden1, 'OutputMode', 'sequence', 'Name', 'lstm_1')
    dropoutLayer(dropout1, 'Name', 'dropout_1')
    batchNormalizationLayer('Name', 'bn_1')
    eluLayer(1, 'Name', 'elu_1')  % Activation after batch norm

    lstmLayer(nhidden2, 'OutputMode', 'last', 'Name', 'lstm_2')
    dropoutLayer(dropout2, 'Name', 'dropout_2')
    batchNormalizationLayer('Name', 'bn_2')
    eluLayer(1, 'Name', 'elu_2')  % Activation after batch norm

    fullyConnectedLayer(nclass, 'Name', 'fc_class')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'crossentropy')
];

end
