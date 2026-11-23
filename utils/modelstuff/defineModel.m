function model = defineModel(data, config, i)
%DEFINEMODEL
% Initializes a model struct with architecture and training metadata.
% Computes basic input properties and prepares training options.
%
% Inputs:
%   data    - Struct containing xtrain, ytrain, and dsVal (datastore)
%   config  - Experiment configuration
%   i       - Fold index (used if loading pretrained models)
%
% Output:
%   model   - Struct with metadata, training options, and pretrained network (if applicable)

%--------------------------------------------
% Initialize model struct with architecture name
%--------------------------------------------
model = struct();
model.typeNet = config.typeNet;

%--------------------------------------------
% Extract input shape from training data
%--------------------------------------------
model.nchans       = size(data.xtrain, 1);  % Number of EEG channels
model.signalLength = size(data.xtrain, 2);  % Time length per trial

%--------------------------------------------
% Determine number of classes from training labels
% Uses categorical info (robust to label format)
%--------------------------------------------
model.nclass = numel(categories(data.ytrain));

%--------------------------------------------
% Get training options (e.g., optimizer, learning rate, etc.)
%--------------------------------------------
model.options = get_trainingOptions(config);

% Add validation data explicitly to training options
model.options.ValidationData = data.dsVal;

%--------------------------------------------
% Retrieve pretrained network weights 
%--------------------------------------------
model = retrievePretrainedNet(model, config, i);

end
