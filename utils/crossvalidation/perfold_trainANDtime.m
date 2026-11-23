function model = perfold_trainANDtime(data, model)
%PERFOLD_TRAINANDTIME
% Trains a model on a single fold and records training time.
%
% Inputs:
%   data  - Struct with training data (data.dsTrain)
%   model - Struct with layer architecture and training options
%
% Output:
%   model - Updated with trained network, training info, and timing


%------------------------------------------------------------
% Start timer
%------------------------------------------------------------
tic;

%------------------------------------------------------------
% Train the model using MATLAB's built-in function
%------------------------------------------------------------
[model.trainedNet, model.info] = trainNetwork(data.dsTrain, model.layers, model.options);

%------------------------------------------------------------
% Record the elapsed training time
%------------------------------------------------------------
model.trainingTime = toc;

end
