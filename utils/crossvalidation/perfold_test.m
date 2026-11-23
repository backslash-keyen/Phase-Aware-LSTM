function evals = perfold_test(model, data)
%PERFOLD_TEST
% Evaluates the trained model on train, validation, and test datasets.
%
% Inputs:
%   model - Struct containing the trained network and timing info
%   data  - Struct with fields dsTrain, dsVal, dsTest (datastores)
%
% Output:
%   evals - Struct containing evaluation metrics for all splits and training time

%------------------------------------------------------------
% Evaluate on the training set
%------------------------------------------------------------
[evals, ~] = perfold_metrics(model, data, 'train', []);

%------------------------------------------------------------
% Evaluate on the validation set and append to evals
%------------------------------------------------------------
evals = perfold_metrics(model, data, 'validation', evals);

%------------------------------------------------------------
% Evaluate on the test set and append to evals
%------------------------------------------------------------
evals = perfold_metrics(model, data, 'test', evals);

%------------------------------------------------------------
% Attach training time to the eval results
%------------------------------------------------------------
evals.trainingTime = model.trainingTime;

end
