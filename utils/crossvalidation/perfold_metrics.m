function [modelMetrics, model] = perfold_metrics(model, dataFeed, dataset, evaluation)
%PERFOLD_METRICS
% Evaluates the trained model on the specified dataset split (train/val/test).
%
% Inputs:
%   model      - Struct containing trained network and metadata
%   dataFeed   - Struct with datastores: dsTrain, dsVal, dsTest
%   dataset    - String specifying dataset to evaluate ('train', 'validation', or 'test')
%   evaluation - Struct to accumulate metrics across folds (optional)
%
% Outputs:
%   modelMetrics - Updated evaluation struct with accuracy and confusion matrix
%   model        - Unchanged model struct (for compatibility)

%------------------------------------------------------------
% Initialize or update the evaluation struct
%------------------------------------------------------------
modelMetrics = evaluation;  % If empty, starts fresh; else appends

%------------------------------------------------------------
% Choose the appropriate dataset and evaluate
%------------------------------------------------------------
switch dataset
    case 'train'
        dsData = dataFeed.dsTrain;
        [modelMetrics.accuracyTrain, modelMetrics.confusedMatrix_train] = ...
            perfold_evaluate(model, dsData, modelMetrics);

    case 'validation'
        dsData = dataFeed.dsVal;
        [modelMetrics.accuracyVal, modelMetrics.confusedMatrix_val] = ...
            perfold_evaluate(model, dsData, modelMetrics);

    case 'test'
        dsData = dataFeed.dsTest;
        [modelMetrics.accuracyTest, modelMetrics.confusedMatrix_test] = ...
            perfold_evaluate(model, dsData, modelMetrics);

    otherwise
        error('Unknown dataset type: %s. Choose train, validation, or test.', dataset);
end

end
