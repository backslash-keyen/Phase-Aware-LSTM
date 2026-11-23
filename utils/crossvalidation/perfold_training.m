function [model, evaluation] = perfold_training(config, i)
%PERFOLD_CROSSVALIDATION
% Runs training and evaluation for a single fold `i` of cross-validation.
%
% Inputs:
%   config      - Experiment configuration object
%   i           - Fold index (1 to config.ncval)
%
% Outputs:
%   model       - Trained model for this fold
%   evaluation  - Evaluation results (e.g., metrics, predictions)

%------------------------------------------------------------
% Display fold info
%------------------------------------------------------------
fprintf('\n=========== Training Fold %d ===========\n', i);

%--------------------------------------------
% Step 1: Load fold-specific dataset
%--------------------------------------------
data = perfold_dataset(config, i);

%--------------------------------------------
% Step 2: Define model architecture
%--------------------------------------------
model = defineModel(data, config, i);

%--------------------------------------------
% Step 3: Ensure reproducibility
%--------------------------------------------
ensureReproducibility(i);

%--------------------------------------------
% Step 4: Train the model
%--------------------------------------------
model = perfold_trainANDtime(data, model);

%--------------------------------------------
% Step 5: Evaluate the model
%--------------------------------------------
evaluation = perfold_test(model, data);

%--------------------------------------------
% Step 6: Verbose Reporting (if enabled)
%--------------------------------------------
if isfield(config, 'verbose') && config.verbose
    perfold_printevaluation(i, evaluation);  % Modular reporting
end

%--------------------------------------------
% Step 7: Save model if in source-domain
%--------------------------------------------
switch config.domain
    case {'source-domain','source-domain-2'}
        perfold_saveModel(config, model, i);
end

end
