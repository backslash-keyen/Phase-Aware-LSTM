function [trainedModels, evals] = kFold_crossValidation(config)
%KFOLD_CROSSVALIDATION
% Performs k-fold cross-validation for the given configuration.
% Supports optional parallel training using `parTrain` flag.
%
% Inputs:
%   config          - Configuration struct/object containing:
%                     .ncval (number of folds)
%                     .parTrain (boolean flag for parallelization)
%
% Outputs:
%   trainedModels   - Cell array of trained models for each fold
%   evals           - Cell array of evaluation results per fold

%-------------------------------------------------------------
% Initialize output containers
%-------------------------------------------------------------
trainedModels = cell(1, config.ncval);  % To hold trained models per fold
evals         = cell(1, config.ncval);  % To hold evaluation results

%-------------------------------------------------------------
% Choose serial or parallel execution
%-------------------------------------------------------------
switch config.parTrain
    case true
        % Parallel execution using parfor
        if isempty(gcp('nocreate'))
            parpool;  % Start parallel pool if not already running
        end

        parfor i = 1:config.ncval
            [trainedModels{i}, evals{i}] = perfold_training(config, i);
        end

    case false
        % Serial execution
        for i = 1:config.ncval
            [trainedModels{i}, evals{i}] = perfold_training(config, i);
        end
end

%-------------------------------------------------------------
% Save cross-validation results (trained models + eval metrics)
%-------------------------------------------------------------
kFold_saveResults(config, trainedModels, evals);

end
