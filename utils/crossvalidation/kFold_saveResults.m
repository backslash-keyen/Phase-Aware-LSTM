function kFold_saveResults(config, trainedModels, evals)
%KFOLD_SAVERESULTS
% Saves the results of k-fold cross-validation to disk.
% Stores the full trained model set and evaluation metrics, and logs summary statistics to a CSV.
%
% Inputs:
%   config         - Configuration struct with experiment metadata
%   trainedModels  - Cell array of trained models per fold
%   evals          - Cell array of evaluation structs per fold

%------------------------------------------------------------
% Step 1: Define output directory (e.g., .../crossValidationModels/NEC21/DeepRNN)
%------------------------------------------------------------
outputDir = fullfile(config.projectLoc, 'cvalresults', config.datasetType, config.typeNet);

% Create the directory if it doesn't exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%------------------------------------------------------------
% Step 2: Generate a timestamp for versioned saving
%------------------------------------------------------------
timestamp = datestr(now, 'yymmdd_HHMM');  % e.g., "250730_1442"

%------------------------------------------------------------
% Step 3: Construct result filename
% e.g., crossvalidation_DeepRNN_target-domain_DeepRNN_EXP01_250730_1442.mat
%------------------------------------------------------------
resultName = sprintf('cval_%s_%s_%d_%s_%s.mat', ...
    config.typeNet, config.domain, config.ncats,config.layerName, timestamp);

filename = fullfile(outputDir, resultName);

%------------------------------------------------------------
% Step 4: Save trained models and evaluations
% Use -v7.3 for large variable support
%------------------------------------------------------------
save(filename, "trainedModels", "evals", '-v7.3');

%------------------------------------------------------------
% Step 5: Generate and save summary metrics (mean accuracy, time, etc.)
%------------------------------------------------------------
[meanTrainAcc, meanValAcc, meanTestAcc, meanTrainTime] = ...
    generateResultSummary(config, evals);

writeSummaryCSV(config, meanTrainAcc, meanValAcc, meanTestAcc, meanTrainTime, timestamp);

end
