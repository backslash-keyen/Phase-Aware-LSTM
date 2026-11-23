clearvars; clc; close all;
warning('off');

% Add all subfolders in this project to the MATLAB path
addpath(genpath(pwd));

%============================
% User Options
%============================
isDemo = true;        % true  -> run tiny demo (ncval = 1, small data)
                      % false -> run full experiment (ncval = 10, user data)

%----------------------------
% Environment Configuration
%----------------------------
% Default configuration (auto-detects project root, sets folder structure)
configs = configureExperiment();

% Parallel training: set 'true' if you have Parallel Computing Toolbox
configs.parTrain = false;   % true  = use parfor, false = serial training

%----------------------------
% Experiment Specifications
%----------------------------
% These are the main knobs users may want to change:
configs.typeNet      = 'DeepRNN';     % 'DeepRNN' | 'EEGNet' | 'transformer'
configs.layerName    = 'phase-LSTM';     % which layer to read features from
configs.domain       = 'target-domain'; % 'source-domain' | 'target-domain' | 'transfer'
configs.datasetType  = 'CWL';         % 'CWL' | 'MI' | 'ER'
configs.ncats        = 2;             % 2 or 3 for 'CWL' | 2 for 'MI'

% Cross-validation folds:
% Demo mode: single fold, tiny example data
if isDemo, configs.ncval = 1;end

fprintf('Running Experiment: %s - %s (dataset: %s, ncval=%d)...\n', ...
    configs.typeNet, configs.layerName, configs.datasetType, configs.ncval);

%----------------------------
% Run Experiment
%----------------------------
try
    [trainedModels, evals] = kFold_crossValidation(configs);

    fprintf('Experiment Completed: %s | %s | dataset=%s | ncval=%d\n\n', ...
        configs.typeNet, configs.layerName, configs.datasetType, configs.ncval);

catch ME
    fprintf('Error in Experiment: %s | %s | dataset=%s | ncval=%d\n', ...
        configs.typeNet, configs.layerName, configs.datasetType, configs.ncval);
    fprintf('Error Message: %s\n\n', ME.message);
    rethrow(ME);  % Keep this so power-users get full stack trace
end
