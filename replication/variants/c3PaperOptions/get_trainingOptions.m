function hyperParameters = get_trainingOptions(config)
%GET_TRAININGOPTIONS  (replication variant 'c3PaperOptions', opt-in)
% Shadows utils/modelstuff/trainingoptions/get_trainingOptions.m only when
% runReplication(..., 'variant', 'c3PaperOptions') puts this folder first on
% the path. The original file is not changed.
%
% The released get_trainingOptions.m has no 3-class branch: CWL DeepRNN
% 3-class runs get the 2-class file get_trainingOptions_DeepRNN_CWL.m
% (LR 0.0029, mini-batch 197) with epochs capped at 30 (source) / 15 (target).
% The Nov 2025 pipeline behind the paper (L:\plstm\helperLibrary\modelframe\
% get_trainingOptions.m) routes CWL DeepRNN ncats == 3 to the shipped
% get_trainingOptions_CWL_C3_DeepRNN.m (LR 0.004687, mini-batch 234, beta1 0.9 as stored in the HPC runs,
% 68 epochs, best-validation) and sets the epochs per domain:
%   source-domain, source-domain-2   68 (the file's value, no override)
%   target-domain                    50
%   target-domain-2                  15
% then the common settings (Plots none, Shuffle never, GPU if available).
% This variant does exactly that for CWL DeepRNN 3-class runs with any
% non-phase layer. Every other experiment goes to the next
% get_trainingOptions on the path unchanged (e.g. dualSourceTransfer's).
if ~(strcmpi(config.datasetType, 'cwl') && strcmpi(config.typeNet, 'deeprnn') && ...
     ~isempty(config.ncats) && config.ncats == 3 && ...
     ~any(strcmpi(config.layerName, {'phase-lstm','phase-lstm-notrig','phase-lstm-ver5'})))
    hyperParameters = callOriginal_c3PaperOptions('get_trainingOptions', config);
    return
end
hyperParameters = get_trainingOptions_CWL_C3_DeepRNN();
% The options stored in the HPC 3-class result files (source and targets,
% 5 Nov 2025) have GradientDecayFactor 0.9, not the file's 0.8853; every other
% value matches the file. Use what the HPC runs actually trained with.
hyperParameters.GradientDecayFactor = 0.9;
switch lower(config.domain)
    case 'target-domain'
        hyperParameters.MaxEpochs = 50;
    case 'target-domain-2'
        hyperParameters.MaxEpochs = 15;
end
hyperParameters.Plots   = 'none';
hyperParameters.Shuffle = 'never';
if canUseGPU
    hyperParameters.ExecutionEnvironment = 'gpu';
else
    hyperParameters.ExecutionEnvironment = 'cpu';
end
fprintf('[c3PaperOptions] %s %s: get_trainingOptions_CWL_C3_DeepRNN, MaxEpochs %d\n', ...
    config.domain, config.layerName, hyperParameters.MaxEpochs);
end
