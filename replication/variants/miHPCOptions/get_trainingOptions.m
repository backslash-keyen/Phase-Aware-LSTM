function hyperParameters = get_trainingOptions(config)
%GET_TRAININGOPTIONS  (replication variant 'miHPCOptions', opt-in)
% Shadows utils/modelstuff/trainingoptions/get_trainingOptions.m ONLY when
% runReplication(..., 'variant', 'miHPCOptions') puts this folder first on
% the path. The original file is not changed.
%
% MI (BCI IV 2a -> 2b) only; every other dataset goes to the original
% unchanged. The original routes MI to get_trainingOptions_Graz_<net>, but
% only the DeepRNN file ships; this folder adds the EEGNet and Transformer
% files from the authors' Nov 2025 tree (L:\plstm\helperLibrary\modelframe\
% trainingoptions, copied verbatim). The original also caps epochs with
% min(file, 30 | 15), which leaves MI DeepRNN at the file's 3 epochs.
%
% Epochs below are the ones stored in the HPC MI result files
% (L:\plstm\HPC\PhaseAwareLSTM\cvalresults\MI, Nov 2025); all other option
% values there equal the Graz_* files exactly:
%   DeepRNN      source 30, target 2 (FFNN/LSTM/GRU/CHARM), phase layers 30
%   EEGNet       source 30, target 15
%   Transformer  no HPC MI record: the Graz file's own 50 for source, 15 target
% ExecutionEnvironment, Shuffle and Plots stay as the original sets them.
hyperParameters = callOriginal_miHPCOptions('get_trainingOptions', config);
if ~strcmpi(config.datasetType, 'mi'), return; end

isSource = startsWith(lower(config.domain), 'source');
try
    isPhase = startsWith(lower(char(config.layerName)), 'phase');
catch
    isPhase = false;
end
switch lower(config.typeNet)
    case 'deeprnn'
        if isSource, ep = 30; elseif isPhase, ep = 30; else, ep = 2; end
    case 'eegnet'
        if isSource, ep = 30; else, ep = 15; end
    case 'transformer'
        if isSource, ep = 50; else, ep = 15; end
    otherwise
        return
end
hyperParameters.MaxEpochs = ep;
end
