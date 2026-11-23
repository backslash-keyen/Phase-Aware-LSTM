function dataFold = perfold_dataset(config, i)
%DATASET_LOADPERFOLD
% Loads the fold-specific dataset for the given config and fold index.
% Resolves file paths based on dataset and domain settings.
%
% Inputs:
%   config   - Experiment configuration object
%   i        - Fold number (1-based index)
%
% Output:
%   dataFold - Struct containing loaded EEG data and metadata

%--------------------------------------------
% Step 1: Set base dataset folder
%--------------------------------------------
basePath = fullfile(config.projectLoc, 'datasets','folds');  % Root EEG dataset folder

%--------------------------------------------
% Step 2: Map datasetType to data folder
%--------------------------------------------
%--------------------------------------------
% Resolve dataset folder based on datasetType
% This block is case-insensitive due to `lower(...)`
%--------------------------------------------
switch lower(config.datasetType)
    case 'cwl'
        switch config.ncats 
            case 2                
                dataFolder = fullfile('CWL','NEC21_C2');
            case 3
                dataFolder = fullfile('CWL','NEC21_C3');
        end    
    case 'mi'
        switch config.ncats 
            case 2                
                dataFolder = fullfile('MI','MI_C2');
            case 3
                error('oh no can do! try 2 or 4 categories');
            case 4
                dataFolder = fullfile('MI','MI_C4');
        end    
        
    case 'grazdata_b'
        dataFolder = 'BCI_IV_2b';
    otherwise
        error('Unknown datasetType: %s', config.datasetType);
end


%--------------------------------------------
% Step 3: Map domain to subfolder and filename
%--------------------------------------------
switch lower(config.domain)
    case 'source-domain'
        domainFolder = 'fullchannels';
        filename = sprintf('source_fold%d.mat', i);
    case 'source-domain-2'
        domainFolder = 'dualchannelsource';
        filename = sprintf('source_fold%d.mat', i);

    case 'target-domain'
        switch config.datasetType
            case 'mi'
                domainFolder = 'targetdomain1';
            otherwise
                domainFolder = 'targetdomain1';
        end
        filename = sprintf('target_fold%d.mat', i);
        

    case 'target-domain-2'
        domainFolder = 'targetdomain2';
        filename = sprintf('target_fold%d.mat', i);

    case 'temporalmasking-1'
        domainFolder = 'temporalMasking_Fp1_Fz';
        filename = sprintf('target_fold%d.mat', i);

    case 'temporalmasking-2'
        domainFolder = 'temporalMasking_F3_F7';
        filename = sprintf('target_fold%d.mat', i);

    otherwise
        error('Unknown domain: %s', config.domain);
end

%--------------------------------------------
% Step 4: Load data and convert to datastore format
%--------------------------------------------
fullPath = fullfile(basePath, dataFolder, domainFolder, filename);
dataFold = load(fullPath);
dataFold = makeDatastore(dataFold);  % Converts struct to consistent format

end



% domainMap = struct( ...
%     'source-domain',        {'FullChannelSet', 'source_fold%d.mat'}, ...
%     'target-domain',        {'DualChannels_Fp1_Fz', 'target_fold%d.mat'}, ...
%     'target-domain-2',      {'DualChannels_F3_F7', 'target_fold%d.mat'}, ...
%     'temporalmasking-1',    {'temporalMasking_Fp1_Fz', 'target_fold%d.mat'}, ...
%     'temporalmasking-2',    {'temporalMasking_F3_F7', 'target_fold%d.mat'} ...
% );
% 
% domainKey = lower(config.domain);
% if isfield(domainMap, domainKey)
%     domainFolder = domainMap.(domainKey){1};
%     filename     = sprintf(domainMap.(domainKey){2}, i);
% else
%     error('Unknown domain: %s', config.domain);
% end
