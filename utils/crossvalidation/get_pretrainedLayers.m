function model = get_pretrainedLayers(model, config, i)
%GET_PRETRAINEDLAYERS
% Loads a pretrained source-domain classifier for a given fold and dataset.
% Stores the loaded network into model.pretrainedNet.
%
% Inputs:
%   model   - Model struct to update
%   config  - Configuration struct with dataset/domain/net type
%   i       - Fold index (used to select correct source model file)
%
% Output:
%   model   - Updated with field: pretrainedNet

%------------------------------------------------------------
% Determine file location and naming based on dataset
%------------------------------------------------------------
switch lower(config.datasetType)

    case 'cwl'        
        switch config.layerName
            case 'baseTransfer'                        
                location = eval(sprintf('config.sourceModelLoc_CWL_%d',config.ncats-1));               
                filename = sprintf("sourceDomain2Classifier_N%d_%s_%d.mat", config.ncats,config.typeNet, i);
            otherwise
               location = eval(sprintf('config.sourceModelLoc_CWL_%d',config.ncats-1));
                filename = sprintf("sourceDomainClassifier_N%d_%s_%d.mat", config.ncats, config.typeNet, i);
                fprintf('loaded')
        end
        locDir   = fullfile(config.typeNet);        

    case 'drowsiness'
        location = config.sourceModelLoc_drowsiness;
        locDir   = fullfile('Drowsiness', config.typeNet);  % Fixed spelling from 'Dowsiness'
        filename = sprintf("drowsiness_sourceDomainClassifier_%s_%d.mat", config.typeNet, i);

    case 'mi'
        switch config.layerName
            case 'baseTransfer'                        
                location = eval(sprintf('config.sourceModelLoc_MI_%d',config.ncats-1));
                    filename = sprintf("sourceDomain2Classifier_N%d_%s_%d.mat", config.ncats,config.typeNet, i);
            otherwise
               location = eval(sprintf('config.sourceModelLoc_MI_%d',config.ncats-1));
                    filename = sprintf("sourceDomainClassifier_N%d_%s_%d.mat", config.ncats, config.typeNet, i);
                fprintf('loaded')
        end
        % locDir   = fullfile(config.typeNet); 
        locDir   = fullfile(config.typeNet);        

    % case 'mi'
        % location = config.sourceModelLoc_BCI_IV_2b;
        % locDir   = fullfile('grazB', config.typeNet);
        % filename = sprintf("MI_sourceDomainClassifier_%s_%d.mat", config.typeNet, i);

    otherwise
        error('Unknown datasetType: %s', config.datasetType);
end

%------------------------------------------------------------
% Load the pretrained model file
%------------------------------------------------------------
loadPath = fullfile(location, locDir, filename);
tempLoad = load(loadPath);

% Store network inside model struct
model.pretrainedNet = tempLoad.net;

end
