function perfold_saveModel(config, model, i)
%PERFOLD_SAVEMODEL Saves the trained model for the current fold.
%
%   Depending on the dataset and domain type, this function saves:
%   - model.trainedNet (the trained network)
%   - model.options (training options)
%   - model.layers (full layer graph)
%
%   Inputs:
%       config - Experiment configuration struct
%       model  - Trained model struct
%       i      - Fold index (used to name the saved file)

    % Extract fields for saving
    net = model.trainedNet;
    options = model.options;
    layers = model.layers;

    % Standardize strings for robustness
    datasetType = lower(config.datasetType);
    domainType = lower(config.domain);

    % Determine dataset-specific saving logic
    switch datasetType
        case 'cwl'
            switch domainType
                case 'source-domain'
                    % Save source-domain model (NEC21)                    
                    location = eval(sprintf('config.sourceModelLoc_CWL_%d',config.ncats-1));
                    filename = sprintf("sourceDomainClassifier_N%d_%s_%d.mat", config.ncats, config.typeNet, i);
                case 'source-domain-2'                    
                    location = eval(sprintf('config.sourceModelLoc_CWL_%d',config.ncats-1));
                    filename = sprintf("sourceDomain2Classifier_N%d_%s_%d.mat", config.ncats,config.typeNet, i);
                otherwise
                    error("Unsupported domain type: %s", config.domain);
            end
            savePath = fullfile(location,config.typeNet)

        case 'er'
            

        case 'mi'
            switch domainType
                case 'source-domain'
                    % Save source-domain model (NEC21)                    
                    location = eval(sprintf('config.sourceModelLoc_MI_%d',config.ncats-1));
                    filename = sprintf("sourceDomainClassifier_N%d_%s_%d.mat", config.ncats, config.typeNet, i);
                case 'source-domain-2'                    
                    location = eval(sprintf('config.sourceModelLoc_MI_%d',config.ncats-1));
                    filename = sprintf("sourceDomain2Classifier_N%d_%s_%d.mat", config.ncats,config.typeNet, i);
                otherwise
                    error("Unsupported domain type: %s", config.domain);
            end
            savePath = fullfile(location,config.typeNet);
        otherwise
            error("Unsupported datasetType: %s", config.datasetType);
    end

    % Create directory if it doesn't exist
    if ~exist(savePath, 'dir')
        mkdir(savePath);
    end

    % Save model to .mat file using -v7.3 (for large networks)
    save(fullfile(savePath, filename), 'net', 'options', 'layers', '-v7.3');
end
