clear



configs = configProjectionTL()
experimentConfig  = projectionExperiment()


experimentConfig.ncategories = 2;
experimentConfig.type = 'cross-validation';
experimentConfig.ncval = 10;
experimentConfig.layerName = 'source-domain'; % Name of the Model in consideration
experimentConfig.domain = 'target-domain'; % domain in consideraton
experimentConfig.sourceChannel = [1:61];
experimentConfig.targetChannel = [3, 4];
% experimentConfig.targetChannel = [1, 2];
experimentConfig.typeNet = 'DeepRNN';
experimentConfig.datasetType = 'nec21'


% [dataFeeders,dataFeed] = dataFeederUnified(configs,experimentConfig); % Set up the datastream
[dataFeeders,dataFeed] = dataFeeder(configs,experimentConfig); % Set up the datastream

% Set masking ratio: fraction of time to mask per trial
mask_ratio = 0.1;

% Output folder
folder = 'E:\Collection\HPC_ProjectionTL_Verification\Datasets\NEC_21\temporalMasking_F3_F7';
% folder = 'E:\Collection\HPC_ProjectionTL_Verification\Datasets\NEC_21\temporalMasking_Fp1_Fz';

for i = 1:length(dataFeed)
    % Apply masking
    xtrain = applyTemporalMaskSameTime(dataFeed(i).xtrain, mask_ratio);
    xval   = applyTemporalMaskSameTime(dataFeed(i).xval, mask_ratio);
    xtest  = applyTemporalMaskSameTime(dataFeed(i).xtest, mask_ratio);

    % Labels
    ytrain = double(dataFeed(i).ytrain);
    yval   = double(dataFeed(i).yval);
    ytest  = double(dataFeed(i).ytest);

    % Save masked data
    save(fullfile(folder, sprintf('target_fold%d', i)), ...
         'xtrain', 'xval', 'xtest', 'ytrain', 'yval', 'ytest');

    % Clear for next fold
    clear xtrain ytrain xval yval xtest ytest
end
