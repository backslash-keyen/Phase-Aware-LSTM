function p = pretrainedFile(projectLoc, k, fold)
%PRETRAINEDFILE  Source model a target experiment loads for one fold.
% Mirrors utils/crossvalidation/get_pretrainedLayers.m / perfold_saveModel.m.
root = fullfile(projectLoc, 'pretrainedModels', lower(k.datasetType), ...
                sprintf('%s%d', lower(k.datasetType), k.ncats - 1), k.typeNet);
if strcmpi(k.layerName, 'baseTransfer')
    stem = 'sourceDomain2Classifier';
else
    stem = 'sourceDomainClassifier';
end
p = fullfile(root, sprintf('%s_N%d_%s_%d.mat', stem, k.ncats, k.typeNet, fold));
end
