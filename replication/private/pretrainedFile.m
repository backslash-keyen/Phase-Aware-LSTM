function p = pretrainedFile(projectLoc, k, fold, sourceModelRoot, v)
%PRETRAINEDFILE  Source model a target experiment loads for one fold.
% Mirrors utils/crossvalidation/get_pretrainedLayers.m / perfold_saveModel.m
% (and replication/variants/charmCommon/charmSourceFile.m for charmSourceTrained).
if nargin >= 5 && strcmpi(k.layerName, 'charm') && isfield(v, 'names') && any(strcmp(v.names, 'charmSourceTrained'))
    ds = lower(k.datasetType);
    p = fullfile(projectLoc, 'pretrainedModels_charm', ds, sprintf('%s%d', ds, k.ncats - 1), k.typeNet, ...
        sprintf('sourceDomainCHARMClassifier_N%d_%s_%d.mat', k.ncats, k.typeNet, fold));
    return
end
if nargin < 4 || isempty(sourceModelRoot)
    sourceModelRoot = fullfile(projectLoc, 'pretrainedModels');
end
root = fullfile(sourceModelRoot, lower(k.datasetType), ...
                sprintf('%s%d', lower(k.datasetType), k.ncats - 1), k.typeNet);
if strcmpi(k.layerName, 'baseTransfer')
    stem = 'sourceDomain2Classifier';
else
    stem = 'sourceDomainClassifier';
end
p = fullfile(root, sprintf('%s_N%d_%s_%d.mat', stem, k.ncats, k.typeNet, fold));
end
