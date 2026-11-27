function p = charmSourceFile(config, i)
%CHARMSOURCEFILE  Where the charmSourceTrained variant keeps its CHARM + source-net models.
% pretrainedModels_charm/<ds>/<ds><ncats-1>/<typeNet>/sourceDomainCHARMClassifier_N<ncats>_<typeNet>_<i>.mat
% A separate root, so the plain source nets in pretrainedModels/ are never touched
% (pretrainedModels/ may be a junction to another set). Mirrored by
% replication/private/pretrainedFile.m.
loc = '';
if (isobject(config) && isprop(config, 'projectLoc')) || (isstruct(config) && isfield(config, 'projectLoc'))
    loc = char(config.projectLoc);
end
ds = lower(config.datasetType);
p = fullfile(loc, 'pretrainedModels_charm', ds, sprintf('%s%d', ds, config.ncats - 1), config.typeNet, ...
    sprintf('sourceDomainCHARMClassifier_N%d_%s_%d.mat', config.ncats, config.typeNet, i));
end
