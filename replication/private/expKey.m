function k = expKey(datasetType, ncats, typeNet, domain, layerName)
%EXPKEY  One experiment = one kFold_crossValidation call with this config.
k.kind        = 'exp';
k.id          = sprintf('%s_N%d_%s_%s_%s', lower(datasetType), ncats, domain, typeNet, lower(layerName));
k.reason      = '';
k.datasetType = upper(datasetType);
k.ncats       = ncats;
k.typeNet     = typeNet;
k.domain      = domain;
k.layerName   = layerName;
end
