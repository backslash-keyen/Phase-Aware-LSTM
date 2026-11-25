function s = sourceDependency(k)
%SOURCEDEPENDENCY  Source experiment whose saved models target key k loads.
% Returns [] for source experiments.
if startsWith(lower(k.domain), 'source')
    s = [];
elseif strcmpi(k.layerName, 'baseTransfer')
    s = expKey(k.datasetType, k.ncats, k.typeNet, 'source-domain-2', k.typeNet);
else
    s = expKey(k.datasetType, k.ncats, k.typeNet, 'source-domain', k.typeNet);
end
end
