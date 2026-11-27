function s = sourceDependency(k, v)
%SOURCEDEPENDENCY  Source experiment whose saved models target key k loads.
% Returns [] for source experiments. With the charmSourceTrained variant (v),
% a CHARM target run loads the CHARM + source net of the source run with layer 'charm'.
if nargin < 2, v = struct('names', {{}}); end
if startsWith(lower(k.domain), 'source')
    s = [];
elseif strcmpi(k.layerName, 'baseTransfer')
    s = expKey(k.datasetType, k.ncats, k.typeNet, 'source-domain-2', k.typeNet);
elseif strcmpi(k.layerName, 'charm') && isfield(v, 'names') && any(strcmp(v.names, 'charmSourceTrained'))
    s = expKey(k.datasetType, k.ncats, k.typeNet, 'source-domain', 'charm');
else
    s = expKey(k.datasetType, k.ncats, k.typeNet, 'source-domain', k.typeNet);
end
end
