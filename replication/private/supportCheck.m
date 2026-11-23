function [ok, reason] = supportCheck(k, useShims)
%SUPPORTCHECK  Can the released code run experiment k?
ok = true; reason = '';
if strcmpi(k.datasetType, 'MI') && any(strcmpi(k.typeNet, {'EEGNet','transformer'}))
    ok = false;
    reason = sprintf(['get_trainingOptions.m calls get_trainingOptions_Graz_%s, ' ...
                      'which is not in the repository'], k.typeNet);
    return
end
needsShim = ~any(strcmpi(k.domain, {'source-domain','target-domain'})) || ...
            strcmpi(k.layerName, 'baseTransfer');
if needsShim && ~useShims
    ok = false;
    reason = sprintf(['get_trainingOptions.m / define_layerArchitecture.m reject domain "%s"' ...
                      ' / layer "%s"; rerun with ''useShims'', true'], k.domain, k.layerName);
end
end
