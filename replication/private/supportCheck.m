function [ok, reason] = supportCheck(k)
%SUPPORTCHECK  Can the released code, unchanged, run experiment k?
ok = true; reason = '';
if strcmpi(k.datasetType, 'MI') && any(strcmpi(k.typeNet, {'EEGNet','transformer'}))
    ok = false;
    reason = sprintf(['get_trainingOptions.m calls get_trainingOptions_Graz_%s, ' ...
                      'which is not in the repository'], k.typeNet);
    return
end
if ~any(strcmpi(k.domain, {'source-domain','target-domain'}))
    ok = false;
    reason = sprintf(['get_trainingOptions.m and define_layerArchitecture.m accept only ' ...
                      'source-domain and target-domain, not "%s"'], k.domain);
    return
end
if strcmpi(k.layerName, 'baseTransfer')
    ok = false;
    reason = ['define_layerArchitecture.m has no builder for layer "baseTransfer" ' ...
              '(common-channel selection), which get_pretrainedLayers.m loads'];
end
end
