function [ok, reason] = supportCheck(k, v)
%SUPPORTCHECK  Can the released code (plus the active variant, if any) run experiment k?
if nargin < 2, v = struct('name', '', 'extraDomains', {{}}); end
ok = true; reason = '';
if isfield(v, 'sourceModelRoot') && ~isempty(v.sourceModelRoot) && startsWith(lower(k.domain), 'source')
    ok = false;
    reason = 'authorSource is read-only: source-domain runs (which save models) are not run with it';
    return
end
if strcmpi(k.datasetType, 'MI') && any(strcmpi(k.typeNet, {'EEGNet','transformer'})) && ...
        ~(isfield(v, 'names') && any(strcmp(v.names, 'miHPCOptions')))
    ok = false;
    reason = sprintf(['get_trainingOptions.m calls get_trainingOptions_Graz_%s, ' ...
                      'which is not in the repository; run with ''variant'', ''miHPCOptions''], k.typeNet);
    return
end
if ~any(strcmpi(k.domain, [{'source-domain','target-domain'}, v.extraDomains]))
    ok = false;
    reason = sprintf(['get_trainingOptions.m and define_layerArchitecture.m accept only ' ...
                      'source-domain and target-domain, not "%s"'], k.domain);
    if strcmpi(k.domain, 'target-domain-2')
        reason = [reason '; run with ''variant'', ''targetDomain2'''];
    end
    return
end
if strcmpi(k.layerName, 'baseTransfer') && ~(isfield(v, 'names') && any(strcmp(v.names, 'dualSourceTransfer')))
    ok = false;
    reason = ['define_layerArchitecture.m has no builder for layer "baseTransfer" ' ...
              '(common-channel selection), which get_pretrainedLayers.m loads'];
end
end
