function [ok, reason] = supportCheck(k, v)
%SUPPORTCHECK  Can the released code (plus the active variant, if any) run experiment k?
if nargin < 2, v = struct('name', '', 'extraDomains', {{}}); end
ok = true; reason = '';
if isfield(v, 'sourceModelRoot') && ~isempty(v.sourceModelRoot) && startsWith(lower(k.domain), 'source')
    ok = false;
    reason = 'authorSource is read-only: source-domain runs (which save models) are not run with it';
    return
end
if strcmpi(k.datasetType, 'MI') && any(strcmpi(k.typeNet, {'EEGNet','transformer'}))
    ok = false;
    reason = sprintf(['get_trainingOptions.m calls get_trainingOptions_Graz_%s, ' ...
                      'which is not in the repository'], k.typeNet);
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
if strcmpi(k.layerName, 'charm')
    names = {};
    if isfield(v, 'names'), names = v.names; end
    if ~any(ismember({'charmBase','charm3Conv','charmInstNorm','charmSourceTrained'}, names))
        ok = false;
        reason = ['the CHARM projection is not in the released code; run with ''variant'', ' ...
                  '''charmBase'' (the Nov 2025 layer, ported unchanged) or charm3Conv / charmInstNorm / charmSourceTrained'];
        return
    end
    if startsWith(lower(k.domain), 'source') && ~any(strcmp(names, 'charmSourceTrained'))
        ok = false;
        reason = 'a source run with CHARM exists only for charmSourceTrained';
        return
    end
end
if strcmpi(k.layerName, 'baseTransfer') && ~(isfield(v, 'names') && any(strcmp(v.names, 'dualSourceTransfer')))
    ok = false;
    reason = ['define_layerArchitecture.m has no builder for layer "baseTransfer" ' ...
              '(common-channel selection), which get_pretrainedLayers.m loads'];
end
end
