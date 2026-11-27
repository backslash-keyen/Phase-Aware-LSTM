function layers = define_layerArchitecture(model, config)
%DEFINE_LAYERARCHITECTURE  (Phased LSTM variants) adds layer 'phased-lstm'; everything else is the released function.
%
% The released code has no Phased LSTM (Neil, Pfeiffer and Liu 2016). The version
% folder next to phased/common on the path (replication/variants/phased/<name>)
% says which one through phasedSettings():
%   phasedBase   the Nov 2025 layer and builder behind the paper's Phased-LSTM rows,
%                ported unchanged (phasedLSTM, get_pLayer_phasedLSTM)
%   any other    phasedLSTMCfg with that version's settings (get_pLayer_phasedVersion);
%                see phasedVersions.m for the list
if ~strcmpi(config.layerName, 'phased-lstm')
    layers = callOriginal_phased('define_layerArchitecture', model, config);
    return
end
assert(startsWith(lower(config.domain), 'target'), 'phased:sourceRun', ...
    'The Phased LSTM is a projection: it runs only in the target domains.');
s = phasedSettings();
if s.port
    layers = get_pLayer_phasedLSTM(model);          % the replication row, unchanged
else
    layers = get_pLayer_phasedVersion(model, s);
end
fprintf('[%s] Phased LSTM projection built for %s\n', s.name, config.domain);
end
