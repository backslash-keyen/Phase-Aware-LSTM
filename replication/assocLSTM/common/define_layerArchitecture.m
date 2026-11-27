function layers = define_layerArchitecture(model, config)
%DEFINE_LAYERARCHITECTURE  (al_<version> variants) adds the projection 'assoc-lstm'.
% The released code has no Associative LSTM. The version folder next to this
% one on the path (replication/assocLSTM/versions/<id>) names the version
% through assocVersionId(); assocVersions.m holds its configuration.
% Everything else goes to the released function.
if ~strcmpi(config.layerName, 'assoc-lstm')
    layers = callOriginal_assoc('define_layerArchitecture', model, config);
    return
end
assert(~startsWith(lower(config.domain), 'source'), 'assoc:sourceRun', ...
    'The Associative LSTM is a target-domain projection; it has no source run.');
id = assocVersionId();
layers = get_pLayer_assocLSTM(model, id);
fprintf('[assoc-lstm] version %s built for %s\n', id, config.domain);
end
