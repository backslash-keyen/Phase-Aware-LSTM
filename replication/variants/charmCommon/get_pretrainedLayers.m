function model = get_pretrainedLayers(model, config, i)
%GET_PRETRAINEDLAYERS  (CHARM variants) charmSourceTrained target runs load the CHARM + source net.
% Everything else (and every other CHARM variant) loads the plain source net
% through the released function.
s = charmSettings();
if strcmpi(config.layerName, 'charm') && s.sourceTrained && ~startsWith(lower(config.domain), 'source')
    f = charmSourceFile(config, i);
    tmp = load(f, 'net');
    model.pretrainedNet = tmp.net;
    fprintf('[charmSourceTrained] loaded %s\n', f);
else
    model = callOriginal_charm('get_pretrainedLayers', model, config, i);
end
end
