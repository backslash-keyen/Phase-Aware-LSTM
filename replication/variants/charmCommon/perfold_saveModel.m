function perfold_saveModel(config, model, i)
%PERFOLD_SAVEMODEL  (CHARM variants) a source run with layer 'charm' saves to pretrainedModels_charm/.
% Its net is CHARM + source net (charmSourceTrained), so it must not overwrite the
% plain source models (sourceDomainClassifier_*). Everything else: released function.
if strcmpi(config.layerName, 'charm') && startsWith(lower(config.domain), 'source')
    net = model.trainedNet; options = model.options; layers = model.layers; %#ok<NASGU>
    f = charmSourceFile(config, i);
    if ~isfolder(fileparts(f)), mkdir(fileparts(f)); end
    save(f, 'net', 'options', 'layers', '-v7.3');
    fprintf('[charmSourceTrained] saved %s\n', f);
else
    callOriginal_charm('perfold_saveModel', config, model, i);
end
end
