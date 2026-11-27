function layers = define_layerArchitecture(model, config)
%DEFINE_LAYERARCHITECTURE  (CHARM variants) adds layer 'charm'; everything else is the released function.
%
% The released code has no CHARM. The variant folder next to charmCommon on the
% path (charmBase, charm3Conv, charmInstNorm or charmSourceTrained) says which
% CHARM through charmSettings():
%   charmBase           the Nov 2025 builder and layer, unchanged (get_pLayer_charmLayer)
%   charm3Conv          three-layer mconv (charmLayer3Conv), otherwise as charmBase
%   charmInstNorm       instanceNormalizationLayer on the target channels, then charmBase
%   charmSourceTrained  source-domain + charm: CHARM-base trained jointly with the
%                       DeepRNN on the full-montage source folds;
%                       target-domain + charm: that CHARM + DeepRNN with the input
%                       layer swapped for the target channels, all of it fine-tuned
if ~strcmpi(config.layerName, 'charm')
    layers = callOriginal_charm('define_layerArchitecture', model, config);
    return
end
s = charmSettings();
if startsWith(lower(config.domain), 'source')
    assert(s.sourceTrained, 'charm:sourceRun', 'Only charmSourceTrained trains CHARM on the source data.');
    layers = charmSourceNet(model, config);
elseif s.sourceTrained
    layers = charmTransferNet(model);
elseif strcmp(s.layer, 'base') && ~s.inputNorm
    layers = get_pLayer_charmLayer(model);          % the replication row, unchanged
else
    layers = get_pLayer_charmVersion(model, s);
end
fprintf('[%s] CHARM network built for %s\n', s.name, config.domain);
end

function layers = charmSourceNet(model, config)
c = config; c.layerName = config.typeNet;
base = callOriginal_charm('define_layerArchitecture', model, c);   % e.g. get_model_DeepRNN
assert(isa(base(1), 'nnet.cnn.layer.SequenceInputLayer'), 'charm:sourceInput', ...
    'charmSourceTrained supports source nets that start with a sequence input layer (DeepRNN).');
M = base(1).InputSize;                     % canonical channels = source montage size
layers = [base(1); charmLayer(M, 32, 'Name', 'charmLayer'); base(2:end)];
end

function layers = charmTransferNet(model)
L = model.pretrainedNet.Layers;
assert(numel(L) > 2 && isa(L(2), 'charmLayer'), 'charm:noSourceCharm', ...
    'charmSourceTrained needs the CHARM + source net (sourceDomainCHARMClassifier_*), got %s.', class(L(2)));
if ~isscalar(model.nchans), nchans = numel(model.nchans); else, nchans = model.nchans; end
inp  = sequenceInputLayer(nchans, 'MinLength', model.signalLength, 'Name', 'input');
rest = freezeFunctionalLayers(L(3:end), model.typeNet);   % no-op, as for the other baselines
layers = layerGraph([inp; L(2); rest]);
fprintf('[charmSourceTrained] source-trained CHARM (M = %d) now reads %d target channels\n', L(2).NumCanonical, nchans);
end
