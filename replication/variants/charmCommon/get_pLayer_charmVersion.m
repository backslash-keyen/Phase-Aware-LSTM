function layers = get_pLayer_charmVersion(model, s)
%GET_PLAYER_CHARMVERSION  get_pLayer_charmLayer with a different CHARM front end.
% Same graph as the ported builder (sequenceInputLayer -> CHARM(M = source input
% size, d = 32) -> source layers 2:end, freezeFunctionalLayers as released), with
%   s.layer = '3conv'   charmLayer3Conv (three-layer mconv) in place of charmLayer
%   s.inputNorm = true  instanceNormalizationLayer between the input and CHARM:
%                       each channel of each trial normalised over time, with a
%                       learnable per-channel scale and offset (paper Sec. 2.3)
if ~isscalar(model.nchans), nchans = numel(model.nchans); else, nchans = model.nchans; end
assert(~strcmpi(model.typeNet, 'transformer'), 'charm:transformer', ...
    'The CHARM versions are built for DeepRNN / EEGNet; the transformer uses charmBase.');
net = model.pretrainedNet;
pretrained = net.Layers;
sourceDimension = pretrained(1).InputSize;
oldLayers = freezeFunctionalLayers(pretrained(2:end), model.typeNet);
if strcmpi(model.typeNet, 'EEGNet') && contains(class(oldLayers(1)), 'conv', 'IgnoreCase', true)
    oldLayers = [functionLayer(@(X) reshapeOutputForEEGNet(X), 'Name', 'reshape_for_EEGNet', 'Formattable', true); oldLayers];
end
switch s.layer
    case 'base',  charm = charmLayer(sourceDimension(1), 32, 'Name', 'charmLayer');
    case '3conv', charm = charmLayer3Conv(sourceDimension(1), 32, 'NumConv', 3, 'Name', 'charmLayer');
    otherwise, error('charm:layer', 'Unknown CHARM layer %s', s.layer);
end
front = sequenceInputLayer(nchans, 'MinLength', model.signalLength, 'Name', 'input');
if s.inputNorm
    front = [front; instanceNormalizationLayer('Name', 'instnorm')];
end
layers = layerGraph([front; charm; oldLayers]);
end
