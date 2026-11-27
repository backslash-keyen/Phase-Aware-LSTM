function layers = get_pLayer_assocLSTM(model, version)
%GET_PLAYER_ASSOCLSTM  Associative LSTM projection in front of the pretrained source net.
% Same graph as the released get_pLayer_LSTM (the LSTM projection row), with
% lstmLayer(61) replaced by the Associative LSTM (Danihelka et al. 2016):
%   sequenceInputLayer(2) -> associativeLSTMLayer(N = source input size)
%     [-> fullyConnectedLayer(N) readout, for output 'h'] -> source layers 2:end
% N = 61 complex cells, so Nh = 122 real values. For output 'h' the layer
% emits h = [Re h; Im h] (Eq. 30) and a linear readout maps it to the N
% channels the source net expects, as the paper's networks read h through
% learned output weights. 'real' and 'mod' emit N values directly.
% The source layers go through freezeFunctionalLayers, which freezes nothing
% (its result is not assigned), as for the FFNN / LSTM / GRU baselines.
if nargin < 2, version = 'paper'; end
if ~isscalar(model.nchans), nchans = numel(model.nchans); else, nchans = model.nchans; end
assert(~strcmpi(model.typeNet, 'transformer'), 'assoc:transformer', ...
    'The Associative LSTM projection is built for DeepRNN / EEGNet.');
net = model.pretrainedNet;
pretrained = net.Layers;
sourceDimension = pretrained(1).InputSize;
N = sourceDimension(1);
oldLayers = pretrained(2:end);
oldLayers = freezeFunctionalLayers(oldLayers, model.typeNet);
if strcmpi(model.typeNet, 'EEGNet') && contains(class(oldLayers(1)), 'conv', 'IgnoreCase', true)
    oldLayers = [functionLayer(@(X) reshapeOutputForEEGNet(X), 'Name', 'reshape_for_EEGNet', 'Formattable', true); oldLayers];
end
cfg = assocVersions(version);
proj = associativeLSTMLayer(N, 'Name', 'assocLSTM', 'Version', version);
front = [sequenceInputLayer(nchans, 'MinLength', model.signalLength, 'Name', 'input'); proj];
if strcmp(cfg.output, 'h')
    front = [front; fullyConnectedLayer(N, 'Name', 'assocReadout')];
end
layers = layerGraph([front; oldLayers]);
end
