function layers = get_pLayer_phasedVersion(model, s)
%GET_PLAYER_PHASEDVERSION  get_pLayer_phasedLSTM with phasedLSTMCfg (version settings s).
% Same graph as the ported builder:
%   sequenceInputLayer(2) -> Phased LSTM (H = source input size, name 'phaseLSTM')
%   -> source layers 2:end through freezeFunctionalLayers (which freezes nothing,
%   as for the FFNN / LSTM / GRU baselines)
% Per-parameter factors, so that "not learned" means what it meant in Nov 2025:
%   tau, s     learn-rate 1, L2 factor s.timingL2. With timingGrad false their
%              gradient is zero, so with L2 1 weight decay alone moves them (as in
%              phasedLSTM.m) and with L2 0 they stay at init
%   r_on       learn-rate 1 if s.ronLearn (L2 0), else 0 (a fixed property in Nov 2025)
%   peepholes  learn-rate 1 and L2 1 (like W and U) if s.peephole, else 0 (unused)
assert(~strcmpi(model.typeNet, 'transformer'), 'phased:transformer', ...
    'The Phased LSTM versions are built for DeepRNN / EEGNet; the transformer uses phasedBase.');
net = model.pretrainedNet;
pretrained = net.Layers;
sourceDimension = pretrained(1).InputSize;
oldLayers = freezeFunctionalLayers(pretrained(2:end), model.typeNet);
if strcmpi(model.typeNet, 'EEGNet') && contains(class(oldLayers(1)), 'conv', 'IgnoreCase', true)
    oldLayers = [functionLayer(@(X) reshapeOutputForEEGNet(X), 'Name', 'reshape_for_EEGNet', 'Formattable', true); oldLayers];
end
p = phasedLSTMCfg(sourceDimension(1), s, 'Name', 'phaseLSTM');
p = setL2Factor(p, 'tau', s.timingL2);
p = setL2Factor(p, 'shi', s.timingL2);
p = setL2Factor(p, 'ron', 0);
p = setLearnRateFactor(p, 'ron', double(s.ronLearn));
for q = {'Pi','Pf','Po'}
    p = setLearnRateFactor(p, q{1}, double(s.peephole));
    p = setL2Factor(p, q{1}, double(s.peephole));
end
projInput = sequenceInputLayer(2, 'MinLength', model.signalLength, 'Name', 'input');
layers = layerGraph([projInput; p; oldLayers]);
end
