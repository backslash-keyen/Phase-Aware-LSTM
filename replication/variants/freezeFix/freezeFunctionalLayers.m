function layers = freezeFunctionalLayers(layers, modelType)
%FREEZEFUNCTIONALLAYERS  (replication variant 'freezeFix', opt-in)
% Shadows utils/modelstuff/helpers/freezeFunctionalLayers.m ONLY when
% runReplication(..., 'variant', 'freezeFix') puts this folder first on the
% path. The original file is not changed.
%
% Why: in the original, the DeepRNN branch calls
%     layers(1).setLearnRateFactor("RecurrentWeights", 0.1);
% without assigning the result back. Layers are value objects, so nothing
% is frozen and the whole pretrained network is fine-tuned at full rate.
%
% This variant (Keyen, 2026-11-23): for DeepRNN, every recurrent layer of
% the pretrained source network (lstm_1, lstm_2) is frozen with learn-rate
% factor 0 on all its parameters, and the result is assigned back. Every
% other layer trains normally: the new projection layer, batch norm, the
% fully connected classifier. EEGNet and Transformer keep the original
% behaviour, since those branches already assign back.

if ~strcmp(modelType, 'DeepRNN')
    layers = variantCallOriginal('freezeFunctionalLayers', layers, modelType);
    return
end

for i = 1:numel(layers)
    L = layers(i);
    if isa(L, 'nnet.cnn.layer.LSTMLayer') || isa(L, 'nnet.cnn.layer.BiLSTMLayer') || ...
       isa(L, 'nnet.cnn.layer.GRULayer')
        L.InputWeightsLearnRateFactor     = 0;
        L.RecurrentWeightsLearnRateFactor = 0;
        L.BiasLearnRateFactor             = 0;
        layers(i) = L;                    % assigned back (the original does not)
    end
end
end
