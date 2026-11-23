function layers = freezeFunctionalLayers(layers, modelType)
%FREEZEFUNCTIONALLAYERS  (replication variant 'freezeIntended', opt-in)
% Shadows utils/modelstuff/helpers/freezeFunctionalLayers.m ONLY when
% runReplication(..., 'variant', 'freezeIntended') puts this folder first
% on the path. The original file is not changed.
%
% The original DeepRNN branch calls
%     layers(1).setLearnRateFactor("RecurrentWeights", 0.1);
% without assigning the result back. Layers are value objects, so the
% factors are never applied and the source network trains at full rate.
%
% This variant applies exactly the factors the original intends, and
% assigns them back (paper: "reduced learning rates applied only to
% essential functional layers"):
%   DeepRNN : layers(1) (lstm_1) = 0.1, layers(5) (lstm_2) = 0.2 on
%             RecurrentWeights, InputWeights and Bias; all other layers
%             (batch norm, fc_class, the projection) at full rate, as in
%             the original.
%   EEGNet / Transformer: the original function, unchanged (those
%             branches already assign back).

if ~strcmp(modelType, 'DeepRNN')
    layers = variantCallOriginal('freezeFunctionalLayers', layers, modelType);
    return
end

layers(1) = setFactors(layers(1), 0.1);
layers(5) = setFactors(layers(5), 0.2);
end

function L = setFactors(L, f)
L = setLearnRateFactor(L, "RecurrentWeights", f);
L = setLearnRateFactor(L, "InputWeights", f);
L = setLearnRateFactor(L, "Bias", f);
end
