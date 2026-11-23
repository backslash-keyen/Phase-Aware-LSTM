function layers = freezeFunctionalLayers(layers, modelType)
%FREEZEFUNCTIONALLAYERS Adjusts learning rates of key functional layers
% to preserve pretrained knowledge while allowing slow adaptation

switch modelType
    case 'DeepRNN'
        % Freeze initial and final LSTM layers (soft freeze)
        layers(1).setLearnRateFactor("RecurrentWeights", 0.1);
        layers(1).setLearnRateFactor("InputWeights", 0.1);
        layers(1).setLearnRateFactor("Bias", 0.1);

        layers(5).setLearnRateFactor("RecurrentWeights", 0.2);
        layers(5).setLearnRateFactor("InputWeights", 0.2);
        layers(5).setLearnRateFactor("Bias", 0.2);

    case 'EEGNet'
        % Apply small learning rate to all layers with trainable params
        for i = 1:numel(layers)
            layer = layers(i);
            if isprop(layer, 'WeightLearnRateFactor')
                layer.WeightLearnRateFactor = 0.1;
            end
            if isprop(layer, 'BiasLearnRateFactor')
                layer.BiasLearnRateFactor = 0.1;
            end
            layers(i) = layer;
        end

    case 'transformer'
         % Set low LR (e.g., 0.1) for functional transformer layers
  lowLRLayers = [
    "attn1", "fc1", "fc2", ...
    "norm1", "norm2", ...
    "fc_out"
];
for i = 1:numel(layers)
    layer = layers(i);
    
    % Check for learnable weights
    if isprop(layer, 'WeightLearnRateFactor')
        layer.WeightLearnRateFactor = 0.1;
    end
    if isprop(layer, 'BiasLearnRateFactor')
        layer.BiasLearnRateFactor = 0.1;
    end

    % Replace the layer back into graph
    layers(i) = layer;
end

    otherwise
        % Leave others unchanged
end

end
