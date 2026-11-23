function lgraph = get_TransformerSource(model)
%GET_TRANSFORMERSOURCE  Minimal Transformer encoder for EEG classification.
% One encoder block (pre-norm), bidirectional self-attn (no causal mask).
% Pools by selecting the last time step via indexing1dLayer (no GAP).

numChannels = model.nchans;       % EEG channels
maxPosition = model.signalLength; % time steps per trial
nClass      = model.nclass;

% --- Hyperparameters (EEG-friendly) ---
dModel   = 64;                    % constant model dim inside Transformer
numHeads = 4;                      % must divide dModel
ffnDim   = 2*dModel;               % inner FFN dim
dropP    = 0.2;

% ---- Stem: project channels -> d_model and add positional encoding ----
layers = [
    sequenceInputLayer(numChannels, "Name","input")

    % Project channel features into model space
    fullyConnectedLayer(dModel, "Name","in_proj")

    % Positional embeddings must be d_model-sized
    positionEmbeddingLayer(dModel, maxPosition, "Name","pos_emb")
    additionLayer(2,"Name","add_pos")  % X = in_proj + pos_emb

    % ===================== Encoder Block (single) =====================

    % Pre-norm + MHA + dropout + residual
    layerNormalizationLayer("Name","b1_ln1")
    selfAttentionLayer(numHeads, dModel/numHeads, "Name","b1_mha") % no mask
    dropoutLayer(dropP,"Name","b1_drop1")
    additionLayer(2,"Name","b1_res1")        % Y = X + MHA(LN(X))

    % Pre-norm + FFN + dropout + residual
    layerNormalizationLayer("Name","b1_ln2")
    fullyConnectedLayer(ffnDim,"Name","b1_ff1")
    reluLayer("Name","b1_relu1")
    dropoutLayer(dropP,"Name","b1_drop2")
    fullyConnectedLayer(dModel,"Name","b1_ff2")
    dropoutLayer(dropP,"Name","b1_drop3")
    additionLayer(2,"Name","b1_res2")        % Z = Y + FFN(LN(Y))

    % ---- Index last time step and classify ----
    indexing1dLayer("last", "Name", "selectLast")
    % dropoutLayer(dropP,"Name","head_drop")
    fullyConnectedLayer(nClass,"Name","fc_out")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","output")
];

lgraph = layerGraph(layers);

% ---- Wiring (pos add, residuals in the block) ----
% stem
lgraph = connectLayers(lgraph,"in_proj","add_pos/in2");

% Block residual 1: add_pos (X) + b1_drop1 (MHA(LN(X)))
lgraph = connectLayers(lgraph,"add_pos","b1_res1/in2");

% Block residual 2: b1_res1 (Y) + b1_drop3 (FFN(LN(Y)))
lgraph = connectLayers(lgraph,"b1_res1","b1_res2/in2");

end
