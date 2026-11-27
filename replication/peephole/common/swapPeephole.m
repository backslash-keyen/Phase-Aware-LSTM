function layers = swapPeephole(model, callerFile, versionId)
%SWAPPEEPHOLE  Build the LSTM projection with the repository's own get_pLayer_LSTM,
% then replace its lstmLayer (named 'phaseLSTM' in that builder) by
% peepholeLSTMLayer(versionId) with the same number of hidden units and output mode.
% Everything else (input layer, frozen source layers, options) is the LSTM row's.
d = fileparts(callerFile);
rmpath(d);
restore = onCleanup(@() addpath(d, '-begin'));
layers = get_pLayer_LSTM(model);
clear restore
old = layers.Layers(strcmp({layers.Layers.Name}, 'phaseLSTM'));
assert(isscalar(old) && isa(old, 'nnet.cnn.layer.LSTMLayer'), ...
    'swapPeephole: expected one lstmLayer named ''phaseLSTM'' from get_pLayer_LSTM.');
fprintf('[ph] LSTM projection = peepholeLSTMLayer version %s (%d hidden units)\n', versionId, old.NumHiddenUnits);
layers = replaceLayer(layers, 'phaseLSTM', peepholeLSTMLayer(old.NumHiddenUnits, ...
    'Name', old.Name, 'OutputMode', old.OutputMode, 'Version', versionId));
end
