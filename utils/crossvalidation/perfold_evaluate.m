function [accuracy, confusedMatrix] = perfold_evaluate(model, dsData, modelMetrics)
%PERFOLD_EVALUATE
% Figureless evaluation with confusionmat. Robust in scripts/parallel.

arguments
    model
    dsData
    modelMetrics = []; %#ok<INUSD>
end

% --- Reset and split the combined datastore ---
reset(dsData);
xDS = dsData.UnderlyingDatastores{1};
yDS = dsData.UnderlyingDatastores{2};

% --- Read labels once (as categorical) ---
origOutType = yDS.OutputType;
yDS.OutputType = 'same';
yTrue = yDS.readall;
yDS.OutputType = origOutType;
if ~iscategorical(yTrue); yTrue = categorical(yTrue); end

% --- Read features & classify (handle 3D stack fallback) ---
try
    X = xDS.readall;
    [yPred, yScore] = classify(model.trainedNet, X);
catch
    Xcell = xDS.readall;
    if ~iscell(Xcell)
        rethrow(lasterror); %#ok<LERR>
    end
    X3 = cat(3, Xcell{:});
    [yPred, yScore] = classify(model.trainedNet, arrayDatastore(X3, "IterationDimension", 3));
end
if ~iscategorical(yPred); yPred = categorical(yPred); end

% --- Establish a stable class order for the CM ---
% Use yTrue category order; append any unseen predicted categories at the end.
catsTrue = categories(yTrue);
catsPred = categories(yPred);
catsAll  = [catsTrue; setdiff(catsPred, catsTrue, 'stable')];
classLabels = categorical(catsAll, catsAll, 'Protected', true);

% Align yTrue and yPred to this order (unseen => <undefined>, which confusionmat ignores)
yTrue = categorical(yTrue, catsAll);
yPred = categorical(yPred, catsAll);

% --- Build absolute confusion matrix (rows=actual, cols=predicted) ---
CM = confusionmat(yTrue, yPred, 'Order', classLabels);

% --- Scalar accuracy (pure counts) ---
accuracy = trace(CM) / sum(CM(:));

% --- Build the full metrics struct from counts (no charts needed) ---
confusedMatrix = cm_buildMetrics_fromCounts(classLabels, CM, yTrue, yScore);

end
