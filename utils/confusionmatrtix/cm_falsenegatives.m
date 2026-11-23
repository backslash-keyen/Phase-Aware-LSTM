function falsenegative = cm_falsenegatives(classLabels, classMetrics)
%CM_FALSENEGATIVES False negatives.
%   Binary (numel(classLabels)==2):
%       Returns scalar FN = true class positive, predicted negative.
%   Multi-class (numel(classLabels)>=3):
%       Returns FN(i) = instances actually class i but predicted as others.
%
%   Assumes classMetrics is a square confusion matrix with rows = actual,
%   columns = predicted, ordered as classLabels.

    nClasses = numel(classLabels);

    if ~isnumeric(classMetrics) || size(classMetrics,1) ~= size(classMetrics,2)
        error('classMetrics must be a square confusion matrix.');
    end
    if size(classMetrics,1) ~= nClasses
        error('classMetrics size must match numel(classLabels).');
    end

    if nClasses == 2
        % --- Preserve existing binary behavior ---
        posMask = (classLabels == classLabels(2)); % Actual
        negMask = (classLabels == classLabels(1)); % Predicted
        falsenegative = classMetrics(posMask, negMask);

    elseif nClasses >= 3
        % --- Multi-class ---
        CM = classMetrics;
        TP = diag(CM);
        FN = sum(CM,2) - TP;     % Actually i but predicted as others
        falsenegative = FN;
    else
        error('At least two classes are required.');
    end
end
