function falsepositive = cm_falsepositives(classLabels, classMetrics)
%CM_FALSEPOSITIVES False positives.
%   Binary (numel(classLabels)==2):
%       Returns scalar FP = true class negative, predicted positive.
%   Multi-class (numel(classLabels)>=3):
%       Returns FP(i) = instances predicted as class i but actually others.
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
        negMask = (classLabels == classLabels(1)); % Actual
        posMask = (classLabels == classLabels(2)); % Predicted
        falsepositive = classMetrics(negMask, posMask);

    elseif nClasses >= 3
        % --- Multi-class ---
        CM = classMetrics;
        TP = diag(CM);
        FP = sum(CM,1).' - TP;   % Predicted as i but actually others
        falsepositive = FP;
    else
        error('At least two classes are required.');
    end
end
