function truenegative = cm_truenegatives(classLabels, classMetrics)
%CM_TRUENEGATIVES True negatives.
%   Binary (numel(classLabels)==2):
%       Returns scalar TN for classLabels(1) as before.
%   Multi-class (numel(classLabels)>=3):
%       Returns a column vector TN(i) for each class i, aligned with classLabels.
%
%   Assumes classMetrics is a square confusion matrix with rows = actual,
%   cols = predicted, ordered as classLabels.

    nClasses = numel(classLabels);

    if ~isnumeric(classMetrics) || size(classMetrics,1) ~= size(classMetrics,2)
        error('classMetrics must be a square confusion matrix.');
    end
    if size(classMetrics,1) ~= nClasses
        error('classMetrics size must match numel(classLabels).');
    end

    if nClasses == 2
        % --- Preserve existing binary behavior (TN for classLabels(1)) ---
        negMask = (classLabels == classLabels(1));
        truenegative = classMetrics(negMask, negMask);

    elseif nClasses >= 3
        % --- Multi-class: TN per class ---
        CM  = classMetrics;
        TP  = diag(CM);
        FP  = sum(CM,1).' - TP;   % predicted as i but actually others (column sums minus TP)
        FN  = sum(CM,2)  - TP;    % actually i but predicted others (row sums minus TP)
        TOT = sum(CM(:));
        truenegative = TOT - (TP + FP + FN);
    else
        error('At least two classes are required.');
    end
end
