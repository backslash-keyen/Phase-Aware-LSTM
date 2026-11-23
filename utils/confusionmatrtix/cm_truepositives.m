function truepositive = cm_truepositives(classLabels, classMetrics)
%CM_TRUEPOSITIVES Extract true positives from a confusion matrix.
%   Binary (numel(classLabels)==2):
%       Returns scalar TP for the positive class = classLabels(2)
%       (i.e., true=classLabels(2) and predicted=classLabels(2)).
%
%   Multi-class (numel(classLabels)>=3):
%       Returns a column vector of TPs for each class, aligned with
%       classLabels: truepositive(i) = CM(i,i).
%
%   Assumes classMetrics is a square confusion matrix with rows = actual
%   classes and columns = predicted classes, ordered to match classLabels.

    % Basic checks
    nClasses = numel(classLabels);
    if ~ismatrix(classMetrics) || size(classMetrics,1) ~= size(classMetrics,2)
        error('classMetrics must be a square confusion matrix.');
    end
    if size(classMetrics,1) ~= nClasses
        error('Size mismatch: numel(classLabels) must match size(classMetrics,1).');
    end

    if nClasses == 2
        % --- Preserve existing binary behavior ---
        % Logical index for positive class = classLabels(2)
        posMask = (classLabels == classLabels(2));

        % True positive: actual positive, predicted positive
        truepositive = classMetrics(posMask, posMask);

    elseif nClasses >= 3
        % --- Multi-class: TP per class is the diagonal ---
        truepositive = diag(classMetrics);

    else
        error('At least two classes are required.');
    end
end
