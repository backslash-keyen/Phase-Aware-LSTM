function accuracy = cm_accuracy(confusedMatrix)
%CM_ACCURACY Computes classification accuracy.
%   Binary (struct input with fields TP,TN,FP,FN):
%       Accuracy = (TP + TN) / (TP + TN + FP + FN)
%   Multi-class (numeric square confusion matrix CM):
%       Accuracy = trace(CM) / sum(CM(:))

    if isstruct(confusedMatrix)
        % --- Preserve existing binary behavior (field names unchanged) ---
        TP = confusedMatrix.truePositive;
        TN = confusedMatrix.trueNegative;
        FP = confusedMatrix.falsePositive;
        FN = confusedMatrix.falseNegetive;  % Preserving original spelling
        accuracy = (TP + TN) / (TP + TN + FP + FN);

    elseif isnumeric(confusedMatrix) && ismatrix(confusedMatrix) ...
            && size(confusedMatrix,1) == size(confusedMatrix,2)
        % --- Multi-class overall accuracy from confusion matrix ---
        CM = confusedMatrix;
        total = sum(CM(:));
        if total == 0
            accuracy = NaN;
        else
            accuracy = trace(CM) / total;
        end
    else
        error(['Input must be either a struct with fields truePositive, ', ...
               'trueNegative, falsePositive, falseNegetive (binary), ', ...
               'or a square numeric confusion matrix (multi-class).']);
    end
end
