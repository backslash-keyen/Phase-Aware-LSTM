function confusedMatrix = cm_presentReality(confusingMatrix, y, scores)
%CM_PRESENTREALITY
% Extracts key performance metrics from a confusion matrix object *and*
% the provided labels/scores. Internally reconstructs the absolute-count
% confusion matrix to ensure correct metrics.
%
% Inputs:
%   confusingMatrix : confusionchart object (for labels & display only)
%   y               : Ground-truth labels (categorical or convertible)
%   scores          : Prediction scores
%                     - Binary: N×1 scores for positive class (classLabels(2))
%                     - Multi:  N×K scores, columns aligned with classLabels
%
% Output:
%   confusedMatrix  : Struct with raw CM, TP/TN/FP/FN (scalar for binary,
%                     vectors for multi-class), accuracy, sensitivity, F1,
%                     and ROC (when scores provided).
%
% Notes:
%   - Rows = actual, columns = predicted, ordered as classLabels.
%   - Binary behavior preserved, including 'falseNegetive' spelling.

%------------------------------------------------------------
% Step 1: Extract labels & reconstruct ABSOLUTE confusion matrix
%------------------------------------------------------------
[classLabels, classMetrics] = confusionMatrix_extraction(confusingMatrix, y, scores);

confusedMatrix.classLabels  = classLabels;    % (K×1)
confusedMatrix.classMetrics = classMetrics;   % (K×K, counts)

%------------------------------------------------------------
% Step 2: Core counts (scalar for binary; vectors for multi-class)
%------------------------------------------------------------
confusedMatrix.truePositive  = cm_truepositives(classLabels, classMetrics);
confusedMatrix.trueNegative  = cm_truenegatives(classLabels, classMetrics);
confusedMatrix.falseNegetive = cm_falsenegatives(classLabels, classMetrics); % spelling preserved
confusedMatrix.falsePositive = cm_falsepositives(classLabels, classMetrics);

%------------------------------------------------------------
% Step 3: High-level metrics
%   - For accuracy, pass numeric CM (works for K>=2).
%   - For sensitivity: per-class vector by default.
%   - F1: include; matches your earlier request.
%------------------------------------------------------------
confusedMatrix.accuracy     = cm_accuracy(classMetrics);                        % scalar
confusedMatrix.sensitivity  = cm_sensitivity(classMetrics, 'Average','none');   % K×1
confusedMatrix.f1           = cm_f1score(classMetrics, 'Average','none');       % K×1

% Optional macro/micro summaries (handy, but optional)
confusedMatrix.sensitivity_macro = cm_sensitivity(classMetrics,'Average','macro');
confusedMatrix.sensitivity_micro = cm_sensitivity(classMetrics,'Average','micro');
confusedMatrix.f1_macro          = cm_f1score(classMetrics,'Average','macro');
confusedMatrix.f1_micro          = cm_f1score(classMetrics,'Average','micro');
confusedMatrix.f1_weighted       = cm_f1score(classMetrics,'Average','weighted');

%------------------------------------------------------------
% Step 4: ROC (requires per-example scores)
%   - Binary: yScore is N×1 for positive class = classLabels(2)
%   - Multi:  yScore is N×K, columns aligned with classLabels
%------------------------------------------------------------
try
    if ~isempty(scores)
        roc = cm_roc( harmonizeLabels(y, classLabels), normalizeScores(scores, classLabels), classLabels );
        confusedMatrix.roc = roc;   % perClass + micro (with AUC)
    end
catch ME
    % Don't hard-fail the whole report if ROC inputs are off
    confusedMatrix.roc_error = ME.message;
end
end

% --------- helpers (local to file) ---------------------------------------
function yOut = harmonizeLabels(yIn, classLabels)
    % Ensure y is categorical with same categories & order as classLabels
    if ~iscategorical(yIn)
        yIn = categorical(yIn);
    end
    % Map to target categories; unknowns -> <undefined>
    yOut = categorical(yIn, categories(classLabels), 'Protected', true);
    % Reorder categories to match classLabels
    yOut = reordercats(yOut, categories(classLabels));
end

function S = normalizeScores(scores, classLabels)
    % Ensure scores shape matches classLabels:
    % - If binary and vector N×1: use as-is for pos class classLabels(2).
    % - If multi: N×K aligned with classLabels.
    K = numel(classLabels);
    if isvector(scores)
        scores = scores(:);
        if K ~= 2
            error('For multi-class, scores must be N×K with K = numel(classLabels).');
        end
        S = scores;  % N×1 for binary (pos=classLabels(2))
        return;
    end
    if size(scores,2) ~= K
        error('scores must be N×K with columns aligned to classLabels.');
    end
    S = scores;
end
