function confusedMatrix = cm_buildMetrics_fromCounts(classLabels, classMetrics, yTrue, yScore)
%CM_BUILDMETRICS_FROMCOUNTS
% Assemble your metrics struct from absolute-count confusion matrix.
% Preserves binary field names and adds multi-class vectors.

% Store labels and CM
confusedMatrix.classLabels  = classLabels;   % categorical Kx1 (ordered)
confusedMatrix.classMetrics = classMetrics;  % numeric KxK (counts)

% Core counts (scalar for K==2, vectors for K>=3)
confusedMatrix.truePositive  = cm_truepositives(classLabels, classMetrics);
confusedMatrix.trueNegative  = cm_truenegatives(classLabels, classMetrics);
confusedMatrix.falseNegetive = cm_falsenegatives(classLabels, classMetrics); % spelling preserved
confusedMatrix.falsePositive = cm_falsepositives(classLabels, classMetrics);

% Accuracy & classwise metrics
confusedMatrix.accuracy     = cm_accuracy(classMetrics);                          % scalar
confusedMatrix.sensitivity  = cm_sensitivity(classMetrics, 'Average','none');     % Kx1
confusedMatrix.f1           = cm_f1score(classMetrics, 'Average','none');         % Kx1
confusedMatrix.f1_macro     = cm_f1score(classMetrics, 'Average','macro');
confusedMatrix.f1_micro     = cm_f1score(classMetrics, 'Average','micro');
confusedMatrix.f1_weighted  = cm_f1score(classMetrics, 'Average','weighted');
confusedMatrix.sensitivity_macro = cm_sensitivity(classMetrics,'Average','macro');
confusedMatrix.sensitivity_micro = cm_sensitivity(classMetrics,'Average','micro');

% ROC (needs per-example scores)
try
    if exist('yScore','var') && ~isempty(yScore)
        % Harmonize yTrue to the same category order
        yTrueH = categorical(yTrue, categories(classLabels), 'Protected', true);
        yTrueH = reordercats(yTrueH, categories(classLabels));

        % Ensure yScore shape matches: N×1 (binary) or N×K (multi-class)
        if numel(categories(classLabels)) == 2 && isvector(yScore)
            roc = cm_roc(yTrueH, yScore(:), classLabels);
        else
            if size(yScore,2) ~= numel(categories(classLabels))
                error('yScore must be N×K with columns aligned to classLabels.');
            end
            roc = cm_roc(yTrueH, yScore, classLabels);
        end
        confusedMatrix.roc = roc;
    end
catch ME
    confusedMatrix.roc_error = ME.message;
end
end
