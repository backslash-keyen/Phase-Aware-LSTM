function [classLabels, classMetrics] = confusionMatrix_extraction(confusingmatrix, y, scores)
%CONFUSIONMATRIX_EXTRACTION
% Returns the class labels (categorical, ordered) and ABSOLUTE confusion
% matrix (counts). Uses y & scores to reconstruct predictions so downstream
% metrics (TP/TN/FP/FN, accuracy, F1, etc.) are correct.

% --- 1) Get labels from the chart and coerce to ordered categorical ---
raw = confusingmatrix.ClassLabels;                 % could be numeric/string/cellstr/categorical
classNames = toClassNames(raw);                    % cellstr of non-blank names, Kx1
classLabels = categorical(classNames, classNames, 'Protected', true);  % ordered by display

% --- 2) Harmonize y and derive predictions from scores (argmax / threshold) ---
y = harmonizeY(y, classLabels);                    % categorical with same categories & order
yPred = derivePredictionsFromScores(scores, classLabels, numel(y));

% --- 3) Absolute confusion matrix in the chart's order (rows=actual, cols=pred) ---
classMetrics = confusionmat(y, yPred, 'Order', classLabels);
end

% ================== local helpers ==================

function names = toClassNames(raw)
    % Convert whatever arrives (numeric/string/cell/categorical) into a
    % non-empty cellstr of class names, preserving order.
    if iscategorical(raw)
        % Use the displayed values (not categories()) to preserve chart order
        vals = cellstr(raw(:));
    elseif isstring(raw)
        vals = cellstr(raw(:));
    elseif isnumeric(raw) || islogical(raw)
        vals = cellstr(string(raw(:)));
    elseif iscellstr(raw)
        vals = raw(:);
    else
        vals = cellstr(string(raw(:)));  % fallback
    end

    % Replace blanks or missing with "C1","C2",...
    blank = cellfun(@(s) isempty(s) || all(isspace(s)), vals);
    if any(blank)
        idx = find(blank);
        for k = 1:numel(idx)
            vals{idx(k)} = sprintf('C%d', idx(k));
        end
    end
    names = vals;
end

function yOut = harmonizeY(yIn, classLabels)
    % Coerce y to categorical with the same categories & order as classLabels.
    if ~iscategorical(yIn); yIn = categorical(yIn); end
    cats = cellstr(classLabels);                      % target names
    yOut = categorical(yIn, cats, 'Protected', true); % unknowns -> <undefined>
    yOut = reordercats(yOut, cats);                   % now valid (names, not values)
end

function yPred = derivePredictionsFromScores(scores, classLabels, N)
    K = numel(classLabels);
    if isempty(scores)
        error(['scores are required to reconstruct predictions. ', ...
               'Provide N×1 (binary) or N×K (multi-class) scores.']);
    end

    if isvector(scores)
        % Binary: scores are probs/scores for positive class = classLabels(2)
        if K ~= 2
            error('Vector scores provided but numel(classLabels) ~= 2.');
        end
        s = scores(:);
        if numel(s) ~= N; error('scores length must match numel(y).'); end
        pos = s >= 0.5;                                   % adjust threshold upstream if needed
        yPred = repmat(classLabels(1), N, 1);
        yPred(pos) = classLabels(2);
        return;
    end

    % Multi-class: N×K, argmax across columns aligned with classLabels
    if size(scores,1) ~= N || size(scores,2) ~= K
        error('scores must be N×K with columns aligned to classLabels.');
    end
    [~, idx] = max(scores, [], 2);
    yPred = classLabels(idx);
end
