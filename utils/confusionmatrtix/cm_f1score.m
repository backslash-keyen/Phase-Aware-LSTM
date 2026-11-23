function F1 = cm_f1score(confusedMatrix, varargin)
%CM_F1SCORE Compute F1 score.
%   Binary (STRUCT input with fields truePositive, trueNegative, falsePositive, falseNegetive):
%       Returns scalar F1 for the positive class.
%
%   Multi-class (numeric square confusion matrix CM):
%       Returns per-class F1 vector aligned with class order in CM.
%
%   Name-Value (multi-class only):
%       'Average' : 'none' (default) | 'macro' | 'micro' | 'weighted'
%
%   Convention: rows = actual, columns = predicted.

    p = inputParser;
    addParameter(p, 'Average', 'none', @(s)ischar(s)||isstring(s));
    parse(p, varargin{:});
    avg = lower(string(p.Results.Average));

    % --- Binary: struct path preserved (field names unchanged) ---
    if isstruct(confusedMatrix)
        TP = confusedMatrix.truePositive;
        FP = confusedMatrix.falsePositive;
        FN = confusedMatrix.falseNegetive;  % preserved spelling
        % (TN not needed)
        P = TP ./ (TP + FP);
        R = TP ./ (TP + FN);
        denom = P + R;
        F1 = 2 * P .* R ./ denom;
        F1(denom == 0) = NaN;
        return;
    end

    % --- Multi-class: numeric square CM ---
    CM = confusedMatrix;
    if ~isnumeric(CM) || size(CM,1) ~= size(CM,2)
        error('For multi-class, input must be a square numeric confusion matrix.');
    end

    TP = diag(CM);
    FP = sum(CM,1).' - TP;     % predicted as i, actually others
    FN = sum(CM,2)  - TP;      % actually i, predicted as others
    % (TN not needed)
    P = TP ./ (TP + FP);
    R = TP ./ (TP + FN);
    denom = P + R;
    f1_vec = 2 * P .* R ./ denom;
    f1_vec(denom == 0) = NaN;

    switch avg
        case "none"
            F1 = f1_vec;
        case "macro"
            F1 = mean(f1_vec, 'omitnan');
        case "micro"
            TPg = sum(TP);
            FPg = sum(FP);
            FNg = sum(FN);
            Pg = TPg / (TPg + FPg);
            Rg = TPg / (TPg + FNg);
            if Pg + Rg == 0
                F1 = NaN;
            else
                F1 = 2 * Pg * Rg / (Pg + Rg);
            end
        case "weighted"
            support = sum(CM,2);                 % actual counts per class
            w = support / sum(support);
            F1 = nansum(f1_vec .* w);
        otherwise
            error('Unknown Average option. Use ''none'', ''macro'', ''micro'', or ''weighted''.');
    end
end
