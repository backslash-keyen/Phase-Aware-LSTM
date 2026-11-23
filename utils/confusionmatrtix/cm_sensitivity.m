function sensitivity = cm_sensitivity(confusedMatrix, varargin)
%CM_SENSITIVITY Recall/TPR.
%   Binary (struct input with fields TP,FN):
%       sensitivity = TP / (TP + FN)
%   Multi-class (numeric square confusion matrix CM):
%       sensitivity(i) = TP_i / (TP_i + FN_i)  (vector per class)
%
%   Optional Name-Value for multi-class:
%       'Average' : 'none' (default) | 'macro' | 'micro'
%           - 'none'  : returns per-class vector
%           - 'macro' : unweighted mean of class recalls
%           - 'micro' : global TP / (global TP + global FN)

    p = inputParser;
    addParameter(p, 'Average', 'none', @(s)ischar(s) || isstring(s));
    parse(p, varargin{:});
    avg = lower(string(p.Results.Average));

    if isstruct(confusedMatrix)
        % --- Binary behavior preserved (field name unchanged) ---
        TP = confusedMatrix.truePositive;
        FN = confusedMatrix.falseNegetive; % preserving original spelling
        denom = TP + FN;
        sensitivity = TP ./ denom;

    elseif isnumeric(confusedMatrix) && ismatrix(confusedMatrix) ...
            && size(confusedMatrix,1) == size(confusedMatrix,2)
        % --- Multi-class ---
        CM = confusedMatrix;
        TP = diag(CM);
        FN = sum(CM,2) - TP;
        denom = TP + FN;
        sens_vec = TP ./ denom;
        sens_vec(denom == 0) = NaN;   % define 0/0 as NaN

        switch avg
            case "none"
                sensitivity = sens_vec;
            case "macro"
                sensitivity = mean(sens_vec, 'omitnan');
            case "micro"
                TPg = sum(TP);
                FNg = sum(FN);
                sensitivity = TPg / (TPg + FNg);
            otherwise
                error('Unknown Average option. Use ''none'', ''macro'', or ''micro''.');
        end
    else
        error(['Input must be either a struct with fields truePositive, ', ...
               'falseNegetive (binary), or a square numeric confusion matrix (multi-class).']);
    end
end
