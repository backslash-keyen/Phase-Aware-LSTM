function roc = cm_roc(yTrue, yScore, classLabels, varargin)
%CM_ROC ROC curves (binary & multi-class one-vs-rest) + AUC.
%   Binary:
%       yTrue  : N×1 vector of labels (same type/values as classLabels)
%       yScore : N×1 numeric scores/probabilities for positive class
%       classLabels : [negClass, posClass] in desired order
%
%   Multi-class:
%       yTrue  : N×1 labels drawn from classLabels
%       yScore : N×K matrix of scores/probabilities, columns aligned to classLabels
%       classLabels : 1×K list of class IDs (numeric, logical, string, categorical)
%
%   Name-Value:
%       'UsePerfcurve' : true (default) | false
%
%   Output (STRUCT):
%       For binary:
%         roc.fpr, roc.tpr, roc.thresholds, roc.auc
%       For multi-class:
%         roc.perClass(i).label, .fpr, .tpr, .thresholds, .auc
%         roc.micro.fpr, roc.micro.tpr, roc.micro.thresholds, roc.micro.auc
%
%   Notes:
%     - Requires per-example scores; a confusion matrix is insufficient for ROC.
%     - If Statistics Toolbox is present, uses PERFcurve by default.
%     - Fallback computes ROC by sweeping unique thresholds.

    p = inputParser;
    addParameter(p, 'UsePerfcurve', true, @(b)islogical(b)||isscalar(b));
    parse(p, varargin{:});
    wantPerf = p.Results.UsePerfcurve && exist('perfcurve','file')==2;

    yTrue = yTrue(:);
    L = classLabels(:);

    if numel(L) == 2 && isvector(yScore) && size(yScore,2)==1
        % --- Binary ---
        posClass = L(2);
        if wantPerf
            [fpr,tpr,thr,auc] = perfcurve(yTrue, yScore, posClass);
        else
            [fpr,tpr,thr,auc] = localROC(yTrue == posClass, yScore);
        end
        roc.fpr = fpr; roc.tpr = tpr; roc.thresholds = thr; roc.auc = auc;
        return;
    end

    % --- Multi-class (one-vs-rest for each class) ---
    if ~ismatrix(yScore) || size(yScore,2) ~= numel(L)
        error('For multi-class, yScore must be N×K with columns aligned to classLabels.');
    end
    N = numel(yTrue);
    K = numel(L);
    roc.perClass = repmat(struct('label',[], 'fpr',[],'tpr',[],'thresholds',[],'auc',[]), K, 1);

    % per-class (OvR)
    for i = 1:K
        label_i = L(i);
        y_bin   = (yTrue == label_i);
        score_i = yScore(:,i);
        if wantPerf
            [fpr,tpr,thr,auc] = perfcurve(y_bin, score_i, true);
        else
            [fpr,tpr,thr,auc] = localROC(y_bin, score_i);
        end
        roc.perClass(i).label = label_i;
        roc.perClass(i).fpr = fpr;
        roc.perClass(i).tpr = tpr;
        roc.perClass(i).thresholds = thr;
        roc.perClass(i).auc = auc;
    end

    % micro-average: pool all decisions
    Y_micro = false(N*K,1);
    S_micro = zeros(N*K,1);
    idx = 1;
    for i = 1:K
        rng_i = idx:(idx+N-1);
        Y_micro(rng_i) = (yTrue == L(i));
        S_micro(rng_i) = yScore(:,i);
        idx = idx + N;
    end
    if wantPerf
        [fpr,tpr,thr,auc] = perfcurve(Y_micro, S_micro, true);
    else
        [fpr,tpr,thr,auc] = localROC(Y_micro, S_micro);
    end
    roc.micro.fpr = fpr;
    roc.micro.tpr = tpr;
    roc.micro.thresholds = thr;
    roc.micro.auc = auc;
end

% --- Minimal internal ROC/AUC for when perfcurve is unavailable ---
function [fpr,tpr,thr,auc] = localROC(yPos, scores)
    % yPos: logical positives
    if ~islogical(yPos); yPos = logical(yPos); end
    scores = double(scores(:));
    yPos   = yPos(:);
    % sort by score desc
    [scores, order] = sort(scores, 'descend');
    yPos = yPos(order);

    P = sum(yPos);
    N = numel(yPos) - P;
    if P==0 || N==0
        fpr = [0;1]; tpr = [0;1]; thr = [Inf;-Inf]; auc = NaN; return;
    end

    % unique thresholds
    [thr, ia] = unique(scores, 'stable');              % descending
    thr = [Inf; thr; -Inf];                            % guard rails
    ia  = [0; ia; numel(scores)];
    tp = zeros(numel(thr),1);
    fp = zeros(numel(thr),1);

    % sweep thresholds: predict positive when score >= thr(k)
    % Efficient cum-sum trick
    cumPos = cumsum(yPos);                 % at rank r, #positives among top r
    cumNeg = ( (1:numel(yPos))' - cumPos); % #negatives among top r

    for k = 1:numel(thr)
        r = ia(k); % number of instances with score >= thr(k) minus 1 (due to padding)
        tp(k) = cumPos(max(r,0));
        fp(k) = cumNeg(max(r,0));
    end

    tpr = tp / P;
    fpr = fp / N;

    % Monotone fix (optional): ensure non-decreasing FPR, TPR
    [fpr, uniq] = unique(fpr, 'stable');
    tpr = tpr(uniq);
    thr = thr(uniq);

    % AUC via trapezoidal rule
    auc = trapz(fpr, tpr);
end
