function [t, df, p, meanDiff] = pairedTTest1(a, b)
%PAIREDTTEST1  One-tailed paired t-test, H1: mean(a - b) > 0.
% Base MATLAB only (no Statistics Toolbox): the t CDF is computed with
% betainc. Matches ttest(a, b, 'Tail', 'right').
d = a(:) - b(:);
d = d(~isnan(d));
n = numel(d);
df = n - 1;
meanDiff = mean(d);
if n < 2
    t = NaN; p = NaN; return
end
s = std(d);
if s == 0
    t = sign(meanDiff) * Inf;
    p = double(meanDiff <= 0);
    if meanDiff == 0, p = NaN; end
    return
end
t = meanDiff / (s / sqrt(n));
% upper tail of Student t: P(T > t)
x = df / (df + t^2);
tail = 0.5 * betainc(x, df/2, 0.5);
if t >= 0, p = tail; else, p = 1 - tail; end
end
