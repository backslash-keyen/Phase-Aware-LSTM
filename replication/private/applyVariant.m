function T = applyVariant(T, v)
%APPLYVARIANT  Point the table cells a variant affects at the variant's experiments.
if isempty(v.name), return; end
for t = 1:numel(T)
    for r = 1:numel(T(t).rows)
        for c = 1:numel(T(t).rows(r).cells)
            k = T(t).rows(r).cells{c};
            if strcmp(k.kind, 'exp') && v.affects(k)
                k.id = [k.id v.suffix];
                T(t).rows(r).cells{c} = k;
            end
        end
    end
    T(t).caption = sprintf('%s [variant: %s]', T(t).caption, v.name);
end
end
