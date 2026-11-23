function files = foldFiles(projectLoc, k)
%FOLDFILES  Sorted list of fold files present for experiment key k.
folder = dataFolder(projectLoc, k.datasetType, k.ncats, k.domain);
if startsWith(lower(k.domain), 'source'), stem = 'source_fold'; else, stem = 'target_fold'; end
d = dir(fullfile(folder, [stem '*.mat']));
idx = zeros(1, numel(d));
for i = 1:numel(d)
    tok = regexp(d(i).name, [stem '(\d+)\.mat$'], 'tokens', 'once');
    if isempty(tok), idx(i) = NaN; else, idx(i) = str2double(tok{1}); end
end
keep = ~isnan(idx);
[idx, order] = sort(idx(keep));
d = d(keep); d = d(order);
files = struct('fold', num2cell(idx), 'path', fullfile({d.folder}, {d.name}));
end
