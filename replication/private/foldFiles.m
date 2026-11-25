function files = foldFiles(projectLoc, k)
%FOLDFILES  Sorted list of fold files present for experiment key k.
% Returns an empty struct (fields fold, path) when the folder is missing
% or holds no fold files, so the caller can log and skip the experiment.
files = struct('fold', {}, 'path', {});
folder = dataFolder(projectLoc, k.datasetType, k.ncats, k.domain);
if ~isfolder(folder), return; end
if startsWith(lower(k.domain), 'source'), stem = 'source_fold'; else, stem = 'target_fold'; end
d = dir(fullfile(folder, [stem '*.mat']));
for i = 1:numel(d)
    tok = regexp(d(i).name, ['^' stem '(\d+)\.mat$'], 'tokens', 'once');
    if ~isempty(tok)
        files(end+1) = struct('fold', str2double(tok{1}), 'path', fullfile(d(i).folder, d(i).name)); %#ok<AGROW>
    end
end
if ~isempty(files)
    [~, order] = sort([files.fold]);
    files = files(order);
end
end
