function grid = experimentGrid(varargin)
%EXPERIMENTGRID  Every experiment needed to rebuild the paper's tables.
%
%   grid = experimentGrid()                         % everything, run order
%   grid = experimentGrid('tables', {'table2','table4'})
%   grid = experimentGrid('datasets', {'CWL'})
%
% The grid is derived from paperTables(): every table cell that points to
% an experiment contributes that experiment, and every target experiment
% pulls in the source experiment whose saved models it loads. Rows are
% sorted so that all source-domain runs come first (they write
% pretrainedModels/), then all target-domain runs.

p = inputParser;
addParameter(p, 'tables',   {});
addParameter(p, 'datasets', {});
addParameter(p, 'typeNets', {});
parse(p, varargin{:});
o = p.Results;

T = paperTables();
if ~isempty(o.tables)
    T = T(ismember({T.id}, cellstr(o.tables)));
end

keys = {};
usedBy = containers.Map('KeyType','char','ValueType','any');
for t = 1:numel(T)
    for r = 1:numel(T(t).rows)
        cells = T(t).rows(r).cells;
        for c = 1:numel(cells)
            k = cells{c};
            if ~strcmp(k.kind, 'exp'), continue; end
            [keys, usedBy] = addKey(keys, usedBy, k, T(t).id);
            dep = sourceDependency(k);
            if ~isempty(dep)
                [keys, usedBy] = addKey(keys, usedBy, dep, ['source for ' k.id]);
            end
        end
    end
end
grid = [keys{:}];
if isempty(grid), return; end

if ~isempty(o.datasets)
    grid = grid(ismember(upper({grid.datasetType}), upper(cellstr(o.datasets))));
end
if ~isempty(o.typeNets)
    grid = grid(ismember(lower({grid.typeNet}), lower(cellstr(o.typeNets))));
end

% Run order: stage (source first), then dataset, classes, network, domain, layer
stage = double(~startsWith(lower({grid.domain}), 'source'));   % 0 = source, 1 = target
src2  = double(strcmpi({grid.domain}, 'source-domain-2'));     % plain source before dual-channel source
[~, order] = sortrows([stage(:), src2(:), (1:numel(grid))']);
grid = grid(order);

for g = 1:numel(grid)
    grid(g).usedBy = strjoin(usedBy(grid(g).id), '; ');
end
end

function [keys, usedBy] = addKey(keys, usedBy, k, who)
if isKey(usedBy, k.id)
    usedBy(k.id) = unique([usedBy(k.id), {who}], 'stable');
else
    keys{end+1} = k;
    usedBy(k.id) = {who};
end
end
