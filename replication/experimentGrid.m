function grid = experimentGrid(varargin)
%EXPERIMENTGRID  Every experiment needed to rebuild the paper's tables.
%
%   grid = experimentGrid()                         % everything, run order
%   grid = experimentGrid('tables', {'table2','table4'})
%   grid = experimentGrid('datasets', {'CWL'})
%   grid = experimentGrid('datasets','CWL', 'ncats',2, 'typeNets','DeepRNN', 'domains','source-domain')
%   grid = experimentGrid('ids', {'cwl_N2_source-domain_DeepRNN_deeprnn'})
%
% 'variant' (e.g. 'freezeIntended') renames the experiments that variant affects
% to <id>__<variant>, so their results are kept apart; see variantInfo.m.
%
% Filters (all optional, combined with AND): tables, datasets, ncats,
% typeNets, domains, layerNames, ids. Filtering never adds the source runs
% a target run depends on; those are checked on disk at run time.
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
addParameter(p, 'ncats',    []);
addParameter(p, 'domains',  {});
addParameter(p, 'layerNames', {});
addParameter(p, 'ids',      {});
addParameter(p, 'variant',  '');
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
if ~isempty(o.ncats)
    grid = grid(ismember([grid.ncats], o.ncats));
end
if ~isempty(o.domains)
    grid = grid(ismember(lower({grid.domain}), lower(cellstr(o.domains))));
end
if ~isempty(o.layerNames)
    grid = grid(ismember(lower({grid.layerName}), lower(cellstr(o.layerNames))));
end
v = variantInfo(o.variant);
for g = 1:numel(grid)
    grid(g).baseId = grid(g).id;
    if v.affects(grid(g)), grid(g).id = [grid(g).id v.suffixFor(grid(g))]; end
end
if ~isempty(o.ids)
    grid = grid(ismember({grid.id}, cellstr(o.ids)) | ismember({grid.baseId}, cellstr(o.ids)));
end
if isempty(grid), return; end

% Run order: stage (source first), then dataset, classes, network, domain, layer
stage = double(~startsWith(lower({grid.domain}), 'source'));   % 0 = source, 1 = target
src2  = double(strcmpi({grid.domain}, 'source-domain-2'));     % plain source before dual-channel source
[~, order] = sortrows([stage(:), src2(:), (1:numel(grid))']);
grid = grid(order);

for g = 1:numel(grid)
    grid(g).usedBy = strjoin(usedBy(grid(g).baseId), '; ');
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
