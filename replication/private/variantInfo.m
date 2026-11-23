function v = variantInfo(name)
%VARIANTINFO  Opt-in code variants. '' = released code unchanged (default).
%
% A variant changes behaviour ONLY by putting replication/variants/<name>
% first on the path. Its results get their own experiment ids
% (<id>__<name>), fold files, cval files, summary CSV and tables, so they
% never mix with the unchanged-code results.
v.name = char(name);
v.dir  = '';
v.suffix = '';
v.affects = @(k) false;
if isempty(v.name), return; end
repDir = fileparts(fileparts(mfilename('fullpath')));
v.dir = fullfile(repDir, 'variants', v.name);
if ~isfolder(v.dir)
    error('replication:unknownVariant', 'Unknown variant "%s" (no folder %s).', v.name, v.dir);
end
v.suffix = ['__' v.name];
switch v.name
    case 'freezeIntended'
        % freezeFunctionalLayers is called only when a projection is put in
        % front of a pretrained source network
        v.affects = @(k) ~startsWith(lower(k.domain), 'source') && ...
            any(strcmpi(k.layerName, {'ffnn','lstm','gru','phase-lstm'}));
end
end
