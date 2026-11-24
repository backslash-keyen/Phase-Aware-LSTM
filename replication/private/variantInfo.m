function v = variantInfo(names, sourceModelRoot)
%VARIANTINFO  Opt-in code variants. '' or {} = released code unchanged (default).
%
%   v = variantInfo('freezeIntended')
%   v = variantInfo({'authorSource','targetDomain2'}, 'M:\Earth\pretrainedModels')
%
% Variants combine. Each one changes behaviour only by
%   * putting replication/variants/<name> first on the path (path variants), or
%   * overriding fields of the configureExperiment object (config variants).
% Experiments a combination affects are renamed <id>__<name1>__<name2>
% (names in canonical order), with their own fold files, cval files,
% summary CSV and tables, so they never mix with, or get skipped because of,
% the unchanged-code results.
%
% Available
%   freezeIntended  path    DeepRNN learn-rate factors 0.1 / 0.2 actually applied
%   targetDomain2   path    run 'target-domain-2' (F3-F7) through the original
%                           target-domain options and builders
%   authorSource    config  target runs load the authors' source models from
%                           sourceModelRoot (read-only); source runs are refused
%   fastPhaseLSTM   path    speed only: the phase-aware projection uses
%                           phaseAwareLSTMFast (same arithmetic, no full-size
%                           copy of Y per time step), bit-identical
%   fastestPhaseLSTM path   speed only: phaseAwareLSTMFastest (version 3,
%                           same model, one stacked matmul per step; equal
%                           to the original up to float rounding)
%   backwardPhaseLSTM path  speed only: phaseAwareLSTMBackward (version 4,
%                           version-3 forward + hand-written backward pass)
%   fusedPhaseLSTM  path    speed only: phaseAwareLSTMFused (version 5,
%                           version 4 with each time step fused into one GPU
%                           kernel, forward and backward; CPU = version 4)
%   The phase-LSTM speed variants replace the same layer, so at most one of
%   them can be used at a time.

if nargin < 1 || isempty(names), names = {}; end
if nargin < 2 || isempty(sourceModelRoot), sourceModelRoot = 'M:\Earth\pretrainedModels'; end
names = cellstr(names);
order = {'authorSource','freezeIntended','targetDomain2','fastPhaseLSTM','fastestPhaseLSTM','backwardPhaseLSTM','fusedPhaseLSTM'};
unknown = setdiff(names, order);
if ~isempty(unknown)
    error('replication:unknownVariant', 'Unknown variant(s): %s. Known: %s.', ...
          strjoin(unknown, ', '), strjoin(order, ', '));
end
names = order(ismember(order, names));        % canonical order -> stable ids
speedVariants = {'fastPhaseLSTM','fastestPhaseLSTM','backwardPhaseLSTM','fusedPhaseLSTM'};
if nnz(ismember(names, speedVariants)) > 1
    error('replication:variantConflict', 'Use at most one of %s.', strjoin(speedVariants, ', '));
end

repDir = fileparts(fileparts(mfilename('fullpath')));
v.names        = names;
v.name         = strjoin(names, '+');          % '' when none
v.suffix       = strjoin(strcat('__', names), '');
v.dirs         = {};
v.extraDomains = {};
v.sourceModelRoot = '';
tests = {};

for i = 1:numel(names)
    switch names{i}
        case 'freezeIntended'
            v.dirs{end+1} = fullfile(repDir, 'variants', 'freezeIntended');
            % freezeFunctionalLayers is called only when a projection is put
            % in front of a pretrained source network
            tests{end+1} = @(k) ~startsWith(lower(k.domain), 'source') && ...
                any(strcmpi(k.layerName, {'ffnn','lstm','gru','phase-lstm'})); %#ok<AGROW>
        case 'targetDomain2'
            v.dirs{end+1} = fullfile(repDir, 'variants', 'targetDomain2');
            v.extraDomains{end+1} = 'target-domain-2';
            tests{end+1} = @(k) strcmpi(k.domain, 'target-domain-2'); %#ok<AGROW>
        case 'authorSource'
            v.sourceModelRoot = char(sourceModelRoot);
            % every target run loads a pretrained source model
            tests{end+1} = @(k) ~startsWith(lower(k.domain), 'source'); %#ok<AGROW>
        case {'fastPhaseLSTM','fastestPhaseLSTM','backwardPhaseLSTM','fusedPhaseLSTM'}
            v.dirs{end+1} = fullfile(repDir, 'variants', names{i});
            tests{end+1} = @(k) strcmpi(k.layerName, 'phase-lstm'); %#ok<AGROW>
    end
end
for i = 1:numel(v.dirs)
    if ~isfolder(v.dirs{i})
        error('replication:variantFolder', 'Variant folder missing: %s', v.dirs{i});
    end
end
if isempty(tests)
    v.affects   = @(k) false;
    v.suffixFor = @(k) '';
    v.nameFor   = @(k) '';
else
    hits = @(k) cellfun(@(f) f(k), tests);
    v.affects   = @(k) any(hits(k));
    % label each experiment only with the variants that actually change it
    v.suffixFor = @(k) strjoin(strcat('__', names(hits(k))), '');
    v.nameFor   = @(k) strjoin(names(hits(k)), '_');
end
end
