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
%   signedPhaseLSTM path    DIAGNOSTIC, changes the model: version 5 with the
%                           signed hidden output o.*tanh(sign(Re c+Im c).*|c|)
%                           of the author's intermediate layers (released:
%                           o.*tanh(|c|), never negative)
%   noCarryPhaseLSTM path   DIAGNOSTIC, changes the model: released layer with
%                           the June 2025 cstm_arch1 behaviours (no memory
%                           carry between steps; gates and fed-back phase
%                           clamped at eps); autodiff, original speed
%   notrigPhaseLSTM path    REPRODUCTION of the Nov 2025 layer behind the
%                           paper's CWL phase-lstm results (phaseAwareLSTM_notrig:
%                           theta = atan(H), clip +-5, unit-forget b1 and b2),
%                           plus its options routing (CWL C2 PhaseAware options,
%                           20 target epochs); autodiff, original speed
%   halfAngMatchedPhaseLSTM path  DIAGNOSTIC pair to notrigPhaseLSTM: released
%                           half-angle layer (fused) with notrig's clip +-5,
%                           biases and options, so only the angle map differs
%   piInputPhaseLSTM path   released layer (fused) with phase bias pi on the
%                           input rotation (one init line) + the phase-aware
%                           options, 20 target epochs: the minimal fix
%   notrigFastPhaseLSTM path  FAST port (fused) of the Nov 2025 notrig layer
%                           behind the paper's CWL phase-lstm results, with
%                           its options routing; same model as notrigPhaseLSTM
%   ver5FastPhaseLSTM path  FAST port (fused) of phaseAwareLSTM_ver5 (Oct 2025,
%                           run as 'phase-lstm-ver5' in Nov 2025): notrig
%                           without clipping and with phase bias pi; notrig
%                           options routing
%   pv_<version>    path    FAST port of one of Keyen's phase-layer versions
%                           (replication/versions/fast/<version>, e.g.
%                           pv_ver1, pv_cstmArch2Apr25), trained with the tuned
%                           options; see replication/versions/README.md
%   pc_<version>    path    the paper's notrig layer with tracked corrections
%                           (replication/phaseLayer, phaseAwareLSTMCorr; e.g.
%                           pc_port, pc_gradCap, pc_complexH), trained with the
%                           tuned options; see replication/phaseLayer/README.md
%   phaseAwareOptions path  CWL DeepRNN phase-lstm target runs train with the
%                           shipped get_trainingOptions_CWL_C2_PhaseAware.m,
%                           which the released get_trainingOptions.m never calls
%   The phase-LSTM speed variants replace the same layer, so at most one of
%   them (and signedPhaseLSTM) can be used at a time.

if nargin < 1 || isempty(names), names = {}; end
if nargin < 2 || isempty(sourceModelRoot), sourceModelRoot = 'M:\Earth\pretrainedModels'; end
names = cellstr(names);
pvDir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'versions', 'fast');
pvNames = names(startsWith(names, 'pv_'));
pcDir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'phaseLayer');
pcNames = names(startsWith(names, 'pc_'));
for i = 1:numel(pcNames)
    if ~isfolder(fullfile(pcDir, 'versions', extractAfter(pcNames{i}, 'pc_')))
        error('replication:unknownVariant', 'Unknown phase-layer correction version %s (folders in %s).', pcNames{i}, fullfile(pcDir, 'versions'));
    end
end
for i = 1:numel(pvNames)
    if ~isfolder(fullfile(pvDir, extractAfter(pvNames{i}, 'pv_'))) || strcmp(pvNames{i}, 'pv_common')
        error('replication:unknownVariant', 'Unknown phase-layer version %s (folders in %s).', pvNames{i}, pvDir);
    end
end
order = {'authorSource','freezeIntended','targetDomain2','fastPhaseLSTM','fastestPhaseLSTM','backwardPhaseLSTM','fusedPhaseLSTM','signedPhaseLSTM','noCarryPhaseLSTM','notrigPhaseLSTM','halfAngMatchedPhaseLSTM','piInputPhaseLSTM','notrigFastPhaseLSTM','ver5FastPhaseLSTM','phaseAwareOptions'};
order = [order, sort(pvNames), sort(pcNames)];
unknown = setdiff(names, order);
if ~isempty(unknown)
    error('replication:unknownVariant', 'Unknown variant(s): %s. Known: %s.', ...
          strjoin(unknown, ', '), strjoin(order, ', '));
end
names = order(ismember(order, names));        % canonical order -> stable ids
if nnz(ismember({'phaseAwareOptions','targetDomain2','notrigPhaseLSTM','halfAngMatchedPhaseLSTM','piInputPhaseLSTM','notrigFastPhaseLSTM','ver5FastPhaseLSTM'}, names)) + numel(pvNames) + numel(pcNames) > 1
    error('replication:variantConflict', 'phaseAwareOptions, targetDomain2 and the phase-layer variants with their own options all replace get_trainingOptions; use one.');
end
speedVariants = {'fastPhaseLSTM','fastestPhaseLSTM','backwardPhaseLSTM','fusedPhaseLSTM','signedPhaseLSTM','noCarryPhaseLSTM','notrigPhaseLSTM','halfAngMatchedPhaseLSTM','piInputPhaseLSTM','notrigFastPhaseLSTM','ver5FastPhaseLSTM'};
if nnz(ismember(names, speedVariants)) + numel(pvNames) + numel(pcNames) > 1
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
        case {'notrigPhaseLSTM','halfAngMatchedPhaseLSTM','piInputPhaseLSTM','notrigFastPhaseLSTM','ver5FastPhaseLSTM'}
            v.dirs{end+1} = fullfile(repDir, 'variants', names{i});
            tests{end+1} = @(k) strcmpi(k.layerName, 'phase-lstm'); %#ok<AGROW>
        case pvNames
            v.dirs{end+1} = fullfile(pvDir, 'common');
            v.dirs{end+1} = fullfile(pvDir, extractAfter(names{i}, 'pv_'));
            tests{end+1} = @(k) strcmpi(k.layerName, 'phase-lstm'); %#ok<AGROW>
        case pcNames
            v.dirs{end+1} = fullfile(pcDir, 'common');
            v.dirs{end+1} = fullfile(pcDir, 'versions', extractAfter(names{i}, 'pc_'));
            tests{end+1} = @(k) strcmpi(k.layerName, 'phase-lstm'); %#ok<AGROW>
        case 'phaseAwareOptions'
            v.dirs{end+1} = fullfile(repDir, 'variants', 'phaseAwareOptions');
            tests{end+1} = @(k) strcmpi(k.datasetType, 'CWL') && strcmpi(k.typeNet, 'DeepRNN') && ...
                strcmpi(k.layerName, 'phase-lstm') && startsWith(lower(k.domain), 'target'); %#ok<AGROW>
        case {'fastPhaseLSTM','fastestPhaseLSTM','backwardPhaseLSTM','fusedPhaseLSTM','signedPhaseLSTM','noCarryPhaseLSTM'}
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
