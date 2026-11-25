function runPhaseVersions(varargin)
%RUNPHASEVERSIONS  Train fast ports of the phase-aware LSTM versions, one at a time.
%
% From the repository root:
%   addpath('replication/versions');
%   runPhaseVersions()                               % notrig and ver5, fold 2
%   runPhaseVersions('ids', {'ver1','ver2'}, 'folds', 2)
%   runPhaseVersions('ids', 'all', 'folds', 2)       % every ported version, fold 2
%   runPhaseVersions('ids', {'notrig'}, 'folds', 1:10)
%
% Each version runs cwl_N2_target-domain_DeepRNN_phase-lstm (Fp1/Fz) through
% runReplication with the tuned phase-aware options (get_trainingOptions in
% replication/versions/fast/common). Results are stored per variant, like any
% other variant (<id>__pv_<version> or <id>__notrigFastPhaseLSTM ...), and a
% rerun resumes from the next unsaved fold.
% notrig and ver5 use their dedicated ports (notrigFastPhaseLSTM,
% ver5FastPhaseLSTM); every other version uses 'pv_<version>'.
% Before each job it waits until at least 0.5 GB of RAM is free (checked
% every minute, gives up after 'waitMinutes').
p = inputParser;
addParameter(p, 'ids', {'notrig', 'ver5'});
addParameter(p, 'folds', 2);
addParameter(p, 'experiment', 'cwl_N2_target-domain_DeepRNN_phase-lstm');
addParameter(p, 'minFreeGB', 0.5);
addParameter(p, 'waitMinutes', 30);
addParameter(p, 'dryRun', false);
parse(p, varargin{:}); o = p.Results;

verDir  = fileparts(mfilename('fullpath'));
repDir  = fileparts(verDir);
addpath(repDir);
V = phaseVersionList();
ported = {V(strcmp({V.status}, 'fast')).id};
ids = o.ids;
if ischar(ids) && strcmp(ids, 'all')
    % paper-pipeline versions first
    ids = [{'notrig', 'ver5', 'halfAng'}, setdiff(ported, {'notrig', 'ver5', 'halfAng'}, 'stable')];
end
ids = cellstr(ids);
bad = setdiff(ids, ported);
assert(isempty(bad), 'not a ported version: %s (see phaseVersionList)', strjoin(bad, ', '));

for i = 1:numel(ids)
    switch ids{i}
        case 'notrig', variant = 'notrigFastPhaseLSTM';
        case 'ver5',   variant = 'ver5FastPhaseLSTM';
        otherwise,     variant = ['pv_' ids{i}];
    end
    waitForRam(o.minFreeGB, o.waitMinutes);
    fprintf('\n=== %s (%d/%d): variant %s, folds %s ===\n', ids{i}, i, numel(ids), variant, mat2str(o.folds));
    t = tic;
    runReplication('ids', {o.experiment}, 'folds', o.folds, 'variant', variant, 'dryRun', o.dryRun);
    fprintf('=== %s done in %.1f min ===\n', ids{i}, toc(t) / 60);
end
end

function waitForRam(minGB, maxMin)
if ~ispc, return; end                 % memory() exists on Windows only
for k = 0:maxMin
    [~, s] = memory();
    free = s.PhysicalMemory.Available / 2^30;
    if free >= minGB, return; end
    fprintf('only %.2f GB RAM free (need %.1f), waiting a minute\n', free, minGB);
    pause(60);
end
error('runPhaseVersions:ram', 'less than %.1f GB RAM free for %d minutes, stopping', minGB, maxMin);
end
