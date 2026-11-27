function runPipelineStudy(varargin)
%RUNPIPELINESTUDY  5-fold phase-aware pipeline study, one job at a time, resumable.
%
% From the repository root:
%   addpath('replication/study');
%   runPipelineStudy()                          % whole queue, folds 1-5
%   runPipelineStudy('only', {'base','clip5'})  % some cells
%   runPipelineStudy('dryRun', true)
%
% Queue (priority order, so a partial run is already useful):
%   study cells from pipelineStudyCells (variants pv_ps_<cell>), then the
%   output-sign layers pv_piSign_{none,sum,realProj,tanhRe} (same engine and
%   base; folds already on disk are skipped), then the paper's own layer
%   notrigFastPhaseLSTM. Each job is CWL C2 DeepRNN target phase-lstm (Fp1/Fz)
%   via runReplication; every fold is saved as it finishes and a rerun resumes.
% Before each job it waits for 'minFreeGB' of free RAM.
p = inputParser;
addParameter(p, 'folds', 1:5);
addParameter(p, 'only', {});
addParameter(p, 'experiment', 'cwl_N2_target-domain_DeepRNN_phase-lstm');
addParameter(p, 'minFreeGB', 2);
addParameter(p, 'waitMinutes', 30);
addParameter(p, 'dryRun', false);
parse(p, varargin{:}); o = p.Results;

studyDir = fileparts(mfilename('fullpath'));
repDir   = fileparts(studyDir);
addpath(repDir);
addpath(fullfile(repDir, 'versions', 'fast', 'common'));
C = pipelineStudyCells();
jobs = [strcat('pv_ps_', {C.id}), ...
        {'pv_piSign_none', 'pv_piSign_sum', 'pv_piSign_realProj', 'pv_piSign_tanhRe', 'notrigFastPhaseLSTM'}];
if ~isempty(o.only)
    want = cellstr(o.only);
    jobs = jobs(ismember(jobs, want) | ismember(erase(jobs, 'pv_ps_'), want));
end
for i = 1:numel(jobs)
    waitForRam(o.minFreeGB, o.waitMinutes);
    fprintf('\n=== study job %d/%d: %s, folds %s ===\n', i, numel(jobs), jobs{i}, mat2str(o.folds));
    t = tic;
    runReplication('ids', {o.experiment}, 'folds', o.folds, 'variant', jobs{i}, 'dryRun', o.dryRun);
    fprintf('=== %s done in %.1f min ===\n', jobs{i}, toc(t) / 60);
end
end

function waitForRam(minGB, maxMin)
if ~ispc, return; end
for k = 0:maxMin
    [~, s] = memory();
    free = s.PhysicalMemory.Available / 2^30;
    if free >= minGB, return; end
    fprintf('only %.2f GB RAM free (need %.1f), waiting a minute\n', free, minGB);
    pause(60);
end
error('runPipelineStudy:ram', 'less than %.1f GB RAM free for %d minutes, stopping', minGB, maxMin);
end
