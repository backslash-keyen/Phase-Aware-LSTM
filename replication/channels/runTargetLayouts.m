function runTargetLayouts(varargin)
%RUNTARGETLAYOUTS  Screen other CWL target channel layouts (extension study).
%
% From the repository root, after buildTargetLayoutFolds:
%   addpath('replication/channels');
%   runTargetLayouts()                                   % fold 2, 3 layouts, LSTM + phase-aware
%   runTargetLayouts('folds', 1:10)
%   runTargetLayouts('layouts', {'FzP4'}, 'projections', {'phase'})
%
% Runs cwl_N2_target-domain_DeepRNN_<layer> through runReplication with the
% opt-in variant tl_<layout> (the pretrained DeepRNN source models, options
% and projection builders are the usual target-domain ones; only the two
% channel rows of the target data differ). Projections:
%   'lstm'   LSTM projection, released options       -> <id>__tl_<layout>
%   'phase'  phase-aware projection with the pi fix  -> <id>__piInputPhaseLSTM__tl_<layout>
%            (piInputPhaseLSTM: fused layer, phase bias pi on the input
%            rotation, tuned phase-aware options, 20 target epochs)
% The Fp1/Fz reference is the existing runs (no variant / piInputPhaseLSTM).
% One job at a time; before each it waits until 'minFreeGB' of RAM is free.
p = inputParser;
addParameter(p, 'layouts', {'FzPz', 'FzPOz', 'AF3AF4'});
addParameter(p, 'projections', {'lstm', 'phase'});
addParameter(p, 'folds', 2);
addParameter(p, 'minFreeGB', 0.5);
addParameter(p, 'waitMinutes', 30);
addParameter(p, 'dryRun', false);
parse(p, varargin{:}); o = p.Results;

repDir = fileparts(fileparts(mfilename('fullpath')));
addpath(repDir);
for i = 1:numel(o.layouts)
    for j = 1:numel(o.projections)
        switch o.projections{j}
            case 'lstm',  id = 'cwl_N2_target-domain_DeepRNN_lstm';       variant = {['tl_' o.layouts{i}]};
            case 'phase', id = 'cwl_N2_target-domain_DeepRNN_phase-lstm'; variant = {'piInputPhaseLSTM', ['tl_' o.layouts{i}]};
            otherwise, error('unknown projection %s (lstm or phase)', o.projections{j});
        end
        waitForRam(o.minFreeGB, o.waitMinutes);
        fprintf('\n=== layout %s, %s, folds %s ===\n', o.layouts{i}, o.projections{j}, mat2str(o.folds));
        t = tic;
        runReplication('ids', {id}, 'folds', o.folds, 'variant', variant, 'dryRun', o.dryRun);
        fprintf('=== layout %s, %s done in %.1f min ===\n', o.layouts{i}, o.projections{j}, toc(t) / 60);
    end
end
end

function waitForRam(minGB, maxMin)
if ~ispc, return; end
for k = 0:maxMin
    [~, s] = memory();
    if s.PhysicalMemory.Available / 2^30 >= minGB, return; end
    if k == 0, fprintf('[layouts] waiting for %.1f GB free RAM...\n', minGB); end
    pause(60);
end
error('runTargetLayouts:ram', 'Less than %.1f GB RAM free for %d min; stopping.', minGB, maxMin);
end
