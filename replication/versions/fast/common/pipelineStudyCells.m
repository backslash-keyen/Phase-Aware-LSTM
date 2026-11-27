function C = pipelineStudyCells(id)
%PIPELINESTUDYCELLS  Cells of the 5-fold phase-aware pipeline study (2026-11-26).
%
%   C = pipelineStudyCells()        all cells (struct array, run order)
%   c = pipelineStudyCells('clip5') one cell
%
% One factor at a time around a single baseline, all through the same fused
% engine (replication/versions/fast/common/phaseVersionEngine.m) and the same
% layer class (phaseAwareLSTM_study_fast), so cells differ only in the field
% named in 'factor'. Every cell trains CWL C2 DeepRNN target (Fp1/Fz),
% folds 1-5, with the tuned phase-aware options unless 'opts' says otherwise.
%
% Baseline 'base' = released half-angle layer + phase bias pi on the input
% rotation (b2 rows H+1:2H) + tuned options = the minimal fix
% (piInputPhaseLSTM), engine form identical to pv_piSign_none.
%
% Fields
%   id, factor, label
%   eng     name-value pairs for phaseVersionEngine.config
%   b1      'released' (unit-forget helper, +1 lands on input-gate rows H+1:2H-1)
%           | 'forget' (+1 on the forget-gate rows 1:H) | 'zero'
%   b2val   value put on the b2 row blocks in b2blk (others 0)
%   b2blk   gate blocks of b2 (1 f, 2 i, 3 c, 4 o) that get b2val
%   qScale  factor on the orthogonal Q init (1 = released)
%   opts    'tuned' | 'default' (released DeepRNN routing)
%
% The output-sign factor reuses pv_piSign_{none,sum,realProj,tanhRe} (same
% engine and base), and the paper's own layer is notrigFastPhaseLSTM; see
% runPipelineStudy.

base = struct('id','base','factor','baseline','label','half-angle, pi on input rotation, clip 20', ...
    'eng',{{'map','halfAngle','k',0.5,'clip',20,'magFloor',1e-9,'sign','none'}}, ...
    'b1','released','b2val',pi,'b2blk',2,'qScale',1,'opts','tuned');

C = base;
C(end+1) = mod(base, 'angAtan',   'angle map',  'atan (notrig)',              'eng', setEng(base.eng, 'map', 'atan'));
C(end+1) = mod(base, 'angPiTanh', 'angle map',  'pi*tanh',                    'eng', setEng(base.eng, 'map', 'piTanh'));
C(end+1) = mod(base, 'angDirect', 'angle map',  'raw pre-activation (no map)', 'eng', setEng(base.eng, 'map', 'direct'));
C(end+1) = mod(base, 'b2zero',    'phase bias', '0 (released)',               'b2val', 0);
C(end+1) = mod(base, 'b2unit',    'phase bias', '1 on input rotation (notrig)', 'b2val', 1);
C(end+1) = mod(base, 'b2all',     'phase-bias gates', 'pi on f, i, c, o',     'b2blk', 1:4);
C(end+1) = mod(base, 'b2forget',  'phase-bias gates', 'pi on forget only',    'b2blk', 1);
C(end+1) = mod(base, 'b2inCand',  'phase-bias gates', 'pi on input + candidate', 'b2blk', [2 3]);
C(end+1) = mod(base, 'clip5',     'clip',       '+-5 (notrig)',               'eng', setEng(base.eng, 'clip', 5));
C(end+1) = mod(base, 'clipInf',   'clip',       'none',                       'eng', setEng(base.eng, 'clip', Inf));
C(end+1) = mod(base, 'b1forget',  'init',       'b1 +1 on true forget rows',  'b1', 'forget');
C(end+1) = mod(base, 'b1zero',    'init',       'b1 = 0',                     'b1', 'zero');
C(end+1) = mod(base, 'qSmall',    'init',       'Q init x 0.1',               'qScale', 0.1);
C(end+1) = mod(base, 'optsDefault', 'options',  'released DeepRNN options',   'opts', 'default');

if nargin
    k = strcmp({C.id}, id);
    assert(any(k), 'pipelineStudyCells: unknown cell %s', id);
    C = C(k);
end
end

function c = mod(c, id, factor, label, varargin)
c.id = id; c.factor = factor; c.label = label;
for i = 1:2:numel(varargin), c.(varargin{i}) = varargin{i+1}; end
end

function e = setEng(e, name, val)
k = find(strcmp(e(1:2:end), name)) * 2;
if isempty(k), e(end+1:end+2) = {name, val}; else, e{k} = val; end
end
