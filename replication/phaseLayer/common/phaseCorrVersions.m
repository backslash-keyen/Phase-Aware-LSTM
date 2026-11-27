function out = phaseCorrVersions(id)
%PHASECORRVERSIONS  Registry of the tracked phase-layer versions (paper layer + corrections).
%
%   list = phaseCorrVersions()        struct array: id, parent, change, set
%   cfg  = phaseCorrVersions('gradCap')   configuration for phaseAwareLSTMCorr
%
% Every version is the paper layer (phaseAwareLSTM_notrig, 3 Nov 2025) with
% the configuration fields in 'set' changed; nothing else differs. Versions
% after gradCap keep gradCap, because without it the layer never trains on
% this PC (see phaseCorrCapGrads), so each later version tests exactly one
% further change against gradCap. The replication run itself ('paper') is
% the unchanged original layer, variant notrigPhaseLSTM; 'port' is this
% class with the defaults and must behave the same.
cap = {'gradCap', 1.25};
V = {
% id                parent      change                                                                 set
'port'              'paper'     'phaseAwareLSTMCorr with the defaults: same model as the paper layer (hand-written backward)' {}
'gradCap'           'port'      'learnable gradients capped at L2 1.25 in double before trainNetwork (its single-precision clip overflows to a zero update)' cap
'forgetRow'         'gradCap'   'unit-forget +1 of b1 on the forget gate f (rows 1:H) instead of the candidate gate i (rows H+1:2H)' [cap, {'forgetRow', 'f'}]
'noPhaseBias'       'gradCap'   'phase bias b2 = 0 (paper: +1 on theta_i, so every candidate starts rotated by atan(1) = 45 deg)' [cap, {'phaseBias', 'zero'}]
'cosSin'            'gradCap'   'phase gates see [cos; sin] of angle(c) instead of atan2 (no 2 pi jump; still 1/|c| in the derivative); Q is 4H x 2H' [cap, {'feedback', 'cosSin'}]
'complexH'          'gradCap'   'phase gates see the complex hidden state o tanh(|c|) e^{i angle(c)} (bounded, smooth at c = 0, no 1/|c|); Q is 4H x 2H' [cap, {'feedback', 'complexH'}]
'lowRec'            'gradCap'   'U and Q drawn at 0.5x scale (recurrent gain over 500 steps)' [cap, {'recScale', 0.5}]
% ideal complex LSTM (phaseAwareLSTMIdeal, IDEAL_LAYER.md); each later one = ideal + one change
'ideal'             '-'         'ideal complex LSTM: real gates, bounded complex candidate and hidden, rotation omega + pi tanh, decaying coupled memory, [Re h; Im h] fed back, output |h|' cap
'idealReadout'      'ideal'     'output Wo [Re h; Im h] + bo instead of |h|' [cap, {'output', 'readout'}]
'idealNoOmega'      'ideal'     'omega = 0 (not trained): input-driven rotation only' [cap, {'omega', false}]
'idealNoNorm'       'ideal'     'memory input i u instead of (1 - f) i u' [cap, {'inputScale', 'plain'}]
'idealAtanRot'      'ideal'     'rotation omega + atan(A_r) (the paper''s angle map) instead of omega + pi tanh(A_r)' [cap, {'rotMap', 'atan'}]
};
list = struct('id', V(:,1), 'parent', V(:,2), 'change', V(:,3), 'set', V(:,4));
if nargin == 0
    out = list;
    return
end
k = find(strcmp({list.id}, id));
if isempty(k)
    error('phaseCorrVersions:unknown', 'Unknown version %s. Known: %s.', id, strjoin({list.id}, ', '));
end
if startsWith(id, 'ideal')
    out = phaseIdealConfig(list(k).set{:});
else
    out = phaseCorrConfig(list(k).set{:});
end
out.id = id;
end

function cfg = phaseCorrConfig(varargin)
% Defaults = the paper layer.
p = inputParser;
addParameter(p, 'gradCap',   Inf);
addParameter(p, 'forgetRow', 'i');
addParameter(p, 'phaseBias', 'unitForget');
addParameter(p, 'feedback',  'atan2');
addParameter(p, 'recScale',  1);
addParameter(p, 'clip',      5);
parse(p, varargin{:});
cfg = p.Results;
assert(any(strcmp(cfg.forgetRow, {'i','f'})));
assert(any(strcmp(cfg.phaseBias, {'unitForget','zero'})));
assert(any(strcmp(cfg.feedback,  {'atan2','cosSin','complexH'})));
end

function cfg = phaseIdealConfig(varargin)
% Defaults = the ideal layer (IDEAL_LAYER.md).
p = inputParser;
addParameter(p, 'gradCap',    Inf);
addParameter(p, 'output',     'mag');       % 'mag' | 'readout'
addParameter(p, 'omega',      true);
addParameter(p, 'inputScale', 'coupled');   % 'coupled' (1 - f) | 'plain'
addParameter(p, 'rotMap',     'piTanh');    % 'piTanh' | 'atan'
addParameter(p, 'fs',         250);         % CWL sampling rate (Hz)
parse(p, varargin{:});
cfg = p.Results;
assert(any(strcmp(cfg.output, {'mag','readout'})));
assert(any(strcmp(cfg.inputScale, {'coupled','plain'})));
assert(any(strcmp(cfg.rotMap, {'piTanh','atan'})));
end
