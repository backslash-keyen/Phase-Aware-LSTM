function out = assocVersions(id)
%ASSOCVERSIONS  Registry of the Associative LSTM projection versions.
%
%   list = assocVersions()           struct array: id, parent, change, set
%   cfg  = assocVersions('paper')    configuration for associativeLSTMLayer
%
% 'paper' is Danihelka et al. (2016), "Associative Long Short-Term Memory",
% ICML, Sec. 5, Eqs. 16-30, as written (see replication/assocLSTM/README.md
% for the choices the paper leaves open). Every other version is its parent
% with the configuration fields in 'set' changed, nothing else.
V = {
% id              parent    change                                                                                   set
'paper'          '-'       'Danihelka 2016 Eqs. 16-30: 61 complex cells (Nh = 122), 4 copies, W_hu learned, forget bias 0, keys bound(), linear readout 122 -> 61' {}
'forgetBias1'    'paper'   'forget-gate bias +1 (Gers 2000; the paper biases only its LSTM baseline)'                   {'forgetBias', 1}
'noWhu'          'paper'   'W_hu = 0: the update u is an embedding of x_t only (the paper''s choice on 4 of its 6 tasks)' {'whu', false}
'copies1'        'paper'   'N_copies = 1'                                                                          {'numCopies', 1}
'copies8'        'paper'   'N_copies = 8'                                                                          {'numCopies', 8}
'unitKeys'       'paper'   'keys r_i, r_o normalised to modulus 1 (the condition under which Eq. 8 holds) instead of bound() (modulus <= 1)' {'keys', 'unit'}
'realOut'        'paper'   'no readout: the 61 real parts Re h feed the source net'                                 {'output', 'real'}
'modOut'         'paper'   'no readout: the 61 moduli |h| feed the source net (as the phase-aware layer)'           {'output', 'mod'}
% literature-grounded version (README, "Ideal version"); idealNoOmega = ideal minus one component
'ideal'          '-'       'unit-modulus keys, forget bias +1, memory rotated each step by a learned per-cell frequency omega (1-40 Hz init), readout as paper' {'keys', 'unit', 'forgetBias', 1, 'omega', true}
'idealNoOmega'   'ideal'   'omega = 0 (not trained): no memory rotation'                                           {'keys', 'unit', 'forgetBias', 1}
};
list = struct('id', V(:,1), 'parent', V(:,2), 'change', V(:,3), 'set', V(:,4));
if nargin == 0
    out = list;
    return
end
k = find(strcmp({list.id}, id));
if isempty(k)
    error('assocVersions:unknown', 'Unknown Associative LSTM version %s. Known: %s.', id, strjoin({list.id}, ', '));
end
out = assocConfig(list(k).set{:});
out.id = id;
end

function cfg = assocConfig(varargin)
% Defaults = the paper (Danihelka et al. 2016, Sec. 5).
p = inputParser;
addParameter(p, 'numComplex', 61);       % complex cells; Nh = 2 x numComplex real values
addParameter(p, 'numCopies',  4);        % redundant copies of the memory (Sec. 3)
addParameter(p, 'whu',        true);     % Eq. 18 with W_hu (false: W_hu = 0)
addParameter(p, 'forgetBias', 0);        % initial bias of g_f
addParameter(p, 'keys',       'bound');  % 'bound' (Eqs. 24-25) | 'unit' (modulus 1)
addParameter(p, 'output',     'h');      % 'h' = [Re h; Im h] + linear readout | 'real' | 'mod'
addParameter(p, 'omega',      false);    % learned per-cell memory rotation e^{i omega}
addParameter(p, 'fs',         250);      % CWL sampling rate (Hz), for the omega init
addParameter(p, 'gradCap',    Inf);      % overflow-safe gradient cap (Inf = off, as the paper)
addParameter(p, 'permSeed',   1);        % seed of the constant random permutations P_s
parse(p, varargin{:});
cfg = p.Results;
assert(any(strcmp(cfg.keys,   {'bound','unit'})));
assert(any(strcmp(cfg.output, {'h','real','mod'})));
assert(cfg.numCopies >= 1 && cfg.numCopies == round(cfg.numCopies));
end
