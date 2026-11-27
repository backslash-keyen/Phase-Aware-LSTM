function out = peepholeVersions(id)
%PEEPHOLEVERSIONS  Registry of the peephole-LSTM projection versions.
%
%   cfg  = peepholeVersions('gers2002')   configuration of one version
%   list = peepholeVersions()             struct array: id, group, cfg, change
%
% 'gers2002' is the faithful version (Gers, Schraudolph & Schmidhuber 2002,
% JMLR 3:115-143). Each correction or variant is gers2002 with exactly one
% field changed. 'ideal' is the vanilla LSTM of Greff et al. (2017), which
% keeps the peepholes; 'idealCIFG' is ideal with one change.
% See replication/peephole/README.md for the reasons and sources.

base = struct('grad', 'truncated', 'gFun', 'identity', 'hFun', 'identity', ...
    'peep', true, 'outPeep', 'current', 'cifg', false, 'init', 'gers', 'forgetBias', NaN);

ideal = base;
ideal.grad = 'full'; ideal.gFun = 'tanh'; ideal.hFun = 'tanh'; ideal.init = 'glorot';

defs = {
  % id            group         base   field         value        what changes
  'gers2002',     'faithful',   base,  '',           [],          'the paper''s peephole LSTM: truncated gradient, g = h = identity, U[-0.1,0.1], biases i 0 / f -2 / o +2'
  'fullGrad',     'correction', base,  'grad',       'full',      'exact BPTT gradient instead of the truncated RTRL/BPTT (Graves & Schmidhuber 2005)'
  'squashIn',     'correction', base,  'gFun',       'tanh',      'cell input squashed by tanh (g of Hochreiter & Schmidhuber 1997)'
  'squashOut',    'correction', base,  'hFun',       'tanh',      'cell output squashed by tanh (h of the 1997 LSTM; Greff et al. 2017: removing it hurts)'
  'forgetPos',    'correction', base,  'forgetBias', 1,           'forget bias +1 instead of -2 (Gers et al. 2000; Jozefowicz et al. 2015)'
  'glorotInit',   'correction', base,  'init',       'glorot',    'lstmLayer initialisation (Glorot W, orthogonal R, unit-forget b, peepholes 0) instead of U[-0.1,0.1]'
  'outPeepPrev',  'variant',    base,  'outPeep',    'previous',  'output gate peeps c_{t-1} (one-phase update) instead of c_t'
  'noPeep',       'variant',    base,  'peep',       false,       'no peepholes: the paper''s traditional LSTM (its Sec. 2) under the same training rule'
  'ideal',        'ideal',      ideal, '',           [],          'vanilla LSTM of Greff et al. 2017: peepholes, full BPTT, g = h = tanh, forget bias +1, Glorot/orthogonal init, peepholes start at 0'
  'idealCIFG',    'ideal',      ideal, 'cifg',       true,        'ideal with coupled input-forget gate i = 1 - f (Greff et al. 2017: no loss, fewer weights)'
};

list = struct('id', {}, 'group', {}, 'cfg', {}, 'change', {});
for k = 1:size(defs, 1)
    cfg = defs{k, 3};
    if ~isempty(defs{k, 4}), cfg.(defs{k, 4}) = defs{k, 5}; end
    list(end+1) = struct('id', defs{k, 1}, 'group', defs{k, 2}, 'cfg', cfg, 'change', defs{k, 6}); %#ok<AGROW>
end
if nargin == 0
    out = list;
    return
end
hit = strcmp({list.id}, char(id));
if ~any(hit)
    error('peepholeVersions:unknown', 'Unknown peephole version %s. Known: %s.', char(id), strjoin({list.id}, ', '));
end
out = list(hit).cfg;
end
