function s = phasedVersions(name)
%PHASEDVERSIONS  Settings of every Phased LSTM version (replication/variants/phased/<name>).
%
%   s = phasedVersions('phasedNeil')   % one version
%   L = phasedVersions()               % all, as a struct array, in run order
%
% Replication
%   phasedBase       the Nov 2025 layer (phasedLSTM.m) and builder behind the
%                    paper's Phased-LSTM rows, ported unchanged (autodiff)
% Corrections: each is phasedGate plus ONE change towards Neil et al. 2016
%   phasedGate       time gate of Neil Eq. 6 (rise 2*phi/r_on to 1 at r_on/2, fall
%                    2 - 2*phi/r_on to 0 at r_on). The Nov 2025 gate rises phi/r_on
%                    to 0.1 at 0.1*r_on, jumps to 2.9 and falls 3 - phi/r_on to 2 at
%                    r_on, so k_t > 1 on 90 % of the open phase (an extrapolating
%                    update). Everything else as phasedBase
%   phasedGateRon    + r_on = 0.05, the value Neil fix in every task (Nov 2025: 0.01)
%   phasedGateLeak   + leak alpha = 1e-3 while training and 0 at test (Neil Sec. 3;
%                    Nov 2025: 0, and its setTrainMode/setTestMode are never called)
%   phasedGateTau    + tau ~ exp(U(0,3)) (Neil Sec. 3.1, and what the Nov 2025
%                    comment says; its code draws exp(U(0,1)))
%   phasedGateLearn  + gradients reach tau and s (Neil learn both). In Nov 2025
%                    computeTimeGate strips them with extractdata, so the only
%                    thing that moves tau and s is L2 weight decay
%   phasedGateNoDecay + no L2 weight decay on tau and s, so they stay at their
%                    init (Neil fix them or learn them; they never decay them)
%   phasedGatePeep   + peephole weights w_ci, w_cf, w_co (Neil Eqs. 1, 2, 4, the
%                    Graves 2013 LSTM; Lasagne's LSTM, which Neil kept, has them on)
% Faithful to the paper
%   phasedNeil       all of the above: Neil Eqs. 1-10 with peepholes, r_on 0.05
%                    fixed, alpha 1e-3 / 0, tau ~ exp(U(0,3)) and s ~ U(0, tau) both
%                    learned and not decayed, unit forget bias
% Ideal (see replication/phased/README.md for the reasoning and sources)
%   phasedIdeal      phasedNeil with tau ~ log-uniform on [40, 500] samples
%                    (0.16 to 2 s at 250 Hz: every open phase is >= 2 samples long,
%                    periods spread exponentially as in the Clockwork RNN, long
%                    periods as Neil Fig. 4b favours for 500-step sequences) and
%                    r_on learned per unit from 0.05 (Neil Sec. 3: all three timing
%                    parameters can be learned)
%   phasedIdealWide  phasedIdeal with r_on starting at 0.5: each unit updates on half
%                    its cycle, because sparse updates cost accuracy on regularly
%                    sampled data (Chen et al. 2019, LSTM vs Phased LSTM gait study)
%
% Fields
%   name, port (true = the ported Nov 2025 layer), gate ('nov2025' | 'neil'),
%   ron (init), ronLearn, leakTrain, leakTest, tauLog ([lo hi]: tau = exp(U(lo,hi))),
%   timingGrad, timingL2 (L2 factor of tau and s), peephole, summary
base = struct('name', '', 'port', false, 'gate', 'nov2025', 'ron', 0.01, 'ronLearn', false, ...
    'leakTrain', 0, 'leakTest', 0, 'tauLog', [0 1], 'timingGrad', false, 'timingL2', 1, ...
    'peephole', false, 'summary', '');
L = base([]);
L(end+1) = mk(base, 'phasedBase', 'Nov 2025 layer and builder, ported unchanged', 'port', true);
gate = mk(base, 'phasedGate', 'Nov 2025 layer with the time gate of Neil Eq. 6', 'gate', 'neil');
L(end+1) = gate;
L(end+1) = mk(gate, 'phasedGateRon',     'phasedGate + r_on 0.05',                         'ron', 0.05);
L(end+1) = mk(gate, 'phasedGateLeak',    'phasedGate + leak 1e-3 in training, 0 at test',  'leakTrain', 1e-3);
L(end+1) = mk(gate, 'phasedGateTau',     'phasedGate + tau ~ exp(U(0,3))',                 'tauLog', [0 3]);
L(end+1) = mk(gate, 'phasedGateLearn',   'phasedGate + gradients to tau and s',            'timingGrad', true);
L(end+1) = mk(gate, 'phasedGateNoDecay', 'phasedGate + no L2 decay on tau and s',          'timingL2', 0);
L(end+1) = mk(gate, 'phasedGatePeep',    'phasedGate + peephole connections',              'peephole', true);
neil = mk(gate, 'phasedNeil', 'Neil et al. 2016 as written (all corrections)', ...
    'ron', 0.05, 'leakTrain', 1e-3, 'tauLog', [0 3], 'timingGrad', true, 'timingL2', 0, 'peephole', true);
L(end+1) = neil;
ideal = mk(neil, 'phasedIdeal', 'phasedNeil + tau log-uniform 40-500 samples + learned r_on', ...
    'tauLog', log([40 500]), 'ronLearn', true);
L(end+1) = ideal;
L(end+1) = mk(ideal, 'phasedIdealWide', 'phasedIdeal with r_on starting at 0.5', 'ron', 0.5);
if nargin == 0 || isempty(name), s = L; return; end
k = find(strcmp({L.name}, name));
if isempty(k)
    error('phased:unknownVersion', 'Unknown Phased LSTM version %s. Known: %s.', name, strjoin({L.name}, ', '));
end
s = L(k);
end

function s = mk(s, name, summary, varargin)
s.name = name; s.summary = summary;
for i = 1:2:numel(varargin), s.(varargin{i}) = varargin{i+1}; end
end
