function cap = phaseGradCap()
%PHASEGRADCAP  (replication variant 'gradCap', opt-in)
% On the path only while runReplication(..., 'variant', {..., 'gradCap'}) runs.
% The fast phase-aware layers' backward() then rescales each learnable
% gradient to L2 norm <= cap in double (capPhaseGrads), so trainNetwork's
% single-precision l2norm clip no longer overflows to a zero update.
% cap = 1.25 = GradientThreshold of the tuned CWL C2 PhaseAware options, so the
% result is exactly the per-learnable l2norm clip those options ask for.
cap = 1.25;
end
