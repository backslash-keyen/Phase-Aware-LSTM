function hyperParameters = get_trainingOptions(config)
%GET_TRAININGOPTIONS  (variant 'pv_ps_optsDefault' only: pipeline study, options factor)
% Undoes the tuned-options shadow in replication/versions/fast/common for this
% one cell: phase-lstm trains with the released routing (DeepRNN options,
% LR 0.0029, GradientThreshold 82.4, ...), everything else as in cell 'base'.
here   = fileparts(mfilename('fullpath'));
common = fullfile(fileparts(here), 'common');
rmpath(here); rmpath(common);   % restored with here first, as runReplication left them
restore = onCleanup(@() addpath(here, common, '-begin'));
hyperParameters = get_trainingOptions(config);
fprintf('[ps_optsDefault] phase-lstm uses the released get_trainingOptions (no tuned options)\n');
end
