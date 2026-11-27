function hyperParameters = get_trainingOptions(config)
%GET_TRAININGOPTIONS  (variant 'dualSourceTransfer') source-domain-2 uses the source-domain options.
% The released function accepts only source-domain and target-domain. The dual-channel
% source models of the paper (Nov 2025) were trained with the same options as the full
% montage source (LR 0.0029, mini-batch 197, 30 epochs); baseTransfer uses the
% unchanged target-domain options of the other baselines.
c = config;
if strcmpi(c.domain, 'source-domain-2'), c.domain = 'source-domain'; end
hyperParameters = callOriginal_dualSourceTransfer('get_trainingOptions', c);
end
