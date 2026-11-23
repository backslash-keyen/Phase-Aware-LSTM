function layers = define_layerArchitecture(model, config)
%DEFINE_LAYERARCHITECTURE (replication shim, opt-in)
% Shadows utils/modelstuff/define_layerArchitecture.m only when
% runReplication(..., 'useShims', true) puts replication/shims on the path.
%
% Adds what the released code references but cannot build:
%   * domains source-domain-2, target-domain-2, temporalmasking-1/2 are
%     routed to the original source-domain / target-domain branches;
%   * layerName 'baseTransfer' (common-channel selection, loaded by
%     get_pretrainedLayers.m from sourceDomain2Classifier_*.mat) fine-tunes
%     the dual-channel source network as is.  ASSUMPTION: the released code
%     has no builder for this case; fine-tuning the whole pretrained graph
%     with the original training options is our reading of "transfer
%     learning with common channel selection".
cfg = config;
switch lower(config.domain)
    case {'source-domain','source-domain-2'}
        cfg.domain = 'source-domain';
    case {'target-domain','target-domain-2','temporalmasking-1','temporalmasking-2'}
        cfg.domain = 'target-domain';
end

if strcmpi(config.layerName, 'baseTransfer')
    layers = layerGraph(model.pretrainedNet);
    return
end

layers = callOriginal('define_layerArchitecture', model, cfg);
end
