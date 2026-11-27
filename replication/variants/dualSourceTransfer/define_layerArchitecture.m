function layers = define_layerArchitecture(model, config)
%DEFINE_LAYERARCHITECTURE  (variant 'dualSourceTransfer', beyond the released code)
%
% The released code has no builder for the dual-channel transfer baseline
% ('baseTransfer') and cannot train the dual-channel source ('source-domain-2').
% In the paper's own results that row is a copy of no transfer: the trained nets
% are bit-identical to the no-transfer nets, so the dual-channel source nets were
% loaded but never used. This variant makes the row a real transfer:
%   source-domain-2       the DeepRNN source model on the Fp1-Fz source folds
%                         (dualchannelsource/), built as in source-domain
%   target-domain +       start from the loaded dual-channel source net
%   baseTransfer          (get_pretrainedLayers.m, sourceDomain2Classifier_*)
%                         and fine-tune all of it on the target folds
% Everything else goes to the released function unchanged.

if strcmpi(config.domain, 'source-domain-2')
    c = config; c.domain = 'source-domain';
    layers = callOriginal_dualSourceTransfer('define_layerArchitecture', model, c);
elseif strcmpi(config.layerName, 'baseTransfer')
    assert(isfield(model, 'pretrainedNet') && ~isempty(model.pretrainedNet), ...
        'dualSourceTransfer:noSource', 'baseTransfer needs the dual-channel source net (sourceDomain2Classifier_*).');
    layers = model.pretrainedNet.Layers;
    assert(isequal(layers(1).InputSize, model.nchans), 'dualSourceTransfer:channels', ...
        'The dual-channel source net takes %s channels; the target data has %d.', mat2str(layers(1).InputSize), model.nchans);
    fprintf('[dualSourceTransfer] fine-tuning the dual-channel source net (%d layers) on the target\n', numel(layers));
else
    layers = callOriginal_dualSourceTransfer('define_layerArchitecture', model, config);
end
end
