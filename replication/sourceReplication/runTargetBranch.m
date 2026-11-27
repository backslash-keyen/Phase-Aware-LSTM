function runTargetBranch(folds, varargin)
%RUNTARGETBRANCH  Paper target baselines (CWL, DeepRNN or EEGNet, Fp1-Fz) from one set of source nets.
%
%   runTargetBranch(1)        % fold 1 of every baseline, in order
%   runTargetBranch(1:10)     % all folds (resumes)
%
%   Run it from the repository root of a scratch copy whose pretrainedModels/
%   holds the source nets of the branch:
%     branch "hpc"  : pretrainedModels/cwl/cwl1/DeepRNN = extractHPCSourceNets.m output
%                     (the nets the paper's target runs started from)
%     branch "ours" : the nets from runSourceReplication (threefry, this machine)
%   Both branches need sourceDomainClassifier_N2_DeepRNN_<i>.mat
%   (full montage). The dual-channel transfer baseline (baseTransfer) is not
%   run: the released define_layerArchitecture.m has no case for it, and in the
%   paper's HPC results its trained nets are bit-identical to the no-transfer
%   nets, so the paper's baseTransfer row equals no transfer (see README).
%
%   runTargetBranch(1, 'ncats', 3, 'layers', {'deeprnn','basetransfer'}, ...
%                   'variant', {'dualSourceTransfer','c3PaperOptions'})
%                             % CWL 3-class (source nets in pretrainedModels/cwl/cwl2/DeepRNN)
%
%   runTargetBranch(1, 'typeNet', 'EEGNet')
%                             % EEGNet backbone (source nets in pretrainedModels/cwl/cwl1/EEGNet,
%                             % sourceDomainClassifier_N2_EEGNet_<i>.mat); layer 'deeprnn' is
%                             % then read as the backbone's own no-transfer row ('eegnet')
%
%   Order: no transfer, then the FFNN, LSTM and GRU projections. Phase-aware projections are not run here.

p = inputParser;
addParameter(p, 'rngType', 'threefry');
addParameter(p, 'layers',  {'deeprnn','ffnn','lstm','gru'});
addParameter(p, 'ncats',   2);
addParameter(p, 'typeNet', 'DeepRNN');
addParameter(p, 'domain',  'target-domain');
addParameter(p, 'variant', '');
parse(p, varargin{:});
o = p.Results;
for k = 1:numel(o.layers)
    layer = o.layers{k};
    if strcmpi(layer, 'deeprnn'), layer = lower(o.typeNet); end   % no-transfer row of the backbone
    id = sprintf('cwl_N%d_%s_%s_%s', o.ncats, o.domain, o.typeNet, layer);
    fprintf('\n===== %s, folds %s =====\n', id, mat2str(folds));
    rr = {'rngType', o.rngType};
    if ~isempty(o.variant), rr = [rr, {'variant', o.variant}]; end
    runReplication('ids', {id}, 'folds', folds, rr{:});
end
end
