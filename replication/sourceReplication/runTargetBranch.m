function runTargetBranch(folds, varargin)
%RUNTARGETBRANCH  Paper target baselines (CWL C2, DeepRNN, Fp1-Fz) from one set of source nets.
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
%   Order: no transfer, then the FFNN, LSTM and GRU projections. Phase-aware projections are not run here.

p = inputParser;
addParameter(p, 'rngType', 'threefry');
addParameter(p, 'layers',  {'deeprnn','ffnn','lstm','gru'});
parse(p, varargin{:});
o = p.Results;
for k = 1:numel(o.layers)
    id = sprintf('cwl_N2_target-domain_DeepRNN_%s', o.layers{k});
    fprintf('\n===== %s, folds %s =====\n', id, mat2str(folds));
    runReplication('ids', {id}, 'folds', folds, 'rngType', o.rngType);
end
end
