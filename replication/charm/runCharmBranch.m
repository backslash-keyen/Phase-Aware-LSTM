function runCharmBranch(folds, varargin)
%RUNCHARMBRANCH  CWL C2 DeepRNN CHARM rows (Fp1-Fz target) for one set of source nets.
%
%   runCharmBranch(1)                                  % fold 1, every CHARM variant
%   runCharmBranch(1:10, 'variants', {'charmBase'})    % all folds of one (resumes)
%
% Run from the repository root of a scratch copy whose pretrainedModels/ holds
% the branch's source nets (pretrainedModels/cwl/cwl1/DeepRNN/
% sourceDomainClassifier_N2_DeepRNN_<i>.mat), as for runTargetBranch:
%   branch "hpc"  : the nets the paper's target runs started from (extractHPCSourceNets.m)
%   branch "ours" : the threefry nets from runSourceReplication (bit-identical to
%                   the Nov 2025 Windows nets)
%
% Variants (replication/charm/README.md):
%   charmBase           the Nov 2025 CHARM, ported unchanged: the replication row
%   charm3Conv          three-layer mconv
%   charmInstNorm       instance normalisation before CHARM
%   charmSourceTrained  CHARM + DeepRNN trained on the 61-channel source folds,
%                       then fine-tuned on the 2 target channels. It does not use
%                       the branch's source nets, so it runs once (pass it only
%                       in one branch); its source run comes first and saves to
%                       pretrainedModels_charm/, never to pretrainedModels/.
p = inputParser;
addParameter(p, 'rngType',  'threefry');
addParameter(p, 'variants', {'charmBase','charm3Conv','charmInstNorm','charmSourceTrained'});
parse(p, varargin{:});
o = p.Results;
tgt = 'cwl_N2_target-domain_DeepRNN_charm';
src = 'cwl_N2_source-domain_DeepRNN_charm';
for k = 1:numel(o.variants)
    v = o.variants{k};
    if strcmp(v, 'charmSourceTrained')
        fprintf('\n===== %s__%s (source), folds %s =====\n', src, v, mat2str(folds));
        runReplication('ids', {src}, 'folds', folds, 'rngType', o.rngType, 'variant', v);
    end
    fprintf('\n===== %s__%s, folds %s =====\n', tgt, v, mat2str(folds));
    runReplication('ids', {tgt}, 'folds', folds, 'rngType', o.rngType, 'variant', v);
end
end
