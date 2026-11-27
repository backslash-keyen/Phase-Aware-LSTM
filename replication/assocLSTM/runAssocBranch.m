function runAssocBranch(folds, varargin)
%RUNASSOCBRANCH  CWL C2 DeepRNN Associative LSTM rows (Fp1-Fz target) for one set of source nets.
%
%   runAssocBranch(1)                                   % fold 1, every version
%   runAssocBranch(1:10, 'versions', {'paper','ideal'}) % 10 folds (resumes)
%
% Run from the root of a run tree whose pretrainedModels/ holds the branch's
% source nets (pretrainedModels/cwl/cwl1/DeepRNN/sourceDomainClassifier_N2_DeepRNN_<i>.mat):
%   branch "hpc"  : the nets the paper's target runs started from (extractHPCSourceNets.m)
%   branch "ours" : the threefry nets from runSourceReplication
% Each version runs as runReplication variant al_<version> and writes
% cvalresults/.../cwl_N2_target-domain_DeepRNN_assoc-lstm__al_<version>/.
p = inputParser;
addParameter(p, 'rngType',  'threefry');
addParameter(p, 'versions', {assocVersions().id});
parse(p, varargin{:});
o = p.Results;
id = 'cwl_N2_target-domain_DeepRNN_assoc-lstm';
for k = 1:numel(o.versions)
    v = ['al_' o.versions{k}];
    fprintf('\n===== %s__%s, folds %s =====\n', id, v, mat2str(folds));
    runReplication('ids', {id}, 'folds', folds, 'rngType', o.rngType, 'variant', v);
end
end
