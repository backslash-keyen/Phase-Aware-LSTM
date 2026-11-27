function runPhasedBranch(folds, varargin)
%RUNPHASEDBRANCH  CWL C2 DeepRNN Phased LSTM rows (Fp1-Fz target) for one set of source nets.
%
%   runPhasedBranch(1)                                    % fold 1, every version
%   runPhasedBranch(1:10, 'variants', {'phasedBase'})     % all folds of one (resumes)
%
% Run from the repository root of a scratch copy whose pretrainedModels/ holds the
% branch's source nets (pretrainedModels/cwl/cwl1/DeepRNN/
% sourceDomainClassifier_N2_DeepRNN_<i>.mat), as for runTargetBranch and runCharmBranch:
%   branch "hpc"  : the nets the paper's target runs started from (extractHPCSourceNets.m)
%   branch "ours" : the threefry nets from runSourceReplication (bit-identical to
%                   the Nov 2025 Windows nets)
% Versions: replication/variants/phased/common/phasedVersions.m and
% replication/phased/README.md. Each writes
% cvalresults/cwl/DeepRNN/folds/cwl_N2_target-domain_DeepRNN_phased-lstm__<version>/.
p = inputParser;
addParameter(p, 'rngType',  'threefry');
addParameter(p, 'variants', {});
parse(p, varargin{:});
o = p.Results;
if isempty(o.variants)
    addpath(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'variants', 'phased', 'common'));
    o.variants = {phasedVersions().name};
end
tgt = 'cwl_N2_target-domain_DeepRNN_phased-lstm';
for k = 1:numel(o.variants)
    v = o.variants{k};
    fprintf('\n===== %s__%s, folds %s =====\n', tgt, v, mat2str(folds));
    runReplication('ids', {tgt}, 'folds', folds, 'rngType', o.rngType, 'variant', v);
end
end
