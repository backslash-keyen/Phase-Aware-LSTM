function runPeepholeBranch(folds, varargin)
%RUNPEEPHOLEBRANCH  CWL C2 DeepRNN peephole-LSTM projection rows (Fp1-Fz target) for one set of source nets.
%
%   runPeepholeBranch(1)                                    % fold 1, every version
%   runPeepholeBranch(1:10, 'versions', {'gers2002'})       % all folds of one (resumes)
%
% Run from the repository root of a scratch copy whose pretrainedModels/ holds
% the branch's source nets (as for runTargetBranch / runCharmBranch):
%   branch "hpc"  : the nets the paper's target runs started from (extractHPCSourceNets.m)
%   branch "ours" : the threefry nets from runSourceReplication
% Each version is the LSTM projection cell (cwl_N2_target-domain_DeepRNN_lstm)
% with the variant ph_<version>, so it trains with the LSTM row's options and
% is stored as cwl_N2_target-domain_DeepRNN_lstm__ph_<version>.
p = inputParser;
addParameter(p, 'rngType',  'threefry');
addParameter(p, 'versions', {});
parse(p, varargin{:});
o = p.Results;
addpath(fullfile(fileparts(mfilename('fullpath')), 'common'));
vs = o.versions;
if isempty(vs), vs = {peepholeVersions().id}; end
id = 'cwl_N2_target-domain_DeepRNN_lstm';
for k = 1:numel(vs)
    fprintf('\n===== %s__ph_%s, folds %s =====\n', id, vs{k}, mat2str(folds));
    runReplication('ids', {id}, 'folds', folds, 'rngType', o.rngType, 'variant', ['ph_' vs{k}]);
end
end
