function T = comparePaperPhaseNets(foldDir, paperFile)
%COMPAREPAPERPHASENETS  Our phase-lstm folds against the paper's HPC nets, fold by fold.
%
%   T = comparePaperPhaseNets(foldDir, paperFile)
%     foldDir   cvalresults/CWL/DeepRNN/folds/<experiment id> of one of our runs
%     paperFile e.g. L:\plstm\HPC\PhaseAwareLSTM\cvalresults\CWL\DeepRNN\
%               cval_DeepRNN_target-domain_2_phase-lstm-notrig_251105_0748.mat
%               (read only)
%
% Run from the repository root (the paper nets need phaseAwareLSTM_notrig,
% which is added from replication/variants/notrigPhaseLSTM). Per fold:
% val/test accuracy of both, whether each phase layer is still at its
% initial biases (b1 and b2 exactly 0/1, i.e. the layer never trained), and
% the cosine between our and the paper's phase weights. For the frozen
% replication ('paper' / pc_port) and a paper fold that also never trained
% (4, 6, 10), cosine 1 means our initialisation equals the paper's.
root = pwd;
addpath(fullfile(root, 'replication', 'variants', 'notrigPhaseLSTM'));
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'common'));
P = load(paperFile, 'trainedModels', 'evals');
n = numel(P.trainedModels);
fold = (1:n)'; oursVal = nan(n,1); oursTest = oursVal; paperVal = oursVal; paperTest = oursVal;
oursFrozen = false(n,1); paperFrozen = oursFrozen; cosW1 = oursVal; cosW2 = oursVal; cosU = oursVal; cosQ = oursVal;
for i = 1:n
    paperVal(i)  = 100 * P.evals{i}.accuracyVal;
    paperTest(i) = 100 * P.evals{i}.accuracyTest;
    Lp = phaseLayer(P.trainedModels{i});
    paperFrozen(i) = atInit(Lp);
    f = fullfile(foldDir, sprintf('fold%d.mat', i));
    if ~isfile(f), continue; end
    S = load(f, 'model', 'evaluation');
    oursVal(i)  = 100 * S.evaluation.accuracyVal;
    oursTest(i) = 100 * S.evaluation.accuracyTest;
    Lo = phaseLayer(S.model);
    oursFrozen(i) = atInit(Lo);
    cosW1(i) = cosSim(Lo.W1, Lp.W1); cosW2(i) = cosSim(Lo.W2, Lp.W2);
    cosU(i)  = cosSim(Lo.U, Lp.U);
    q = extract(Lo.Q); cosQ(i) = cosSim(q(:, 1:size(Lp.Q, 2)), Lp.Q);
end
T = table(fold, oursVal, oursTest, paperVal, paperTest, oursFrozen, paperFrozen, cosW1, cosW2, cosU, cosQ);
disp(T);
fprintf('mean (folds run)   ours val %.2f test %.2f   paper val %.2f test %.2f\n', ...
    mean(oursVal, 'omitnan'), mean(oursTest, 'omitnan'), ...
    mean(paperVal(~isnan(oursVal))), mean(paperTest(~isnan(oursVal))));
end

function L = phaseLayer(net)
L = net.Layers(strcmp({net.Layers.Name}, 'phaseLSTM'));
end

function tf = atInit(L)
b = [extract(L.b1); extract(L.b2)];
tf = all(b == 0 | b == 1);
end

function c = cosSim(a, b)
a = double(extract(a)); b = double(extract(b));
c = (a(:)' * b(:)) / max(norm(a(:)) * norm(b(:)), eps);
end

function v = extract(v)
if isa(v, 'dlarray'), v = extractdata(v); end
v = gather(v);
end
