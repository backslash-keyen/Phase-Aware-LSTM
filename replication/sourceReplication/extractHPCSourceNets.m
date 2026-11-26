function files = extractHPCSourceNets(cvalFile, outDir, stem)
%EXTRACTHPCSOURCENETS  Save the source nets stored in an HPC cval_*.mat as loadable source files.
%
%   extractHPCSourceNets(cvalFile, outDir, stem)
%
%   The paper's target runs (Nov 2025, HPC) started from the source nets saved
%   inside the HPC source-domain result file (trainedModels{i}.trainedNet), not
%   from the pretrainedModels files on the authors' PC. This writes them in the
%   format get_pretrainedLayers.m loads: variables net, layers, options in
%   <outDir>/<stem>_<i>.mat, e.g. stem 'sourceDomainClassifier_N2_DeepRNN'
%   (full montage) or 'sourceDomain2Classifier_N2_DeepRNN' (dual channel).
%   cvalFile is only read.
%
%   Example (CWL C2 DeepRNN, the paper's source nets):
%     d = 'L:\plstm\HPC\PhaseAwareLSTM\cvalresults\CWL\DeepRNN';
%     out = fullfile('pretrainedModels_hpc','cwl','cwl1','DeepRNN');
%     extractHPCSourceNets(fullfile(d,'cval_DeepRNN_source-domain_2_DeepRNN_251104_1925.mat'), out, 'sourceDomainClassifier_N2_DeepRNN');
%     extractHPCSourceNets(fullfile(d,'cval_DeepRNN_source-domain-2_2_DeepRNN_251105_2234.mat'), out, 'sourceDomain2Classifier_N2_DeepRNN');

if ~exist(outDir, 'dir'), mkdir(outDir); end
S = load(cvalFile, 'trainedModels', 'evals');
tm = S.trainedModels;
files = cell(numel(tm), 1);
fprintf('[extractHPCSourceNets] %s: %d folds\n', cvalFile, numel(tm));
for i = 1:numel(tm)
    net = tm{i}.trainedNet; layers = tm{i}.layers; options = tm{i}.options; %#ok<NASGU>
    files{i} = fullfile(outDir, sprintf('%s_%d.mat', stem, i));
    save(files{i}, 'net', 'layers', 'options', '-v7.3');
    fprintf('  fold %2d: %s | %s, %d layers, input %s | HPC val %.2f test %.2f\n', i, files{i}, ...
        class(net), numel(net.Layers), mat2str(net.Layers(1).InputSize), ...
        100*S.evals{i}.accuracyVal, 100*S.evals{i}.accuracyTest);
end
end
