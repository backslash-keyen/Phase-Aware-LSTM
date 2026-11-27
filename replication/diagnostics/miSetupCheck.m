function T = miSetupCheck(projectLoc, hpcCvalDir)
%MISETUPCHECK  Check that an MI (BCI IV 2a -> 2b) run tree is ready. Trains nothing.
%
%   T = miSetupCheck('F:\Phase-Aware-LSTM-mi-runs\_mi_hpc', ...
%                    'L:\plstm\HPC\PhaseAwareLSTM\cvalresults\MI')
%
% For every fold file under <projectLoc>\datasets\folds\MI\MI_C2 it loads the
% fold through the released perfold_dataset.m and reports split sizes, sample
% shape and label categories. It prints the training options the
% 'miHPCOptions' variant gives for each net x domain (MaxEpochs, LR, batch).
% If <projectLoc>\pretrainedModels\mi\mi1\<net> holds source nets, each one is
% evaluated on its source fold (perfold_test) and compared with the val/test
% accuracy stored in the HPC cval file (hpcCvalDir, optional): an exact match
% shows the extracted net, the fold files and the evaluation code agree.
% Returns one row per check.
if nargin < 2, hpcCvalDir = ''; end
repDir = fileparts(fileparts(mfilename('fullpath')));
oldDir = cd(projectLoc); back = onCleanup(@() cd(oldDir));
addpath(genpath(fullfile(projectLoc, 'utils')));
rows = {};

% ---- folds -------------------------------------------------------------
doms = {'source-domain', 'target-domain'};
for d = 1:numel(doms)
    for i = 1:5
        cfg = configureExperiment('datasetType', 'MI', 'ncats', 2, 'domain', doms{d}, 'verbose', false);
        cfg.projectLoc = '';
        try
            f = perfold_dataset(cfg, i);
            n = [size(f.xtrain,3) size(f.xval,3) size(f.xtest,3)];
            cats = strjoin(categories(f.ytest), ',');
            msg = sprintf('train/val/test %d/%d/%d, x %s, classes %s, test counts %s', n, ...
                mat2str(size(f.xtrain,[1 2])), cats, mat2str(countcats(f.ytest)'));
            ok = all(n > 0) && size(f.xtrain,2) == 1125 && numel(categories(f.ytest)) == 2;
        catch ME
            ok = false; msg = ME.message;
        end
        rows(end+1,:) = {'fold', sprintf('%s %d', doms{d}, i), ok, msg}; %#ok<AGROW>
        fprintf('[miSetupCheck] fold %-13s %d: %d | %s\n', doms{d}, i, ok, msg);
    end
end

% ---- options (variant miHPCOptions) -------------------------------------
vdir = fullfile(repDir, 'variants', 'miHPCOptions');
addpath(vdir, '-begin'); rmv = onCleanup(@() rmpath(vdir));
combos = {'DeepRNN','source-domain','DeepRNN'; 'DeepRNN','target-domain','ffnn'; ...
          'DeepRNN','target-domain','phase-lstm'; 'EEGNet','source-domain','EEGNet'; ...
          'EEGNet','target-domain','lstm'; 'transformer','source-domain','transformer'; ...
          'transformer','target-domain','lstm'};
for c = 1:size(combos,1)
    cfg = configureExperiment('datasetType', 'MI', 'ncats', 2, 'typeNet', combos{c,1}, ...
        'domain', combos{c,2}, 'layerName', combos{c,3}, 'verbose', false);
    try
        o = get_trainingOptions(cfg);
        msg = sprintf('MaxEpochs %d, LR %g, MB %d, L2 %g, %s', o.MaxEpochs, o.InitialLearnRate, ...
            o.MiniBatchSize, o.L2Regularization, o.OutputNetwork);
        ok = true;
    catch ME
        ok = false; msg = ME.message;
    end
    rows(end+1,:) = {'options', strjoin(combos(c,:), ' '), ok, msg}; %#ok<AGROW>
    fprintf('[miSetupCheck] options %-40s %d | %s\n', strjoin(combos(c,:), ' '), ok, msg);
end
clear rmv

% ---- source nets ---------------------------------------------------------
nets = {'DeepRNN', 'EEGNet', 'transformer'};
for n = 1:numel(nets)
    dirN = fullfile(projectLoc, 'pretrainedModels', 'mi', 'mi1', nets{n});
    ref = [];
    if ~isempty(hpcCvalDir)
        hf = dir(fullfile(hpcCvalDir, nets{n}, sprintf('cval_%s_source-domain_2_%s_*.mat', nets{n}, nets{n})));
        if ~isempty(hf), ref = load(fullfile(hf(1).folder, hf(1).name), 'evals'); end
    end
    for i = 1:5
        p = fullfile(dirN, sprintf('sourceDomainClassifier_N2_%s_%d.mat', nets{n}, i));
        if ~isfile(p), continue; end
        cfg = configureExperiment('datasetType', 'MI', 'ncats', 2, 'domain', 'source-domain', 'verbose', false);
        cfg.projectLoc = '';
        try
            S = load(p, 'net');
            data = perfold_dataset(cfg, i);
            ev = perfold_test(struct('trainedNet', S.net, 'trainingTime', 0), data);
            msg = sprintf('val %.2f test %.2f', 100*ev.accuracyVal, 100*ev.accuracyTest);
            ok = true;
            if ~isempty(ref)
                hv = 100*ref.evals{i}.accuracyVal; ht = 100*ref.evals{i}.accuracyTest;
                ok = abs(hv - 100*ev.accuracyVal) < 1e-6 && abs(ht - 100*ev.accuracyTest) < 1e-6;
                msg = sprintf('%s | HPC val %.2f test %.2f', msg, hv, ht);
            end
        catch ME
            ok = false; msg = ME.message;
        end
        rows(end+1,:) = {'sourceNet', sprintf('%s %d', nets{n}, i), ok, msg}; %#ok<AGROW>
        fprintf('[miSetupCheck] source net %-12s %d: %d | %s\n', nets{n}, i, ok, msg);
    end
end
T = cell2table(rows, 'VariableNames', {'check', 'item', 'ok', 'detail'});
end
