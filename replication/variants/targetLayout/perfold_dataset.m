function dataFold = perfold_dataset(config, i)
%PERFOLD_DATASET  (replication variants 'tl_<layout>', opt-in, extension study)
% Shadows utils/crossvalidation/perfold_dataset.m ONLY when a tl_<layout>
% variant puts this folder first on the path. The original is not changed.
%
% For CWL 'target-domain' runs it loads the target folds of another
% two-channel layout, datasets/folds/CWL/NEC21_C<n>/targetlayout_<layout>/
% target_fold<i>.mat, built from the full-montage source folds by
% replication/channels/buildTargetLayoutFolds.m (same trials, same split,
% only the two channel rows differ). Everything else (domain name, options,
% pretrained source model, projection builders) is the original
% target-domain path. Every other dataset and domain calls the original.
if strcmpi(config.datasetType, 'cwl') && strcmpi(config.domain, 'target-domain')
    layout = activeTargetLayout();
    folder = fullfile(config.projectLoc, 'datasets', 'folds', 'CWL', ...
                      sprintf('NEC21_C%d', config.ncats), ['targetlayout_' layout]);
    fullPath = fullfile(folder, sprintf('target_fold%d.mat', i));
    if ~isfile(fullPath)
        error('replication:targetLayout', 'Target-layout fold missing: %s (run buildTargetLayoutFolds first).', fullPath);
    end
    fprintf('[replication] target layout %s: %s\n', layout, fullPath);
    dataFold = load(fullPath, 'xtrain', 'xval', 'xtest', 'ytrain', 'yval', 'ytest');
    dataFold = makeDatastore(dataFold);
else
    dataFold = callOriginal_targetLayout('perfold_dataset', config, i);
end
end
