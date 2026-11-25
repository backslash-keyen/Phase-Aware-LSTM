function folder = dataFolder(projectLoc, datasetType, ncats, domain)
%DATAFOLDER  Folder that perfold_dataset.m reads for this experiment.
% Mirrors utils/crossvalidation/perfold_dataset.m exactly.
switch lower(datasetType)
    case 'cwl', dsFolder = fullfile('CWL', sprintf('NEC21_C%d', ncats));
    case 'mi',  dsFolder = fullfile('MI',  sprintf('MI_C%d', ncats));
    otherwise,  error('Unknown datasetType %s', datasetType);
end
switch lower(domain)
    case 'source-domain',     sub = 'fullchannels';
    case 'source-domain-2',   sub = 'dualchannelsource';
    case 'target-domain',     sub = 'targetdomain1';
    case 'target-domain-2',   sub = 'targetdomain2';
    case 'temporalmasking-1', sub = 'temporalMasking_Fp1_Fz';
    case 'temporalmasking-2', sub = 'temporalMasking_F3_F7';
    otherwise, error('Unknown domain %s', domain);
end
folder = fullfile(projectLoc, 'datasets', 'folds', dsFolder, sub);
end
