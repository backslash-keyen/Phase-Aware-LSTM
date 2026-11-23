classdef configureExperiment
    %CONFIGUREEXPERIMENT
    % Configuration holder for projection transfer learning experiments.
    % - Auto-detects project root (MATLAB Project or file location)
    % - Sets up folder structure relative to project root
    % - Validates dataset & model names (case-insensitive)
    % - Accepts Name-Value overrides in the constructor
    %
    % Folder conventions (relative to projectLoc):
    %   datasets/folds/...
    %   cvalresults/<dataset>/<model>/
    %   pretrainedModels/
    %       cwl/cwl1/<model>/
    %       cwl/cwl2/<model>/
    %       mi/mi1/<model>/
    %       mi/mi3/<model>/
    %       er/<model>/

    properties
        % Paths
        projectLoc              % Root directory of the project
        datasetLoc              % Path to EEG datasets (folds, etc.)
        sourceModelLoc_CWL_1    % (kept for backwards-compat mapping)
        sourceModelLoc_CWL_2
        sourceModelLoc
        sourceModelLoc_MI_1
        sourceModelLoc_MI_3
        

        % Experiment parameters
        ncval           % # of CV folds
        domain          % 'source-domain' | 'target-domain' | 'transfer'
        typeNet         % 'DeepRNN' | 'EEGNet' | 'transformer' (validated)
        layerName       % Layer to extract features from
        experimentTag   % Free-form tag
        datasetType     % Canonical lower-case: 'cwl' | 'mi' | 'er'
        parTrain        % true/false
        ncats           % optional: number of classes (if you want to store it)

        % Lists for internal looping (allowed sets)
        sourceModels    % allowed model names (as stored)
        datasets        % allowed dataset names (as stored)

        % Output
        resultLoc       % Base path for saving cross-validation results

        verbose = true  % Console output during setup
    end

    methods
        function obj = configureExperiment(varargin)
            % CONSTRUCTOR — supports Name-Value overrides:
            %   configureExperiment('datasetType','MI','typeNet','EEGNet', 'parTrain',false, ...)

            % ---------- Project root (no hard-coded drive letters) ----------
            % obj.projectLoc = obj.detectProjectRoot();

            % ---------- Defaults ----------
            obj.datasetLoc   = fullfile(obj.projectLoc, 'datasets', 'folds');

            % Keep these lists as your "allowed" enumerations
            obj.sourceModels = {'EEGNet','DeepRNN','transformer'};
            obj.datasets     = {'CWL','MI'};

            % Parameters (before overrides)
            obj.ncval         = 10;
            obj.domain        = 'source-domain';
            obj.typeNet       = 'DeepRNN';
            obj.layerName     = 'DeepRNN';
            obj.experimentTag = 'baseline';
            obj.datasetType   = 'CWL';    % user-facing; canonicalized in validateDataset
            obj.parTrain      = true;
            obj.ncats         = [];

            % ---------- Apply Name-Value overrides ----------
            if ~isempty(varargin)
                if mod(numel(varargin),2) ~= 0
                    error('Name-Value inputs must be provided in pairs.');
                end
                for k = 1:2:numel(varargin)
                    name  = varargin{k};
                    value = varargin{k+1};
                    if ~ischar(name) && ~isstring(name)
                        error('Property name at position %d is not a string.', k);
                    end
                    name = char(name);
                    if ~isprop(obj, name)
                        error('Unknown property "%s".', name);
                    end
                    obj.(name) = value;
                end
            end

            % ---------- Validate and standardize dataset/model ----------
            obj = obj.validateDataset();   % sets datasetType -> canonical lower-case
            obj = obj.validateModel();     % normalizes typeNet to stored form

            % ---------- Setup directory structure ----------
            obj = obj.setupDirectories();
        end

        function obj = setupDirectories(obj)
            obj = obj.helper_resultDir();
            obj = obj.helper_sourceModelDir();

            % Back-compat "specific" source folders under the unified root
            interFolderName = 'pretrainedModels';
            obj.sourceModelLoc = fullfile(obj.projectLoc, interFolderName);

            % Canonical roots
            obj.sourceModelLoc_CWL_1 = fullfile(obj.sourceModelLoc, 'cwl', 'cwl1');
            obj.sourceModelLoc_CWL_2 = fullfile(obj.sourceModelLoc, 'cwl', 'cwl2');
            obj.sourceModelLoc_MI_1  = fullfile(obj.sourceModelLoc, 'mi',  'mi1');

            % NOTE: this was pointing to 'mi2' earlier — potential bug.
            % For public repo, keep naming consistent:
            obj.sourceModelLoc_MI_3  = fullfile(obj.sourceModelLoc, 'mi',  'mi3');
            

            % Ensure each root exists and has subfolders for every model type
            roots = { ...
                obj.sourceModelLoc_CWL_1, ...
                obj.sourceModelLoc_CWL_2, ...
                obj.sourceModelLoc_MI_1,  ...
                obj.sourceModelLoc_MI_3,  ...                
            };
            for r = 1:numel(roots)
                obj.ensureDirExists(roots{r});
                obj.ensureModelSubdirs(roots{r}, obj.sourceModels);
            end
        end

        function obj = helper_resultDir(obj)
            % Create result directories for cross-validation models
            baseDir = fullfile(obj.projectLoc, 'cvalresults');
            obj.ensureDirExists(baseDir);

            dsList = obj.canonicalDatasetNames(); % lower-case canonical tokens
            for i = 1:numel(dsList)
                datasetDir = fullfile(baseDir, dsList{i});
                obj.ensureDirExists(datasetDir);
                for j = 1:numel(obj.sourceModels)
                    modelDir = fullfile(datasetDir, obj.sourceModels{j});
                    obj.ensureDirExists(modelDir);
                end
            end

            obj.resultLoc = baseDir;
            if obj.verbose
                fprintf('[info] resultLoc = %s\n', obj.resultLoc);
            end
        end

        function obj = helper_sourceModelDir(obj)
            % Create dataset-level roots and model-type subdirs under pretrainedModels/
            modelRoot = fullfile(obj.projectLoc, 'pretrainedModels');
            obj.ensureDirExists(modelRoot);

            % Tree we want:
            %   pretrainedModels/
            %     cwl/cwl1/<models>
            %     cwl/cwl2/<models>
            %     mi/mi1/<models>
            %     mi/mi3/<models>            

            % Ensure 2-level roots exist
            cwlRoot = fullfile(modelRoot, 'cwl');
            miRoot  = fullfile(modelRoot, 'mi');            
            cellfun(@(p)obj.ensureDirExists(p), {cwlRoot, miRoot});

            % Specific leaves
            leaves = { ...
                fullfile(cwlRoot, 'cwl1'), ...
                fullfile(cwlRoot, 'cwl2'), ...
                fullfile(miRoot,  'mi1'),  ...
                fullfile(miRoot,  'mi3'),  ...                
            };
            for i = 1:numel(leaves)
                obj.ensureDirExists(leaves{i});
                obj.ensureModelSubdirs(leaves{i}, obj.sourceModels);
            end
        end

        function ensureModelSubdirs(obj, rootPath, modelList)
            % Ensure each model type has its own subfolder under rootPath
            for j = 1:numel(modelList)
                mdir = fullfile(rootPath, char(modelList{j}));
                obj.ensureDirExists(mdir);
            end
        end

        function obj = validateDataset(obj)
            % Robust validation using string arrays (case-insensitive)
            dsList = lower(string(obj.datasets));     % allowed set
            dtype  = lower(string(obj.datasetType));  % user-specified

            if ~ismember(dtype, dsList)
                allowed = strjoin(cellstr(dsList), ', ');
                error('Invalid datasetType "%s". Must be one of: %s', obj.datasetType, allowed);
            end

            % Store canonical lower-case token
            obj.datasetType = char(dtype);
        end

        function obj = validateModel(obj)
            % Normalize/validate model names (case-insensitive)
            mlist = string(obj.sourceModels);
            idx = find(lower(mlist) == lower(string(obj.typeNet)), 1, 'first');
            if isempty(idx)
                allowed = strjoin(cellstr(mlist), ', ');
                error('Invalid typeNet "%s". Must be one of: %s', obj.typeNet, allowed);
            end
            % Store as the canonical spelling from the allowed list
            obj.typeNet = char(mlist(idx));
        end

        function summarize(obj)
            % Print a concise summary of the configuration
            fprintf('\n=== Experiment Configuration ===\n');
            fprintf('projectLoc   : %s\n', obj.projectLoc);
            fprintf('datasetLoc   : %s\n', obj.datasetLoc);
            fprintf('resultLoc    : %s\n', obj.resultLoc);
            fprintf('pretrained   : %s\n', fullfile(obj.projectLoc, 'pretrainedModels'));
            fprintf('datasetType  : %s\n', obj.datasetType);
            fprintf('typeNet      : %s\n', obj.typeNet);
            fprintf('layerName    : %s\n', obj.layerName);
            fprintf('domain       : %s\n', obj.domain);
            fprintf('ncval        : %d\n', obj.ncval);
            fprintf('parTrain     : %d\n', obj.parTrain);
            if ~isempty(obj.ncats)
                fprintf('ncats        : %d\n', obj.ncats);
            end
            fprintf('experimentTag: %s\n', obj.experimentTag);
            fprintf('===============================\n\n');
        end

        function obj = setDataset(obj, newDataset)
            obj.datasetType = char(newDataset);
            obj = obj.validateDataset();
            obj = obj.setupDirectories();
        end

        function obj = setModel(obj, newModel)
            obj.typeNet = char(newModel);
            obj = obj.validateModel();
            obj = obj.setupDirectories();
        end
    end

    methods (Access = private)
        function ensureDirExists(obj, pathStr)
            if ~exist(pathStr, 'dir')
                mkdir(pathStr);
                if obj.verbose
                    fprintf('[mkdir] %s\n', pathStr);
                end
            end
        end

        function ds = canonicalDatasetNames(obj)
            % Return canonical dataset folder names (lower-case)
            ds = cellstr(lower(string(obj.datasets)));
        end

        function root = detectProjectRoot(obj) %#ok<INUSD>
            % Try MATLAB Project first
            root = '';
            try
                if exist('matlab.project.currentProject','file') == 2
                    prj = matlab.project.currentProject;
                    if ~isempty(prj)
                        root = char(prj.RootFolder);
                    end
                end
            catch
                % ignore; will fall back below
            end

            if ~isempty(root) && isfolder(root)
                return;
            end

            % Fallback: folder containing this class file
            thisFile = mfilename('fullpath');
            [root, ~, ~] = fileparts(thisFile);

            % Optional: if you keep classes in a 'code' subfolder, go one level up
            % parent = fileparts(root);
            % if isfolder(fullfile(parent,'datasets'))
            %     root = parent;
            % end
        end
    end
end
