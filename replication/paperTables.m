function T = paperTables()
%PAPERTABLES  Schema of every table in the paper, mapped to experiments.
%
% V. K N, S. Sundaram, C. N. Gupta, "Phase-aware LSTM projections for
% learning from heterogeneous electroencephalogram montages",
% Phys. Scr. 100 (2025) 126003, doi:10.1088/1402-4896/ae229d.
%
% Only the LAYOUT of each table is encoded here (caption, column groups,
% sections, row labels). No published numbers are stored anywhere in this
% repository: every cell is filled by buildTables() from fresh runs.
%
% Each cell points to one experiment key (see private/expKey.m), or is
%   dash(...)    a cell the paper itself leaves as "—"
%   missing(...) a cell whose model/data is not in the released code
%
% Domain names follow utils/crossvalidation/perfold_dataset.m:
%   source-domain     full montage            (fullchannels/)
%   source-domain-2   dual-channel source     (dualchannelsource/)  -> channel selection
%   target-domain     Fp1-Fz (CWL) / C3-C4 (MI) (targetdomain1/)
%   target-domain-2   F3-F7 (CWL)             (targetdomain2/)
%   temporalmasking-1 Fp1-Fz with masking     (temporalMasking_Fp1_Fz/)
%   temporalmasking-2 F3-F7 with masking      (temporalMasking_F3_F7/)

nets  = {'DeepRNN','EEGNet','transformer'};
netLabel = containers.Map(nets, {'DeepRNN','EEGNet','Transformer'});
projs = {'ffnn','lstm','gru'};
projLabel = containers.Map(projs, {'FFNN','LSTM','GRU'});

T = {};

%% Table 1 — hyperparameters (built from the training-option files)
t = newTable('table1', 'hyperparameters', ...
    'Hyperparameters for source and target domain training. Values were fixed across folds for each model.');
t.columns = {'Batch','LR','L2','Epochs (Src)','Epochs (Tgt)','Notes'};
notes = struct('CWL', {{'Values from single bayesOpt run','Values from single bayesOpt run','Heuristic choice'}}, ...
               'MI',  {{'—','Values from [ref]','Default Choices'}});
sections = {'CWL','Cognitive Workload Dataset: For binary and multi-category classification'; ...
            'MI', 'Motor Imagery Dataset'};
for s = 1:size(sections,1)
    for n = 1:numel(nets)
        r = newRow(sections{s,2}, netLabel(nets{n}), '');
        r.datasetType = sections{s,1};
        r.typeNet     = nets{n};
        r.note        = notes.(sections{s,1}){n};
        t.rows(end+1) = r;
    end
end
T{end+1} = t;

%% Table 2 — source domain performance
t = newTable('table2', 'accuracy', 'Source domain classification performance (accuracy in %).');
t.groups = {'CWL','MI-2a'};
for n = 1:numel(nets)
    net = nets{n};
    t.rows(end+1) = newRow('Source Domain - Binary Classification', netLabel(net), '', ...
        {expKey('CWL',2,net,'source-domain',net), expKey('MI',2,net,'source-domain',net)});
end
for n = 1:numel(nets)
    net = nets{n};
    t.rows(end+1) = newRow('Source Domain: Multi-Category Classification', netLabel(net), '', ...
        {expKey('CWL',3,net,'source-domain',net), dash()});
end
T{end+1} = t;

%% Table 3 — no-transfer and common-channel-selection baselines
t = newTable('table3', 'accuracy', ...
    'Baseline target domain classification performance (accuracy in %) for no-transfer and transfer under channel selection scheme.');
t.groups = {'CWL','MI-2'};
for n = 1:numel(nets)
    net = nets{n};
    t.rows(end+1) = newRow('No Transfer Learning', netLabel(net), '', ...
        {expKey('CWL',2,net,'target-domain',net), expKey('MI',2,net,'target-domain',net)});
end
for n = 1:numel(nets)
    net = nets{n};
    t.rows(end+1) = newRow('Transfer Learning with Common Channel Selection: Fp1–Fz | C3–C4', netLabel(net), '', ...
        {expKey('CWL',2,net,'target-domain','baseTransfer'), expKey('MI',2,net,'target-domain','baseTransfer')});
end
for n = 1:numel(nets)
    net = nets{n};
    t.rows(end+1) = newRow('Transfer Learning with Common Channel Selection: F3–F7', netLabel(net), '', ...
        {missing(['perfold_dataset.m has a single dualchannelsource/ folder, so an F3-F7 ' ...
                  'channel-selection source model cannot be addressed']), dash()});
end
T{end+1} = t;

%% Table 4 — projection baselines, binary
t = newTable('table4', 'accuracy', ...
    'Performance of projection transfer baselines in Target Domain (Fp1 & Fz) and (C3 & C4) (%), for the binary classification.');
t.groups = {'CWL','MI-2b'};
t = addProjectionBlock(t, 'Target Layout : Fp1-Fz (CWL) and C3-C4 (MI)', nets, netLabel, ...
        [projs, {'charm'}], [projLabel.values(projs), {'CHARM'}], 'target-domain', 2, true);
t = addProjectionBlock(t, 'Target Layout : F3-F7 (CWL)', nets, netLabel, ...
        [projs, {'charm'}], [projLabel.values(projs), {'CHARM'}], 'target-domain-2', 2, false);
T{end+1} = t;

%% Table 5 — F3-F7 multi-category: target baselines + projections
t = newTable('table5', 'accuracy', ...
    'Performance of target and projection baselines in the CWL dataset Target Domain (F3 & F7) (%), for the multi-category classification scheme (low, medium, high workload conditions).');
t.groups = {'CWL'};
for n = 1:numel(nets)
    net = nets{n};
    t.rows(end+1) = newRow('NO-TL', netLabel(net), '', {expKey('CWL',3,net,'target-domain-2',net)});
end
for n = 1:numel(nets)
    t.rows(end+1) = newRow('Target Baselines: TL-Common Channels', netLabel(nets{n}), '', ...
        {missing('no F3-F7 dual-channel source folder in perfold_dataset.m (see Table 3)')});
end
t = addProjectionBlock(t, 'Projection Transfer', nets, netLabel, ...
        [projs, {'charm'}], [projLabel.values(projs), {'CHARM'}], 'target-domain-2', 3, false, true);
T{end+1} = t;

%% Table 6 — phase-aware baselines, binary
phaseProjs  = {'phased-lstm','complex-lstm','phase-lstm'};
phaseLabels = {'Phased-LSTM','Complex-LSTM','Phase-Aware-LSTM (Proposed)'};
t = newTable('table6', 'accuracy', ...
    'Performance of phase-aware baselines in Target Domain (Fp1 & Fz) and (C3 & C4) (%), for the binary classification.');
t.groups = {'CWL','MI-2b'};
t = addProjectionBlock(t, 'Target Layout : Fp1-Fz (CWL) and C3-C4 (MI)', nets, netLabel, ...
        phaseProjs, phaseLabels, 'target-domain', 2, true);
t = addProjectionBlock(t, 'Target Layout : F3-F7 (CWL)', nets, netLabel, ...
        phaseProjs, phaseLabels, 'target-domain-2', 2, false);
T{end+1} = t;

%% Table 7 — phase-aware baselines, Fp1-Fz multi-category
t = newTable('table7', 'accuracy', ...
    'Performance of target and phase-aware baselines in the CWL dataset Target Domain (Fp1 & Fz) (%), for the multi-category classification scheme (low, medium, high workload conditions).');
t.groups = {'CWL'};
t = addProjectionBlock(t, 'Phase-Aware Projections', nets, netLabel, ...
        phaseProjs, phaseLabels, 'target-domain', 3, false, true);
T{end+1} = t;

%% Table 8 — temporal masking (DeepRNN, binary CWL)
t = newTable('table8', 'accuracy', ...
    'Projection transfer accuracy in target domain with temporal masking (F3 & F7) and (Fp1 & Fz) (%).');
t.groups = {'NEC-21'};
maskProj  = {'DeepRNN','ffnn','lstm','gru','phase-lstm'};
maskLabel = {'Baseline','FFNN','LSTM','GRU','phase-LSTM'};
masks = {'temporalmasking-1','Temporal Masking: Fp1—Fz'; 'temporalmasking-2','Temporal Masking: F3—F7'};
for m = 1:size(masks,1)
    for p = 1:numel(maskProj)
        r = newRow(masks{m,2}, 'DeepRNN', maskLabel{p}, {expKey('CWL',2,'DeepRNN',masks{m,1},maskProj{p})});
        if p == 1
            r.note = 'Assumed: "Baseline" = DeepRNN trained directly on the masked target data (no transfer).';
        end
        t.rows(end+1) = r;
    end
end
T{end+1} = t;

%% Table 9 — phase parametrization variants (CWL Fp1-Fz)
t = newTable('table9', 'accuracy', ...
    'Projection Transfer accuracy using various phase-aware LSTM variants with different rotation angle parametrization (CWL: Fp1 & Fz) (%).');
t.groups = {'NEC-21'};
variants = {'atan','atan2-a','atan2-b','tan-half-angle','numeric-sin-cos'};
for v = 1:numel(variants)
    if strcmp(variants{v}, 'tan-half-angle')
        c = expKey('CWL',2,'DeepRNN','target-domain','phase-lstm');
        note = ['Released phaseAwareLSTM.m implements the tangent half-angle form. ' ...
                'Assumed backbone: DeepRNN (the paper does not name one for this table).'];
    else
        c = missing(sprintf('the %s parametrization is not in utils/modelstuff/layerdefinitions/phaseAwareLSTM.m', variants{v}));
        note = '';
    end
    r = newRow('Phase-Aware-LSTM variants', 'Phase-Aware-LSTM', variants{v}, {c});
    r.note = note;
    t.rows(end+1) = r;
end
T{end+1} = t;

%% Table 10 — complexity and runtime
t = newTable('table10', 'complexity', ...
    ['Model complexity and runtime for projection-transfer variants. Columns report parameter count, FLOPs, ' ...
     'on-disk size, training time (s), and desktop inference latency (ms) and memory (KB); dashes denote metrics not recorded.']);
t.columns = {'#Params','FLOPs (M)','Disk-Size (KB)','Training Time (s)','Inference Time (ms)','Inference Mem (KB)'};
cproj  = {'ffnn','lstm','gru','complex-lstm','phase-lstm'};
clabel = {'FFNN','LSTM','GRU','complexLSTM','phaseLSTM'};
for n = 1:2   % DeepRNN, EEGNet only in the paper
    net = nets{n};
    for p = 1:numel(cproj)
        if strcmp(cproj{p}, 'complex-lstm')
            c = missing('Complex-LSTM projection is not in the released code');
        else
            c = expKey('CWL',2,net,'target-domain',cproj{p});
        end
        r = newRow('Target Domain—Projection Transfer', netLabel(net), clabel{p}, {c});
        if p == 1
            r.note = 'Assumed setting: CWL binary, Fp1-Fz target layout (the paper does not state it).';
        end
        t.rows(end+1) = r;
    end
end
T{end+1} = t;

%% Beyond the paper — Associative LSTM projection (CWL Fp1-Fz, binary)
% Not in the paper: Danihelka et al. (2016) as a further projection baseline
% next to FFNN / LSTM / GRU. Runs only with an al_<version> variant
% (replication/assocLSTM); without one the cell is reported as unsupported.
t = newTable('assocLSTM', 'accuracy', ...
    'Not in the paper: Associative LSTM projection (Danihelka et al. 2016), CWL target domain (Fp1 & Fz), binary (%).');
t.groups = {'CWL'};
t.rows(end+1) = newRow('Projection Transfer', 'DeepRNN', 'Associative-LSTM', ...
    {expKey('CWL',2,'DeepRNN','target-domain','assoc-lstm')});
T{end+1} = t;

T = [T{:}];
end

% =====================================================================
function t = addProjectionBlock(t, section, nets, netLabel, projs, labels, domain, ncats, withMI, firstColOnly)
if nargin < 10, firstColOnly = false; end
for n = 1:numel(nets)
    net = nets{n};
    for p = 1:numel(projs)
        proj = projs{p};
        if any(strcmp(proj, {'charm','phased-lstm','complex-lstm'}))
            cwl = missing(sprintf('%s projection is not in the released code', labels{p}));
            mi  = cwl;
        else
            cwl = expKey('CWL', ncats, net, domain, proj);
            mi  = expKey('MI',  ncats, net, 'target-domain', proj);
        end
        if ~withMI, mi = dash(); end
        if firstColOnly
            cells = {cwl};
        else
            cells = {cwl, mi};
        end
        t.rows(end+1) = newRow(section, netLabel(net), labels{p}, cells);
    end
end
end

function t = newTable(id, kind, caption)
t.id      = id;
t.kind    = kind;        % 'accuracy' | 'hyperparameters' | 'complexity'
t.caption = caption;
t.groups  = {};          % accuracy tables: column groups, each split into Validation | Test
t.columns = {};          % other tables: plain columns
t.rows    = struct('section',{},'network',{},'variant',{},'cells',{}, ...
                   'note',{},'datasetType',{},'typeNet',{});
end

function r = newRow(section, network, variant, cells)
if nargin < 4, cells = {}; end
r.section     = section;
r.network     = network;
r.variant     = variant;
r.cells       = cells;
r.note        = '';
r.datasetType = '';
r.typeNet     = '';
end

function c = dash()
c = struct('kind','dash','id','','reason','left blank in the paper');
end

function c = missing(reason)
c = struct('kind','missing','id','','reason',reason);
end
