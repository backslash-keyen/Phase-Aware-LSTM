function report = checkPhaseVersions(varargin)
%CHECKPHASEVERSIONS  Compare every fast phase-layer version with its verbatim original.
%
% Short test (a few minutes). Trains nothing, reads no data, writes nothing.
% From the repository root:
%   addpath('replication/versions');
%   report = checkPhaseVersions();                    % every version
%   report = checkPhaseVersions('ids', {'notrig','ver5'});
%
% For each version in phaseVersionList (plus the dedicated ports
% notrigFastPhaseLSTM, ver5FastPhaseLSTM and fusedPhaseLSTM) it builds the
% ORIGINAL layer (a temporary copy of replication/versions/originals/...,
% renamed so versions with the same class name do not clash) and the FAST
% layer, each after a sequenceInputLayer in a dlnetwork, gives them the same
% weights and input, and reports in double precision over Tcheck steps
%   fwd   largest |Y_fast - Y_orig| / max|Y_orig| (also the four final states)
%   grad  same for every learnable gradient (dlgradient of sum(Y.*w);
%         autodiff through the original, hand-written backward in the fast one)
% It then times one forward+backward of the fast layer at full size
% (H 30, B 197, T 500, single, GPU if available) and runs 2 trainNetwork
% iterations with it (the training code path).
% Expected: fwd and grad below about 1e-6. Short Tcheck because the
% recurrences amplify rounding (atan2 branch cut), see checkPhaseLSTMVersions.
p = inputParser;
addParameter(p, 'ids', {});
addParameter(p, 'H', 30);
addParameter(p, 'B', 8);
addParameter(p, 'C', 2);
addParameter(p, 'Tcheck', 20);
addParameter(p, 'timeFull', true);
addParameter(p, 'Bfull', 197);
addParameter(p, 'Tfull', 500);
addParameter(p, 'trainSmoke', true);
addParameter(p, 'seed', 0);
parse(p, varargin{:}); o = p.Results;

verDir  = fileparts(mfilename('fullpath'));
repDir  = fileparts(verDir);
rootDir = fileparts(repDir);
oldPath = path; restorePath = onCleanup(@() path(oldPath)); %#ok<NASGU>
addpath(genpath(fullfile(rootDir, 'utils')));
addpath(fullfile(verDir, 'fast', 'common'));
tmpRoot = tempname; mkdir(tmpRoot);
cleanTmp = onCleanup(@() rmdir(tmpRoot, 's')); %#ok<NASGU>
useGPU = canUseGPU;

V = phaseVersionList();
V = V(strcmp({V.status}, 'fast'));
cases = struct('name', {}, 'id', {}, 'fastDir', {}, 'fastCtor', {});
for v = V
    cases(end+1) = struct('name', v.id, 'id', v.id, 'fastDir', fullfile(verDir, 'fast', v.id), ...
                          'fastCtor', str2func(v.fastClass)); %#ok<AGROW>
end
cases(end+1) = struct('name', 'notrig (notrigFastPhaseLSTM)', 'id', 'notrig', ...
    'fastDir', fullfile(repDir, 'variants', 'notrigFastPhaseLSTM'), 'fastCtor', @phaseAwareLSTMNotrigFast);
cases(end+1) = struct('name', 'ver5 (ver5FastPhaseLSTM)', 'id', 'ver5', ...
    'fastDir', fullfile(repDir, 'variants', 'ver5FastPhaseLSTM'), 'fastCtor', @phaseAwareLSTMVer5Fast);
cases(end+1) = struct('name', 'halfAng (fusedPhaseLSTM)', 'id', 'halfAng', ...
    'fastDir', fullfile(repDir, 'variants', 'fusedPhaseLSTM'), 'fastCtor', @phaseAwareLSTMFused);
if ~isempty(o.ids), cases = cases(ismember({cases.id}, o.ids)); end

report = struct('name', {}, 'fwd', {}, 'state', {}, 'grad', {}, 'gradDetail', {}, 'seconds', {}, 'train', {}, 'error', {});
fprintf('\n%-30s %10s %10s %10s %9s  %s\n', 'version', 'fwd (Y)', 'states', 'grad', 'sec/iter', 'trainNetwork');
for c = cases
    r = struct('name', c.name, 'fwd', NaN, 'state', NaN, 'grad', NaN, 'gradDetail', [], 'seconds', NaN, 'train', '', 'error', '');
    vrow = V(strcmp({V.id}, c.id));
    addpath(c.fastDir, '-begin');
    try
        origCtor = loadOriginal(vrow, tmpRoot);
        B = o.B;
        if strcmp(c.id, 'halfAngV2'), B = 1; end      % the original only works with batch size 1
        rng(o.seed, 'twister');
        netO = dlnetwork([sequenceInputLayer(o.C, 'MinLength', o.Tcheck, 'Name', 'input'); ...
                          origCtor(o.H, 'Name', 'phaseLSTM', 'OutputMode', 'sequence')]);
        netF = dlnetwork([sequenceInputLayer(o.C, 'MinLength', o.Tcheck, 'Name', 'input'); ...
                          c.fastCtor(o.H, 'Name', 'phaseLSTM', 'OutputMode', 'sequence')]);
        netO = dlupdate(@(x) double(gather(x)), netO);
        netF = copyLearnables(netO, netF);
        X = randn(o.C, B, o.Tcheck);
        w = reshape(sin(1:o.H*B*o.Tcheck), o.H, B, o.Tcheck);
        [yo, so, go] = dlfeval(@lossAndGrads, netO, X, w);
        [yf, sf, gf] = dlfeval(@lossAndGrads, netF, X, w);
        r.fwd = rel(yo, yf);
        % h and phase states must match; the memory states only where the
        % original carries memory (no-carry originals return memory = 0)
        ds = 0;
        for k = 1:height(so)
            a = so.Value{k};
            if any(extractdata(dlarray(a)) ~= 0, 'all'), ds = max(ds, rel(a, sf.Value{k})); end
        end
        r.state = ds;
        names = intersect(go.Parameter, gf.Parameter, 'stable');
        gd = struct();
        for k = 1:numel(names)
            a = go.Value{strcmp(go.Parameter, names{k})};
            b = gf.Value{strcmp(gf.Parameter, names{k})};
            if isempty(a), continue; end
            gd.(names{k}) = rel(a, b);
        end
        r.gradDetail = gd;
        r.grad = max(cell2mat(struct2cell(gd)));
        if o.timeFull
            r.seconds = timeFast(c.fastCtor, o, useGPU);
        end
        if o.trainSmoke
            r.train = smoke(c.fastCtor, o, useGPU);
        end
    catch err
        r.error = getReport(err, 'basic', 'hyperlinks', 'off');
    end
    rmpath(c.fastDir);
    if isempty(r.error)
        fprintf('%-30s %10.2e %10.2e %10.2e %9.3f  %s\n', r.name, r.fwd, r.state, r.grad, r.seconds, r.train);
    else
        fprintf('%-30s ERROR %s\n', r.name, r.error);
    end
    report(end+1) = r; %#ok<AGROW>
end
fprintf(['\nExpected: fwd, states and grad below about 1e-6 (double, %d steps); the half-angle\n' ...
         'versions reach about 1e-6 to 1e-5 because their recurrence amplifies rounding.\n\n'], o.Tcheck);
end

function ctor = loadOriginal(v, tmpRoot)
% Temporary copy of the verbatim original with the class renamed orig_<id>
% (and its constructor, which ver7 misnames), so originals that share a class
% name (cstm_arch1, phaseAwareLSTM, phaseAwareLSTM_ver10) never clash and the
% released phaseAwareLSTM is never shadowed.
src = fileread(v.original);
cls = regexp(src, 'classdef\s+(\w+)', 'tokens', 'once'); cls = cls{1};
new = ['orig_' v.id];
src = regexprep(src, ['\<' cls '\>'], new);
src = regexprep(src, 'function\s+layer\s*=\s*\w+\s*\(\s*numHiddenUnits', ['function layer = ' new '(numHiddenUnits']);
d = fullfile(tmpRoot, v.id); mkdir(d);
fid = fopen(fullfile(d, [new '.m']), 'w'); fwrite(fid, src); fclose(fid);
for h = {'glorotUniform.m', 'blockOrthogonal.m'}
    f = fullfile(fileparts(v.original), h{1});
    if isfile(f), copyfile(f, d); end
end
addpath(d); rehash;
ctor = str2func(new);
end

function netF = copyLearnables(netO, netF)
L = netF.Learnables;
for k = 1:height(L)
    j = find(strcmp(netO.Learnables.Parameter, L.Parameter{k}));
    assert(isscalar(j), 'learnable %s missing in the original', L.Parameter{k});
    L.Value{k} = netO.Learnables.Value{j};
end
netF.Learnables = L;
netF = dlupdate(@(x) double(gather(x)), netF);
end

function [Y, S, gL] = lossAndGrads(net, Xraw, w)
X = dlarray(Xraw, 'CBT');                  % untraced, as in training
[Y, S] = forward(net, X, 'Acceleration', 'none');
Y = stripdims(Y);
loss = sum(Y .* w, 'all');
gL = dlgradient(loss, net.Learnables);
Y = extractdata(Y);
end

function d = rel(a, b)
a = double(gather(extractdata(dlarray(a)))); b = double(gather(extractdata(dlarray(b))));
d = max(abs(a(:) - b(:))) / max(max(abs(a(:))), realmin);
end

function s = timeFast(ctor, o, useGPU)
rng(1, 'twister');
net = dlnetwork([sequenceInputLayer(o.C, 'MinLength', o.Tfull, 'Name', 'input'); ...
                 ctor(o.H, 'Name', 'phaseLSTM', 'OutputMode', 'sequence')]);
X = single(randn(o.C, o.Bfull, o.Tfull));
w = single(reshape(sin(1:o.H*o.Bfull*o.Tfull), o.H, o.Bfull, o.Tfull));
if useGPU, net = dlupdate(@gpuArray, net); X = gpuArray(X); w = gpuArray(w); end
ts = zeros(1, 3);
for i = 1:3
    tic; dlfeval(@lossAndGrads, net, X, w);
    if useGPU, wait(gpuDevice); end
    ts(i) = toc;
end
s = median(ts(2:end));
end

function msg = smoke(ctor, o, useGPU)
Ts = 50; N = 64;
rng(2, 'twister');
Xs = arrayfun(@(i) single(randn(o.C, Ts)), 1:N, 'UniformOutput', false)';
Ys = categorical(randi(2, N, 1));
opts = trainingOptions('adam', 'MaxEpochs', 1, 'MiniBatchSize', 32, 'Verbose', false, ...
    'Plots', 'none', 'ExecutionEnvironment', ternary(useGPU, 'gpu', 'cpu'));
layers = [sequenceInputLayer(o.C, 'MinLength', Ts, 'Name', 'input'); ...
          ctor(o.H, 'Name', 'phaseLSTM', 'OutputMode', 'sequence'); ...
          lstmLayer(8, 'OutputMode', 'last'); fullyConnectedLayer(2); softmaxLayer; classificationLayer];
try
    tic; trainNetwork(Xs, Ys, layers, opts); msg = sprintf('ok (%.1f s)', toc);
catch err
    msg = ['FAILED: ' err.message];
end
end

function s = ternary(c, a, b)
if c, s = a; else, s = b; end
end
