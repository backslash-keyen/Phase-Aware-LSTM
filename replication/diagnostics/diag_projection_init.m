function diag_projection_init(folds, variants, nIter, b2scale)
%DIAG_PROJECTION_INIT  Read-only initialisation / early-training diagnostic of the
% phase-lstm projection (CWL C2 DeepRNN target, our source models, default options).
% Network built with the released pipeline (perfold_dataset -> defineModel), the
% projection swapped by the variant's get_pLayer_phaseLSTM (fused = released model,
% signed = signedPhaseLSTM). Training: a custom loop reproducing trainNetwork's
% default-options update (Adam with the options' beta1/beta2/epsilon, L2 and
% per-parameter l2norm gradient clipping, learn-rate and L2 factors per parameter,
% sequential mini-batches in dsTrain order, cross-entropy). It is an approximation
% of trainNetwork, used only to see what happens in the first iterations.
% Writes nothing except what it prints.
% Usage: diag_projection_init(folds, {'released'|'fused'|'signed'|'halfAngMatched'|'notrig'|'noCarry'}, nIter[, b2scale])
% Caveat: gateStats always uses the released equations; for other variants only
% the bias rows and gradients are meaningful.
if nargin < 3, nIter = 50; end
if nargin < 4, b2scale = 0; end
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));   % repo root (file in replication/diagnostics/)
cd(root);
warning('off', 'all');
addpath(genpath(fullfile(root, 'utils')));
addpath(fullfile(root, 'replication'));
logAt  = [0 5 10 25 50];
gradAt = [0 5 25];
nVal   = 300;

for v = variants
    if strcmp(v{1}, 'released')
        vdir = '';                      % original phaseAwareLSTM (autodiff), slow
    else
        vdir = fullfile(root, 'replication', 'variants', [v{1} 'PhaseLSTM']);
        addpath(vdir, '-begin');
    end
    for i = folds
        fprintf('\n################ %s  FOLD %d ################\n', upper(v{1}), i);
        config = configureExperiment('verbose', false);
        config.typeNet = 'DeepRNN'; config.layerName = 'phase-lstm';
        config.domain = 'target-domain'; config.datasetType = 'cwl';
        config.ncats = 2; config.ncval = 10; config.parTrain = false;
        config = config.validateModel();
        data  = perfold_dataset(config, i);
        model = defineModel(data, config, i);
        opts  = model.options;
        lg = model.layers;
        outName = lg.Layers(end).Name;                 % classification layer
        lg = removeLayers(lg, outName);
        ensureReproducibility(i);
        net = dlnetwork(lg);
        if b2scale ~= 0                                  % diagnostic: phase bias = unit-forget pattern * b2scale
            Lb = net.Learnables;
            r = strcmp(Lb.Layer, 'phaseLSTM') & strcmp(Lb.Parameter, 'b2');
            H0 = size(Lb.Value{r}, 1) / 4;
            bb = zeros(4*H0, 1); bb([1:2*H0, 3*H0+1:4*H0]) = b2scale;   % f, i, o phase rows; candidate 0
            Lb.Value{r} = cast(bb, 'like', Lb.Value{r});
            net.Learnables = Lb;
            fprintf('b2 override: %.4f on rows 1:2H and 3H+1:4H, 0 on 2H+1:3H\n', b2scale);
        end
        pl = net.Layers(strcmp({net.Layers.Name}, 'phaseLSTM'));
        fprintf('projection layer class: %s, H = %d\n', class(pl), pl.NumHiddenUnits);
        H = pl.NumHiddenUnits;

        % ---------- data ----------
        T = readall(data.dsTrain);                      % shuffled order used by trainNetwork
        Xtr = cat(3, T{:,1}); Ytr = cat(1, T{:,2}); clear T
        cats = categories(Ytr);
        Xv = data.xval(:,:,1:min(nVal, end)); Yv = data.yval(1:size(Xv,3)); Yv = Yv(:);
        toDL = @(x) dlarray(gpuArray(single(permute(x, [1 3 2]))), 'CBT');   % C x B x T
        XvD = toDL(Xv);
        TvOH = single(onehotencode(categorical(Yv, cats), 2)');          % 2 x N

        % ---------- 1. initialisation ----------
        b1 = ex(pl.b1); b2 = ex(pl.b2);
        blk = {1:H, H+1:2*H, 2*H+1:3*H, 3*H+1:4*H};
        fprintf('b1 per block [rows 1:H | H+1:2H | 2H+1:3H | 3H+1:4H]: sums %s, nonzeros %s\n', ...
            mat2str(cellfun(@(r) sum(b1(r)), blk)), mat2str(cellfun(@(r) nnz(b1(r)), blk)));
        fprintf('   b1 ones at rows %s ... %s (count %d)\n', mat2str(find(b1==1,1)), mat2str(find(b1==1,1,'last')), nnz(b1==1));
        fprintf('b2: all zero = %d (max |b2| %.3g)\n', all(b2==0), max(abs(b2)));
        gateStats(pl, Xv(:,:,1:100), v{1});

        % ---------- 2-4. early training ----------
        L = net.Learnables;
        nL = height(L);
        lrf = zeros(nL,1); l2f = zeros(nL,1);
        for k = 1:nL
            lrf(k) = getLearnRateFactor(net, L.Layer{k}, L.Parameter{k});
            l2f(k) = getL2Factor(net, L.Layer{k}, L.Parameter{k});
        end
        isProj = strcmp(L.Layer, 'phaseLSTM');
        fprintf('trainable params: projection %d tensors, others with LR factor > 0: %d of %d\n', ...
            nnz(isProj), nnz(lrf > 0 & ~isProj), nnz(~isProj));
        m = cellfun(@(x) zeros(size(x), 'like', x), L.Value, 'UniformOutput', false);
        s = m;
        lr = opts.InitialLearnRate; be1 = opts.GradientDecayFactor; be2 = opts.SquaredGradientDecayFactor;
        ep = opts.Epsilon; l2 = opts.L2Regularization; gth = opts.GradientThreshold;
        B = opts.MiniBatchSize; nB = floor(size(Xtr,3) / B);
        fprintf('options: LR %g beta1 %g beta2 %g eps %g L2 %g gradThresh %g (%s) batch %d\n', ...
            lr, be1, be2, ep, l2, gth, opts.GradientThresholdMethod, B);

        for it = 0:nIter
            if ismember(it, logAt)
                report(net, XvD, TvOH, it, H);
            end
            if it == nIter, break; end
            b = mod(it, nB);
            idx = b*B + (1:B);
            XD = toDL(Xtr(:,:,idx));
            TD = single(onehotencode(Ytr(idx), 2)');
            [loss, G, st] = dlfeval(@lossGrad, net, XD, gpuArray(TD));
            keep = ismember(st.Parameter, {'TrainedMean', 'TrainedVariance'});   % BN stats only;
            S0 = net.State; S0.Value(keep) = st.Value(keep); net.State = S0;   % LSTM states reset per batch
            if ismember(it, gradAt)
                gn = cellfun(@(g) gather(norm(extractdata(g(:)))), G.Value);
                fprintf('   iter %2d grad norms (before L2/clip) | frozen-net trainable total %.3g | batch loss %.4f\n', it, ...
                    sqrt(sum(gn(~isProj & lrf > 0).^2)), gather(extractdata(loss)));
                projIdx = find(isProj);
                fprintf('      projection: %s\n', strjoin(arrayfun(@(k) sprintf('%s %.3g', L.Parameter{k}, gn(k)), projIdx', 'UniformOutput', false), ', '));
            end
            % trainNetwork-style update: L2 into gradient, per-parameter l2norm clip, Adam
            L = net.Learnables;
            for k = 1:nL
                if lrf(k) == 0, continue; end
                g = G.Value{k} + l2 * l2f(k) * L.Value{k};
                gnk = norm(extractdata(g(:)));
                if gnk > gth, g = g * (gth / gnk); end
                m{k} = be1 * m{k} + (1 - be1) * g;
                s{k} = be2 * s{k} + (1 - be2) * g.^2;
                mh = m{k} / (1 - be1^(it+1)); sh = s{k} / (1 - be2^(it+1));
                L.Value{k} = L.Value{k} - lr * lrf(k) * mh ./ (sqrt(sh) + ep);
            end
            net.Learnables = L;
        end
        clear net data model Xtr Ytr XvD
    end
    if ~isempty(vdir), rmpath(vdir); end
end
end

% ---------------------------------------------------------------------
function [loss, G, st] = lossGrad(net, X, T)
[P, st] = forward(net, X, 'Acceleration', 'none');
P = stripdims(P);
loss = -sum(T .* log(max(P, 1e-8)), 'all') / size(T, 2);
G = dlgradient(loss, net.Learnables);
end

function report(net, XvD, TvOH, it, H)
[Yp, P] = predict(net, XvD, 'Outputs', {'phaseLSTM', net.OutputNames{1}}, 'Acceleration', 'none');
Yp = gather(double(extractdata(stripdims(Yp))));      % H x N x T
P  = gather(double(extractdata(stripdims(P))));       % 2 x N
ce = -mean(sum(TvOH .* log(max(P, 1e-8)), 1));
[~, cls] = max(P, [], 1);
[~, tru] = max(TvOH, [], 1);
Z = reshape(permute(Yp, [1 3 2]), H, []);            % H x (T*N)
C = corrcoef(Z');
off = C(~eye(H)); off = off(~isnan(off));
varTot = var(Z, 0, 2);
varSamp = squeeze(mean(var(Yp, 0, 2), 3));            % variance across samples, averaged over time
dep = varSamp ./ max(varTot, eps);
fprintf('iter %2d | proj mean %.4f std %.4f min %.3g max %.3g fracNeg %.3f | unit corr mean %.3f (min %.3f) | input-dependence mean %.3f [%.3f..%.3f] | val CE %.4f | pred class1 %.3f | val acc %.3f\n', ...
    it, mean(Z(:)), std(Z(:)), min(Z(:)), max(Z(:)), mean(Z(:) < 0), mean(off), min(off), mean(dep), min(dep), max(dep), ce, mean(cls == 1), mean(cls == tru));
end

function gateStats(pl, X, vname)
% Released forward equations (as phaseAwareLSTM.predict), double precision, CPU.
W1 = double(ex(pl.W1)); W2 = double(ex(pl.W2));
U  = double(ex(pl.U));  Q  = double(ex(pl.Q));
b1 = double(ex(pl.b1)); b2 = double(ex(pl.b2));
H = pl.NumHiddenUnits; [~, Tn, N] = size(X); e = 1e-9;
i1 = 1:H; i2 = H+1:2*H; i3 = 2*H+1:3*H; i4 = 3*H+1:4*H;
h = zeros(H, N); ph = zeros(H, N); cR = zeros(H, N); cI = zeros(H, N);
acc = zeros(1, 8); phAll = zeros(H, N, Tn); mAll = zeros(H, N, Tn);
for t = 1:Tn
    x = double(squeeze(X(:, t, :)));
    G  = max(min(W1*x + b1 + U*h, 20), -20);
    Hh = max(min(W2*x + b2 + Q*ph, 20), -20);
    sg = 1 ./ (1 + exp(-G));
    f = sg(i1,:); in = sg(i2,:); o = sg(i4,:);
    th = tanh(0.5 * [Hh(i1,:); Hh(i2,:)]); den = 1 + th.^2 + e;
    c = (1 - th.^2) ./ den; s = 2*th ./ den; r = sqrt(c.^2 + s.^2 + e); c = c./r; s = s./r;
    xr = [cR; tanh(G(i3,:))]; yi = [cI; tanh(Hh(i3,:))];
    mg = [f; in];
    xn = mg .* (xr .* c - yi .* s); yn = mg .* (xr .* s + yi .* c);
    cR = xn(1:H,:) + xn(H+1:end,:); cI = yn(1:H,:) + yn(H+1:end,:);
    mag = max(sqrt(cR.^2 + cI.^2), e);
    ph = atan2(cI, cR);
    if strcmp(vname, 'signed'), h = o .* tanh(sign(cR + cI) .* mag); else, h = o .* tanh(mag); end
    ang = atan2(s, c);
    acc = acc + [mean(f(:)) mean(in(:)) mean(o(:)) mean(mag(:)) mean(abs(cR(:))) mean(abs(cI(:))) std(ang(:)) mean(h(:))];
    phAll(:,:,t) = ph;
    mAll(:,:,t) = sqrt(cR.^2 + cI.^2);
end
acc = acc / Tn;
fprintf('init |c|: min %.3g, fraction < 1e-3: %.4f, < 1e-2: %.4f  (atan2 gradient scales as 1/|c|^2)\n', ...
    min(mAll(:)), mean(mAll(:) < 1e-3), mean(mAll(:) < 1e-2));
fprintf('init gates (mean over %d samples x %d steps): forget(rows1:H) %.3f input(H+1:2H) %.3f output %.3f | |c| %.3f |cR| %.3f |cI| %.3f | rotation-angle std %.3f rad | h mean %.3f | hidden-phase std %.3f rad\n', ...
    N, Tn, acc, std(phAll(:)));
end

function v = ex(v)
if isa(v, 'dlarray'), v = extractdata(v); end
v = gather(v);
end
