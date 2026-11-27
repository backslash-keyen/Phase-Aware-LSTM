classdef associativeLSTMLayer < nnet.layer.Layer & nnet.layer.Formattable
    % associativeLSTMLayer  Associative LSTM (Danihelka et al., ICML 2016, Sec. 5).
    %
    % N = NumComplex complex cells, held as Nh = 2N real values h = [Re; Im]
    % (Eq. 16). S = NumCopies redundant copies of the memory. Per time step:
    %   Z = W x_t + U h_{t-1} + b                                   (Eqs. 17-18)
    %   gf, gi, go = sigmoid(.)  [N], applied to Re and Im alike    (Eqs. 19-21)
    %   ri = key(ri^),  ro = key(ro^),  u = bound(u^)               (Eqs. 23-25)
    %   c_s = gf (.) R c_s,prev + (P_s ri) (*) (gi (.) u)            (Eqs. 26-27)
    %   h_t = go (.) bound( 1/S sum_s (P_s ro) (*) c_s )             (Eqs. 29-30)
    % bound(z) = z / max(1, |z|) per complex number (Eq. 22); (*) is complex
    % multiplication (Eq. 28); P_s are constant random permutations of the N
    % cells, the same for Re and Im (Eq. 26). Rows of Z (9N):
    %   [gf; gi; go; Re ri^; Re ro^; Re u^; Im ri^; Im ro^; Im u^]
    % (a fixed reordering of Eq. 17's rows, so that one normalisation serves
    % keys and update). Configuration (assocVersions.m):
    %   keys    'bound'  key = bound(.), as the paper;  'unit'  key = z/|z|
    %   whu     false -> the rows of U that feed u^ are zero and never trained
    %   omega   true  -> R = e^{i omega} (learned per cell, same in every copy);
    %           false -> R = 1, as the paper
    %   output  'h' [Re h; Im h] (2N, the builder adds a linear readout),
    %           'real' Re h (N),  'mod' |h| (N)
    %   gradCap finite -> learnable gradients rescaled in double to L2 norm
    %           <= gradCap before trainNetwork sees them (norms always logged)
    % States start at zero for every sequence. Forward and backward are
    % vectorised per time step (copies along dim 3) and hand-written, CPU or
    % gpuArray; checkAssocLSTM compares them with dlgradient of a per-copy
    % dlarray implementation of the equations.

    properties
        NumComplex
        NumCopies
        Perm            % N x S constant permutations (Eq. 26)
        InvPerm         % N x S inverses
        Cfg
        VersionId
    end

    properties (Learnable)
        W
        U
        b
        omega
    end

    methods
        function layer = associativeLSTMLayer(numComplex, args)
            arguments
                numComplex
                args.Name = "";
                args.Version = "paper";
                args.Cfg = [];
            end
            if isempty(args.Cfg)
                cfg = assocVersions(char(args.Version));
            else
                cfg = args.Cfg;
            end
            cfg.numComplex = numComplex;
            layer.Name       = args.Name;
            layer.NumComplex = numComplex;
            layer.NumCopies  = cfg.numCopies;
            layer.Cfg        = cfg;
            layer.VersionId  = char(args.Version);
            % constant random permutations from their own stream, so the
            % global stream (network init, shuffles) is untouched
            rs = RandStream('threefry', 'Seed', cfg.permSeed);
            P = zeros(numComplex, cfg.numCopies);
            Pi = P;
            for s = 1:cfg.numCopies
                P(:, s) = randperm(rs, numComplex).';
                Pi(P(:, s), s) = (1:numComplex).';
            end
            layer.Perm = P;
            layer.InvPerm = Pi;
            layer.Description = "Associative LSTM (" + layer.VersionId + "), " + numComplex + ...
                " complex cells, " + cfg.numCopies + " copies";
        end

        function layer = initialize(layer, layout)
            % Only empty learnables are drawn, so a trained layer survives.
            % Global-stream draws, in order: W (Glorot), U (orthogonal).
            cfg = layer.Cfg;
            N = layer.NumComplex;
            C = layout.Size(finddim(layout, "C"));
            if isempty(layer.W), layer.W = single(initializeGlorot([9*N, C])); end
            if isempty(layer.U)
                U = single(initializeOrthogonal([9*N, 2*N]));
                if ~cfg.whu, U(uRows(N), :) = 0; end
                layer.U = U;
            end
            if isempty(layer.b)
                b = zeros(9*N, 1, 'single');
                b(1:N) = cfg.forgetBias;
                layer.b = b;
            end
            if isempty(layer.omega)
                if cfg.omega
                    % one frequency per cell, log-spaced over 1-40 Hz (delta
                    % to low gamma), in radians per sample at fs
                    fq = logspace(log10(1), log10(40), N).';
                    layer.omega = single(2*pi*fq/cfg.fs);
                else
                    layer.omega = zeros(N, 1, 'single');
                end
            end
        end

        function Y = predict(layer, X)
            Y = layer.run(X, false);
        end

        function [Y, memory] = forward(layer, X)
            [Y, memory] = layer.run(X, true);
        end

        function varargout = backward(layer, X, varargin)
            % backward(layer, X, Y, dLdY, memory)
            memory = varargin{end};
            dLdY = varargin{end-1};
            cfg = layer.Cfg;
            N = layer.NumComplex; S = layer.NumCopies;
            Xd = raw(stripdims(dlarray(X, 'CBT')));
            [C, B, T] = size(Xd, 1:3);
            W = raw(layer.W); U = raw(layer.U); om = raw(layer.omega);
            dY = raw(stripdims(dLdY));
            [Li, Lo, Lu] = keyIndex(layer.Perm, B, N, Xd);
            lo = normFloor(cfg, N, Xd);
            [cw, sw] = rotation(om, cfg);
            top = 1:N; bot = N+1:2*N;
            Zs = memory.Z; CR = memory.cR; CI = memory.cI; Hs = memory.h;

            dhR = zeros(N, B, 'like', Xd); dhI = dhR;      % from step t+1 through U
            dcR = zeros(N, B, S, 'like', Xd); dcI = dcR;
            dZs = zeros(9*N, B, T, 'like', Xd);
            dom = zeros(N, 1, 'like', Xd);

            for t = T:-1:1
                Z  = Zs(:,:,t);
                sg = 1 ./ (1 + exp(-Z(1:3*N,:)));
                f = sg(top,:); ig = sg(bot,:); o = sg(2*N+1:3*N,:);
                A0 = Z(3*N+1:6*N,:); B0 = Z(6*N+1:9*N,:);
                d  = max(sqrt(A0.^2 + B0.^2), lo);
                A  = A0 ./ d; Bn = B0 ./ d;
                KiR = A(Li); KiI = Bn(Li); KoR = A(Lo); KoI = Bn(Lo);
                uR = A(Lu); uI = Bn(Lu);
                vR = ig .* uR; vI = ig .* uI;
                cpR = CR(:,:,:,t); cpI = CI(:,:,:,t);
                if cfg.omega
                    aR = cw .* cpR - sw .* cpI; aI = sw .* cpR + cw .* cpI;
                else
                    aR = cpR; aI = cpI;
                end
                cR = CR(:,:,:,t+1); cI = CI(:,:,:,t+1);
                mR = sum(KoR .* cR - KoI .* cI, 3) / S;
                mI = sum(KoR .* cI + KoI .* cR, 3) / S;
                dm = max(1, sqrt(mR.^2 + mI.^2));
                bR = mR ./ dm; bI = mI ./ dm;

                % h_t = go (.) b, and the output
                gR = dhR; gI = dhI;
                switch cfg.output
                    case 'h',    gR = gR + dY(top,:,t); gI = gI + dY(bot,:,t);
                    case 'real', gR = gR + dY(:,:,t);
                end
                dgo = gR .* bR + gI .* bI;
                dbR = o .* gR; dbI = o .* gI;
                if strcmp(cfg.output, 'mod')
                    nb = sqrt(bR.^2 + bI.^2 + 1e-12);
                    dgo = dgo + dY(:,:,t) .* nb;
                    gb = dY(:,:,t) .* o ./ nb;
                    dbR = dbR + gb .* bR; dbI = dbI + gb .* bI;
                end
                [dmR, dmI] = normBack(bR, bI, dm, dbR, dbI);

                % m = 1/S sum_s Ko_s (*) c_s
                dpR = dmR / S; dpI = dmI / S;                  % broadcast over copies
                dKoR = dpR .* cR + dpI .* cI;
                dKoI = dpI .* cR - dpR .* cI;
                dcR = dcR + dpR .* KoR + dpI .* KoI;
                dcI = dcI + dpI .* KoR - dpR .* KoI;

                % c_s = f (.) a_s + Ki_s (*) v
                df  = sum(dcR .* aR + dcI .* aI, 3);
                daR = f .* dcR; daI = f .* dcI;
                dKiR = dcR .* vR + dcI .* vI;
                dKiI = dcI .* vR - dcR .* vI;
                dvR = sum(dcR .* KiR + dcI .* KiI, 3);
                dvI = sum(dcI .* KiR - dcR .* KiI, 3);
                if cfg.omega
                    dom = dom + sum(sum(daI .* aR - daR .* aI, 3), 2);
                    dcR = cw .* daR + sw .* daI;
                    dcI = cw .* daI - sw .* daR;
                else
                    dcR = daR; dcI = daI;
                end

                % back through the gathers into the normalised [ri ro u]
                dA = [unperm(dKiR, layer.InvPerm); unperm(dKoR, layer.InvPerm); ig .* dvR];
                dB = [unperm(dKiI, layer.InvPerm); unperm(dKoI, layer.InvPerm); ig .* dvI];
                [dA0, dB0] = normBack(A, Bn, d, dA, dB);
                dig = dvR .* uR + dvI .* uI;
                dZ = [[df; dig; dgo] .* sg .* (1 - sg); dA0; dB0];
                dZs(:,:,t) = dZ;
                dh = U.' * dZ;
                dhR = dh(top,:); dhI = dh(bot,:);
            end

            dZ2 = reshape(dZs, 9*N, B*T);
            dW = dZ2 * reshape(Xd, C, B*T).';
            dU = dZ2 * reshape(Hs(:,:,1:T), 2*N, B*T).';
            if ~cfg.whu, dU(uRows(N), :) = 0; end
            db = sum(dZ2, 2);
            if ~cfg.omega, dom(:) = 0; end
            dX = pagemtimes(W.', dZs);

            grads = assocCapGrads({dW, dU, db, dom}, cfg.gradCap);
            out = [{dX}, grads];
            out = out(1:nargout);
            for k = 1:numel(out)
                if k == 1 && isa(X, 'dlarray') && ~isempty(dims(X))
                    out{k} = dlarray(out{k}, dims(X));
                else
                    out{k} = dlarray(out{k});
                end
            end
            varargout = out;
        end
    end

    methods (Access = private)
        function [Y, memory] = run(layer, X, keep)
            cfg = layer.Cfg;
            N = layer.NumComplex; S = layer.NumCopies;
            Xd = raw(stripdims(dlarray(X, 'CBT')));
            [~, B, T] = size(Xd, 1:3);
            W = raw(layer.W); U = raw(layer.U); b = raw(layer.b); om = raw(layer.omega);
            WX = pagemtimes(W, Xd) + b;
            [Li, Lo, Lu] = keyIndex(layer.Perm, B, N, WX);
            lo = normFloor(cfg, N, WX);
            [cw, sw] = rotation(om, cfg);
            top = 1:N; bot = N+1:2*N;

            h  = zeros(2*N, B, 'like', WX);
            cR = zeros(N, B, S, 'like', WX); cI = cR;
            if strcmp(cfg.output, 'h'), Yd = zeros(2*N, B, T, 'like', WX);
            else, Yd = zeros(N, B, T, 'like', WX); end
            if keep
                Zs = zeros(9*N, B, T, 'like', WX);
                Hs = zeros(2*N, B, T+1, 'like', WX);
                CR = zeros(N, B, S, T+1, 'like', WX); CI = CR;
            end
            for t = 1:T
                Z  = WX(:,:,t) + U * h;
                sg = 1 ./ (1 + exp(-Z(1:3*N,:)));
                A  = Z(3*N+1:6*N,:); Bn = Z(6*N+1:9*N,:);
                d  = max(sqrt(A.^2 + Bn.^2), lo);
                A  = A ./ d; Bn = Bn ./ d;
                ig = sg(bot,:);
                vR = ig .* A(Lu); vI = ig .* Bn(Lu);
                KiR = A(Li); KiI = Bn(Li);
                if cfg.omega
                    [cR, cI] = deal(cw .* cR - sw .* cI, sw .* cR + cw .* cI);
                end
                f  = sg(top,:);
                nR = f .* cR + KiR .* vR - KiI .* vI;
                cI = f .* cI + KiR .* vI + KiI .* vR;
                cR = nR;
                KoR = A(Lo); KoI = Bn(Lo);
                mR = sum(KoR .* cR - KoI .* cI, 3) / S;
                mI = sum(KoR .* cI + KoI .* cR, 3) / S;
                dm = max(1, sqrt(mR.^2 + mI.^2));
                o  = sg(2*N+1:3*N,:);
                hR = o .* (mR ./ dm); hI = o .* (mI ./ dm);
                h  = [hR; hI];
                switch cfg.output
                    case 'h',    Yd(:,:,t) = h;
                    case 'real', Yd(:,:,t) = hR;
                    case 'mod',  Yd(:,:,t) = o .* sqrt((mR.^2 + mI.^2) ./ dm.^2 + 1e-12);
                end
                if keep
                    Zs(:,:,t) = Z;
                    Hs(:,:,t+1) = h;
                    CR(:,:,:,t+1) = cR; CI(:,:,:,t+1) = cI;
                end
            end
            Y = dlarray(Yd, 'CBT');
            if keep
                memory = struct('Z', Zs, 'h', Hs, 'cR', CR, 'cI', CI);
            else
                memory = [];
            end
        end
    end
end

function r = uRows(N)
% rows of Z that hold u^ (Re and Im)
r = [5*N+1:6*N, 8*N+1:9*N];
end

function v = raw(v)
if isa(v, 'dlarray'), v = extractdata(v); end
end

function [Li, Lo, Lu] = keyIndex(Perm, B, N, like)
% Linear indices into the normalised [ri; ro; u] block (3N x B):
% Li(j,b,s) -> ri(Perm(j,s), b), Lo the same for ro, Lu(j,b) -> u(j,b).
S = size(Perm, 2);
col = reshape((0:B-1) * 3*N, 1, B);
Li = reshape(Perm, N, 1, S) + col;
Lo = Li + N;
Lu = (2*N + (1:N)).' + col;
if isa(like, 'gpuArray'), Li = gpuArray(Li); Lo = gpuArray(Lo); Lu = gpuArray(Lu); end
end

function lo = normFloor(cfg, N, like)
% d = max(|z|, lo): lo = 1 is bound() (Eq. 22); lo = 1e-6 is a unit key
lo = ones(3*N, 1, 'like', like);
if strcmp(cfg.keys, 'unit'), lo(1:2*N) = 1e-6; end
end

function [cw, sw] = rotation(om, cfg)
if cfg.omega, cw = cos(om); sw = sin(om); else, cw = []; sw = []; end
end

function g = unperm(G, InvPerm)
% G(j,b,s) is the gradient of key(Perm(j,s), b): sum the copies back
[N, B, S] = size(G);
g = zeros(N, B, 'like', G);
for s = 1:S
    g = g + G(InvPerm(:, s), :, s);
end
end

function [ga, gb] = normBack(an, bn, d, ga, gb)
% z -> z/d with d = max(|z|, lo): identity/d where d = lo (|z| below the
% floor), (I - zh zh')/|z| where d = |z|. an, bn are the normalised values.
act = sqrt(an.^2 + bn.^2) >= 1 - 1e-6;     % normalised modulus 1 <=> d = |z|
dt  = (an .* ga + bn .* gb) .* act;
ga  = (ga - an .* dt) ./ d;
gb  = (gb - bn .* dt) ./ d;
end
