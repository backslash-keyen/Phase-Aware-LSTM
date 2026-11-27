classdef associativeLSTMLayer < nnet.layer.Layer & nnet.layer.Formattable
    % associativeLSTMLayer  Associative LSTM (Danihelka et al., ICML 2016, Sec. 5).
    %
    % N = NumComplex complex cells, held as Nh = 2N real values h = [Re; Im]
    % (Eq. 16). S = NumCopies redundant copies of the memory. Per time step:
    %   [gf gi go ri^ ro^ u^] = W x_t + U h_{t-1} + b            (Eqs. 17-18)
    %   gf, gi, go = sigmoid(.)  [N], applied to Re and Im alike  (Eqs. 19-21)
    %   u  = bound(u^),  ri = key(ri^),  ro = key(ro^)            (Eqs. 23-25)
    %   c_s = gf (.) R c_s,prev + (P_s ri) (*) (gi (.) u)          (Eqs. 26-27)
    %   h_t = go (.) bound( 1/S sum_s (P_s ro) (*) c_s )           (Eqs. 29-30)
    % bound(z) = z / max(1, |z|) per complex number (Eq. 22); (*) is complex
    % multiplication (Eq. 28); P_s are constant random permutations of the N
    % cells, the same for Re and Im. Configuration (assocVersions.m):
    %   keys    'bound'  key = bound(.), as the paper
    %           'unit'   key = z/|z|, modulus exactly 1
    %   whu     false -> the rows of U that feed u^ are zero and never trained
    %   omega   true  -> R = e^{i omega} (learned per cell, same in every copy);
    %           false -> R = 1, as the paper
    %   output  'h'    [Re h; Im h] (2N); the builder adds a linear readout
    %           'real' Re h (N);  'mod' |h| (N)
    %   gradCap finite -> every learnable gradient rescaled in double to L2
    %           norm <= gradCap (Inf/NaN -> 0) before trainNetwork sees it
    % States start at zero for every sequence. Forward and backward are
    % vectorised per time step and hand-written (CPU or gpuArray);
    % checkAssocLSTM compares them with dlgradient of a dlarray version.

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
            % Order of the global-stream draws: W (Glorot), U (orthogonal).
            cfg = layer.Cfg;
            N = layer.NumComplex;
            C = layout.Size(finddim(layout, "C"));
            if isempty(layer.W), layer.W = single(initializeGlorot([9*N, C])); end
            if isempty(layer.U)
                U = single(initializeOrthogonal([9*N, 2*N]));
                if ~cfg.whu, U(7*N+1:9*N, :) = 0; end
                layer.U = U;
            end
            if isempty(layer.b)
                b = zeros(9*N, 1, 'single');
                b(1:N) = cfg.forgetBias;
                layer.b = b;
            end
            if isempty(layer.omega)
                if cfg.omega
                    % one frequency per cell, log-spaced over 1-40 Hz (delta to
                    % low gamma), in radians per sample at fs
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
            idx  = layer.Perm(:);
            iidx = reshape(layer.InvPerm + (0:S-1)*N, [], 1);   % un-permute, block by block
            [cw, sw] = rotation(om, cfg, S);
            top = 1:N; bot = N+1:2*N;

            Zs = memory.Z; CR = memory.cR; CI = memory.cI; Hs = memory.h;
            dhR = zeros(N, B, 'like', Xd); dhI = dhR;      % from step t+1 through U
            dcR = zeros(N*S, B, 'like', Xd); dcI = dcR;
            dZs = zeros(9*N, B, T, 'like', Xd);
            dom = zeros(N, 1, 'like', Xd);

            for t = T:-1:1
                Z = Zs(:,:,t);
                f  = sig(Z(top,:)); ig = sig(Z(bot,:)); o = sig(Z(2*N+1:3*N,:));
                [kiR, kiI] = keyFwd(Z(3*N+1:4*N,:), Z(4*N+1:5*N,:), cfg.keys);
                [koR, koI] = keyFwd(Z(5*N+1:6*N,:), Z(6*N+1:7*N,:), cfg.keys);
                [uR, uI]   = boundFwd(Z(7*N+1:8*N,:), Z(8*N+1:9*N,:));
                vR = ig .* uR; vI = ig .* uI;
                kiRP = kiR(idx,:); kiIP = kiI(idx,:);
                koRP = koR(idx,:); koIP = koI(idx,:);
                cpR = CR(:,:,t); cpI = CI(:,:,t);
                if cfg.omega
                    aR = cw .* cpR - sw .* cpI; aI = sw .* cpR + cw .* cpI;
                else
                    aR = cpR; aI = cpI;
                end
                cR = CR(:,:,t+1); cI = CI(:,:,t+1);
                mR = copySum(koRP .* cR - koIP .* cI, N, S, B) / S;
                mI = copySum(koRP .* cI + koIP .* cR, N, S, B) / S;
                [bR, bI] = boundFwd(mR, mI);

                % output and recurrent gradient into h_t = go (.) b
                gR = dhR; gI = dhI;
                switch cfg.output
                    case 'h'
                        gR = gR + dY(top,:,t); gI = gI + dY(bot,:,t);
                        dgo = 0;
                    case 'real'
                        gR = gR + dY(:,:,t);
                        dgo = 0;
                    case 'mod'
                        nb = modulus(bR, bI);
                        dgo = dY(:,:,t) .* nb;
                        gb = dY(:,:,t) .* o ./ nb;
                end
                dgo = dgo + gR .* bR + gI .* bI;
                dbR = o .* gR; dbI = o .* gI;
                if strcmp(cfg.output, 'mod')
                    dbR = dbR + gb .* bR; dbI = dbI + gb .* bI;
                end
                [dmR, dmI] = boundBack(mR, mI, dbR, dbI, 'bound');

                % m = 1/S sum_s ko_s (*) c_s
                dpR = repmat(dmR, S, 1) / S; dpI = repmat(dmI, S, 1) / S;
                dkoRP = dpR .* cR + dpI .* cI;
                dkoIP = dpI .* cR - dpR .* cI;
                dcR = dcR + dpR .* koRP + dpI .* koIP;
                dcI = dcI + dpI .* koRP - dpR .* koIP;

                % c_s = f (.) a_s + ki_s (*) v
                df  = copySum(dcR .* aR + dcI .* aI, N, S, B);
                daR = repmat(f, S, 1) .* dcR; daI = repmat(f, S, 1) .* dcI;
                vRr = repmat(vR, S, 1); vIr = repmat(vI, S, 1);
                dkiRP = dcR .* vRr + dcI .* vIr;
                dkiIP = dcI .* vRr - dcR .* vIr;
                dvR = copySum(dcR .* kiRP + dcI .* kiIP, N, S, B);
                dvI = copySum(dcI .* kiRP - dcR .* kiIP, N, S, B);
                if cfg.omega
                    dom = dom + copySum(sum(daI .* aR - daR .* aI, 2), N, S, 1);
                    dcR = cw .* daR + sw .* daI;
                    dcI = cw .* daI - sw .* daR;
                else
                    dcR = daR; dcI = daI;
                end

                % un-permute the key gradients
                dkiR = copySum(dkiRP(iidx,:), N, S, B); dkiI = copySum(dkiIP(iidx,:), N, S, B);
                dkoR = copySum(dkoRP(iidx,:), N, S, B); dkoI = copySum(dkoIP(iidx,:), N, S, B);
                [dZkiR, dZkiI] = boundBack(Z(3*N+1:4*N,:), Z(4*N+1:5*N,:), dkiR, dkiI, cfg.keys);
                [dZkoR, dZkoI] = boundBack(Z(5*N+1:6*N,:), Z(6*N+1:7*N,:), dkoR, dkoI, cfg.keys);
                dig = dvR .* uR + dvI .* uI;
                [dZuR, dZuI] = boundBack(Z(7*N+1:8*N,:), Z(8*N+1:9*N,:), ig .* dvR, ig .* dvI, 'bound');

                dZ = [df .* f .* (1 - f); dig .* ig .* (1 - ig); dgo .* o .* (1 - o); ...
                      dZkiR; dZkiI; dZkoR; dZkoI; dZuR; dZuI];
                dZs(:,:,t) = dZ;
                dh = U.' * dZ;
                dhR = dh(top,:); dhI = dh(bot,:);
            end

            dZ2 = reshape(dZs, 9*N, B*T);
            dW = dZ2 * reshape(Xd, C, B*T).';
            dU = dZ2 * reshape(Hs(:,:,1:T), 2*N, B*T).';
            if ~cfg.whu, dU(7*N+1:9*N, :) = 0; end
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
            idx = layer.Perm(:);
            [cw, sw] = rotation(om, cfg, S);
            top = 1:N; bot = N+1:2*N;

            h  = zeros(2*N, B, 'like', WX);
            cR = zeros(N*S, B, 'like', WX); cI = cR;
            switch cfg.output
                case 'h', Yd = zeros(2*N, B, T, 'like', WX);
                otherwise, Yd = zeros(N, B, T, 'like', WX);
            end
            if keep
                Zs = zeros(9*N, B, T, 'like', WX);
                Hs = zeros(2*N, B, T+1, 'like', WX);
                CR = zeros(N*S, B, T+1, 'like', WX); CI = CR;
            end
            for t = 1:T
                Z = WX(:,:,t) + U * h;
                f  = sig(Z(top,:)); ig = sig(Z(bot,:)); o = sig(Z(2*N+1:3*N,:));
                [kiR, kiI] = keyFwd(Z(3*N+1:4*N,:), Z(4*N+1:5*N,:), cfg.keys);
                [koR, koI] = keyFwd(Z(5*N+1:6*N,:), Z(6*N+1:7*N,:), cfg.keys);
                [uR, uI]   = boundFwd(Z(7*N+1:8*N,:), Z(8*N+1:9*N,:));
                vR = repmat(ig .* uR, S, 1); vI = repmat(ig .* uI, S, 1);
                kiRP = kiR(idx,:); kiIP = kiI(idx,:);
                koRP = koR(idx,:); koIP = koI(idx,:);
                fr = repmat(f, S, 1);
                if cfg.omega
                    aR = cw .* cR - sw .* cI; aI = sw .* cR + cw .* cI;
                else
                    aR = cR; aI = cI;
                end
                cR = fr .* aR + kiRP .* vR - kiIP .* vI;
                cI = fr .* aI + kiRP .* vI + kiIP .* vR;
                mR = copySum(koRP .* cR - koIP .* cI, N, S, B) / S;
                mI = copySum(koRP .* cI + koIP .* cR, N, S, B) / S;
                [bR, bI] = boundFwd(mR, mI);
                h = [o .* bR; o .* bI];
                switch cfg.output
                    case 'h',    Yd(:,:,t) = h;
                    case 'real', Yd(:,:,t) = h(top,:);
                    case 'mod',  Yd(:,:,t) = o .* modulus(bR, bI);
                end
                if keep
                    Zs(:,:,t) = Z;
                    Hs(:,:,t+1) = h;
                    CR(:,:,t+1) = cR; CI(:,:,t+1) = cI;
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

function v = raw(v)
if isa(v, 'dlarray'), v = extractdata(v); end
end

function y = sig(x)
y = 1 ./ (1 + exp(-x));
end

function s = copySum(A, N, S, B)
% A is [N*S x B], copies stacked in blocks of N rows -> sum over copies [N x B]
s = reshape(sum(reshape(A, N, S, B), 2), N, B);
end

function [cw, sw] = rotation(om, cfg, S)
if cfg.omega
    cw = repmat(cos(om), S, 1); sw = repmat(sin(om), S, 1);
else
    cw = []; sw = [];
end
end

function n = modulus(a, b)
% |z| smoothed at 0 so that its derivative stays finite
n = sqrt(a.^2 + b.^2 + 1e-12);
end

function [a, b] = boundFwd(a, b)
% Eq. 22: z / max(1, |z|)
d = max(1, sqrt(a.^2 + b.^2));
a = a ./ d; b = b ./ d;
end

function [a, b] = keyFwd(a, b, mode)
if strcmp(mode, 'bound')
    [a, b] = boundFwd(a, b);
else
    d = max(sqrt(a.^2 + b.^2), 1e-6);
    a = a ./ d; b = b ./ d;
end
end

function [ga, gb] = boundBack(a, b, ga, gb, mode)
% Jacobian of z/max(1,|z|) ('bound') or z/max(|z|,eps) ('unit'):
% identity inside the unit disc (bound), (I - zh zh') / |z| outside.
n = sqrt(a.^2 + b.^2);
if strcmp(mode, 'bound')
    act = n > 1;
else
    act = n > 1e-6;
end
ns = max(n, 1e-6);
ah = a ./ ns; bh = b ./ ns;
dt = ah .* ga + bh .* gb;
na = (ga - ah .* dt) ./ ns;
nb = (gb - bh .* dt) ./ ns;
if strcmp(mode, 'bound')
    ga = ga + act .* (na - ga);
    gb = gb + act .* (nb - gb);
else
    ga = act .* na + (~act) .* ga / 1e-6;
    gb = act .* nb + (~act) .* gb / 1e-6;
end
end
