classdef phaseAwareLSTMIdeal < nnet.layer.Layer & nnet.layer.Formattable
    % phaseAwareLSTMIdeal  Complex LSTM designed from the literature (replication/phaseLayer/IDEAL_LAYER.md).
    %
    % Per unit, complex memory c (Re, Im), complex hidden h, real input x,
    % z = [Re h; Im h] fed back:
    %   A = Wx x + Uz z_{t-1} + b,  rows f i o uR uI r (6H)
    %   f, i, o = sigmoid(A_f, A_i, A_o)                 real gates (Wolter & Yao 2018)
    %   u = bound(A_uR + 1i A_uI), bound(v) = tanh(|v|) v/|v|   (Hirose; Danihelka 2016)
    %   theta = omega + pi tanh(A_r)  ('piTanh')  |  omega + atan(A_r) ('atan')
    %   c = f e^{i theta} c_{t-1} + s i u,  s = 1 - f ('coupled') | 1 ('plain')
    %       |eigenvalue| = f < 1 (decay, ceRNN); coupled keeps |c| <= 1 (cgRNN eq. 10)
    %   h = o bound(c)
    %   output 'mag':     Y = |h| = o tanh(|c|)         (the paper layer's interface)
    %          'readout': Y = Wo [Re h; Im h] + bo       (Wolter & Yao eq. 7)
    % omega: per-unit frequency in rad/sample, init 2 pi f/250 with f
    % log-spaced over 1-40 Hz ('omega' true) or 0 and not trained (false).
    % b: +1 on the forget rows. No atan2, no pre-activation clip, no 1/|c|.
    % gradCap (double-precision per-learnable cap) as in phaseAwareLSTMCorr.
    properties
        NumHiddenUnits
        OutputMode
        VersionId
        Cfg
    end
    properties (Learnable)
        Wx
        Uz
        b
        omega
        Wo
        bo
    end
    properties (State)
        hiddenMagnitude
        hiddenPhase
        memoryReal
        memoryImaginary
    end

    methods
        function layer = phaseAwareLSTMIdeal(numHiddenUnits, args)
            arguments
                numHiddenUnits
                args.Name = "";
                args.OutputMode = "sequence";
                args.Version = "ideal";
                args.Cfg = [];
            end
            layer.NumHiddenUnits = numHiddenUnits;
            layer.Name = args.Name;
            layer.OutputMode = args.OutputMode;
            layer.VersionId = char(args.Version);
            if isempty(args.Cfg), layer.Cfg = phaseCorrVersions(layer.VersionId); else, layer.Cfg = args.Cfg; end
            layer.Description = "Ideal complex LSTM (" + layer.VersionId + ") with " + numHiddenUnits + " hidden units";
        end

        function layer = initialize(layer, layout)
            cfg = layer.Cfg; H = layer.NumHiddenUnits;
            C = layout.Size(finddim(layout, "C"));
            if isempty(layer.Wx), layer.Wx = initializeGlorot([6*H, C]); end
            if isempty(layer.Uz), layer.Uz = initializeOrthogonal([6*H, 2*H]); end
            if isempty(layer.b)
                b = zeros(6*H, 1, 'single'); b(1:H) = 1; layer.b = b;
            end
            if isempty(layer.omega)
                if cfg.omega
                    layer.omega = single(2*pi*logspace(0, log10(40), H)' / cfg.fs);
                else
                    layer.omega = zeros(H, 1, 'single');
                end
            end
            if isempty(layer.Wo)
                if strcmp(cfg.output, 'readout')
                    layer.Wo = initializeGlorot([H, 2*H]);
                else
                    layer.Wo = zeros(H, 2*H, 'single');    % unused (zero gradient)
                end
            end
            if isempty(layer.bo), layer.bo = zeros(H, 1, 'single'); end
        end

        function [Y, s1, s2, s3, s4] = predict(layer, X)
            [Y, s1, s2, s3, s4] = layer.run(X, false);
        end

        function [Y, s1, s2, s3, s4, memory] = forward(layer, X)
            [Y, s1, s2, s3, s4, memory] = layer.run(X, true);
        end

        function varargout = backward(layer, X, varargin)
            memory = varargin{end};
            rest = varargin(1:end-1);
            switch numel(rest)
                case {2, 6}, dLdY = rest{2};
                case 10,     dLdY = rest{6};
                otherwise, error('phaseAwareLSTMIdeal:backwardSyntax', 'Unexpected backward call.');
            end
            cfg = layer.Cfg; H = layer.NumHiddenUnits;
            Xd = raw(stripdims(dlarray(X, 'CBT')));
            [C, B, T] = size(Xd, 1:3);
            Wx = raw(layer.Wx); Uz = raw(layer.Uz); Wo = raw(layer.Wo);
            dY = raw(stripdims(dLdY));
            isSeq = layer.OutputMode == "sequence";
            readout = strcmp(cfg.output, 'readout');
            coupled = strcmp(cfg.inputScale, 'coupled');
            As = memory.A; Zs = memory.z; CRs = memory.cR; CIs = memory.cI;
            om = raw(layer.omega);
            r1 = 1:H; r2 = H+1:2*H; r3 = 2*H+1:3*H; r4 = 3*H+1:4*H; r5 = 4*H+1:5*H; r6 = 5*H+1:6*H;

            dz  = zeros(2*H, B, 'like', Xd);
            dcR = zeros(H, B, 'like', Xd); dcI = dcR;
            dAs = zeros(6*H, B, T, 'like', Xd);
            dWo = zeros(H, 2*H, 'like', Xd); dbo = zeros(H, 1, 'like', Xd);
            domega = zeros(H, 1, 'like', Xd);
            for t = T:-1:1
                A = As(:,:,t);
                f = sig(A(r1,:)); ig = sig(A(r2,:)); o = sig(A(r3,:));
                vR = A(r4,:); vI = A(r5,:);
                [gv, gpv] = tanhOverR(sqrt(vR.^2 + vI.^2));
                uR = gv .* vR; uI = gv .* vI;
                [th, dthdA] = rotAngle(A(r6,:), om, cfg.rotMap);
                Cth = cos(th); Sth = sin(th);
                cRp = CRs(:,:,t); cIp = CIs(:,:,t);
                cR = CRs(:,:,t+1); cI = CIs(:,:,t+1);
                rho = sqrt(cR.^2 + cI.^2);
                [g, gp] = tanhOverR(rho);
                zt = Zs(:,:,t+1);

                % output
                if isSeq, dy = dY(:,:,t); elseif t == T, dy = dY; else, dy = []; end
                dhR = dz(r1,:); dhI = dz(r2,:);
                do = zeros(H, B, 'like', Xd);
                if ~isempty(dy)
                    if readout
                        dzy = Wo.' * dy;
                        dhR = dhR + dzy(r1,:); dhI = dhI + dzy(r2,:);
                        dWo = dWo + dy * zt.'; dbo = dbo + sum(dy, 2);
                    else
                        tm = tanh(rho);
                        do = do + dy .* tm;
                        drho = dy .* o .* (1 - tm.^2);
                        ri = 1 ./ max(rho, 1e-12);
                        dcR = dcR + drho .* cR .* ri;
                        dcI = dcI + drho .* cI .* ri;
                    end
                end
                % h = o g(rho) c
                q = gp .* (dhR .* cR + dhI .* cI);
                do  = do + g .* (dhR .* cR + dhI .* cI);
                dcR = dcR + o .* (g .* dhR + q .* cR);
                dcI = dcI + o .* (g .* dhI + q .* cI);
                % c = f e^{i th} cp + s i u
                rR = Cth .* cRp - Sth .* cIp;
                rI = Sth .* cRp + Cth .* cIp;
                if coupled, s = 1 - f; else, s = ones(size(f), 'like', f); end
                du = dcR .* uR + dcI .* uI;
                df = dcR .* rR + dcI .* rI;
                if coupled, df = df - ig .* du; end
                di = s .* du;
                duR = s .* ig .* dcR; duI = s .* ig .* dcI;
                dth = f .* (dcI .* rR - dcR .* rI);
                domega = domega + sum(dth, 2);
                dcRp = f .* (Cth .* dcR + Sth .* dcI);
                dcIp = f .* (-Sth .* dcR + Cth .* dcI);
                % u = g(|v|) v
                qv = gpv .* (duR .* vR + duI .* vI);
                dvR = gv .* duR + qv .* vR;
                dvI = gv .* duI + qv .* vI;
                dA = [df .* f .* (1 - f); di .* ig .* (1 - ig); do .* o .* (1 - o); dvR; dvI; dth .* dthdA];
                dAs(:,:,t) = dA;
                dz = Uz.' * dA;
                dcR = dcRp; dcI = dcIp;
            end
            dA2 = reshape(dAs, 6*H, B*T);
            dWx = dA2 * reshape(Xd, C, B*T).';
            dUz = dA2 * reshape(Zs(:,:,1:T), 2*H, B*T).';
            db  = sum(dA2, 2);
            if ~cfg.omega, domega = zeros(H, 1, 'like', Xd); end   % omega fixed at 0
            dX = pagemtimes(Wx.', dAs);
            grads = {dWx, dUz, db, domega, dWo, dbo};
            if isfinite(cfg.gradCap)
                grads = phaseCorrCapGrads(grads, cfg.gradCap);
            else
                phaseCorrCapGrads(grads, Inf);
            end
            out = [{dX}, grads];
            for k = numel(out)+1:nargout, out{k} = zeros(H, B, 'like', Xd); end
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
        function [Y, hS, phS, cRS, cIS, memory] = run(layer, X, keep)
            cfg = layer.Cfg; H = layer.NumHiddenUnits;
            Xd = raw(stripdims(dlarray(X, 'CBT')));
            [~, B, T] = size(Xd, 1:3);
            Wx = raw(layer.Wx); Uz = raw(layer.Uz); b = raw(layer.b);
            om = raw(layer.omega); Wo = raw(layer.Wo); bo = raw(layer.bo);
            isSeq = layer.OutputMode == "sequence";
            readout = strcmp(cfg.output, 'readout');
            coupled = strcmp(cfg.inputScale, 'coupled');
            WX = pagemtimes(Wx, Xd) + b;
            z = zeros(2*H, B, 'like', WX); cR = zeros(H, B, 'like', WX); cI = cR;
            if keep
                As = zeros(6*H, B, T, 'like', WX);
                Zs = zeros(2*H, B, T+1, 'like', WX);
                CRs = zeros(H, B, T+1, 'like', WX); CIs = CRs;
            end
            if isSeq, Yd = zeros(H, B, T, 'like', WX); end
            r1 = 1:H; r2 = H+1:2*H; r3 = 2*H+1:3*H; r4 = 3*H+1:4*H; r5 = 4*H+1:5*H; r6 = 5*H+1:6*H;
            for t = 1:T
                A = WX(:,:,t) + Uz * z;
                f = sig(A(r1,:)); ig = sig(A(r2,:)); o = sig(A(r3,:));
                vR = A(r4,:); vI = A(r5,:);
                gv = tanhOverR(sqrt(vR.^2 + vI.^2));
                th = rotAngle(A(r6,:), om, cfg.rotMap);
                Cth = cos(th); Sth = sin(th);
                if coupled, s = 1 - f; else, s = 1; end
                nR = f .* (Cth .* cR - Sth .* cI) + s .* ig .* gv .* vR;
                nI = f .* (Sth .* cR + Cth .* cI) + s .* ig .* gv .* vI;
                cR = nR; cI = nI;
                rho = sqrt(cR.^2 + cI.^2);
                g = tanhOverR(rho);
                z = [o .* g .* cR; o .* g .* cI];
                if readout, y = Wo * z + bo; else, y = o .* tanh(rho); end
                if isSeq, Yd(:,:,t) = y; end
                if keep
                    As(:,:,t) = A; Zs(:,:,t+1) = z; CRs(:,:,t+1) = cR; CIs(:,:,t+1) = cI;
                end
            end
            hS = dlarray(o .* tanh(rho)); phS = dlarray(atan2(cI, cR));
            cRS = dlarray(cR); cIS = dlarray(cI);
            if isSeq, Y = dlarray(Yd, 'CBT'); else, Y = dlarray(y, 'CB'); end
            if keep
                memory = struct('A', As, 'z', Zs, 'cR', CRs, 'cI', CIs);
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

function [th, d] = rotAngle(a, om, map)
if strcmp(map, 'piTanh')
    ta = tanh(a); th = om + pi * ta; d = pi * (1 - ta.^2);
else
    th = om + atan(a); d = 1 ./ (1 + a.^2);
end
end

function [g, gp] = tanhOverR(r)
% g = tanh(r)/r, gp = g'(r)/r, series below r = 1e-2.
small = r < 1e-2;
rs = max(r, 1e-2);
t = tanh(rs);
g = t ./ rs;
r2 = r.^2;
g(small) = 1 - r2(small)/3 + 2*r2(small).^2/15;
if nargout > 1
    gp = ((1 - t.^2) .* rs - t) ./ rs.^3;
    gp(small) = -2/3 + 8*r2(small)/15;
end
end
