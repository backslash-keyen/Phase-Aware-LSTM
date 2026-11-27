classdef phaseAwareLSTMCorr < nnet.layer.Layer & nnet.layer.Formattable
    % phaseAwareLSTMCorr  Paper phase-aware LSTM (notrig, 3 Nov 2025) with opt-in corrections.
    %
    % With the default configuration this is the layer behind the paper's
    % CWL phase-lstm results, L:\plstm\helperLibrary\modelframe\
    % phaseAwareLSTM_notrig.m (copy: replication/variants/notrigPhaseLSTM),
    % call for call in initialize() and step for step in the recurrence:
    %   G = clip5(W1 x + b1 + U h),  P = clip5(W2 x + b2 + Q z)
    %   f, i, o = sigmoid(G_f, G_i, G_o),  g = tanh(G_g) + 1i tanh(P_g)
    %   theta_k = atan(P_k) (written as cos = 1/sqrt(1+P^2), sin = P cos)
    %   c = f e^{i theta_f} c_prev + i e^{i theta_i} g
    %   h = o .* tanh(|c|),  z = atan2(Im c, Re c)          (fed back)
    %   b1 = b2 = unit-forget (+1 on rows H+1:2H, i.e. the gate i)
    % Each correction is one field of the configuration (phaseCorrConfig).
    % A version = the defaults plus the fields it names
    % (replication/phaseLayer/phaseCorrVersions.m):
    %   gradCap    finite: every learnable gradient is rescaled in double to
    %              L2 norm <= gradCap (Inf/NaN -> 0) before trainNetwork sees
    %              it. trainNetwork's own per-learnable l2norm clip is computed
    %              in single; the phase layer's squared norm (~1e50) overflows,
    %              the clip factor becomes 0 and the layer never trains.
    %   forgetRow  'i' (paper) | 'f': which gate gets the unit-forget +1 of b1
    %   phaseBias  'unitForget' (paper, theta_i starts at atan(1) = 45 deg) | 'zero'
    %   feedback   what the phase gates see of the previous step:
    %              'atan2'    z = angle(c)               [H]  (paper)
    %              'cosSin'   z = [cos; sin] angle(c)    [2H] (no branch cut)
    %              'complexH' z = h e^{i angle(c)} = o tanh(|c|) c/|c|  [2H]
    %                         (the complex hidden state; bounded and smooth,
    %                         no 1/|c| in its derivative)
    %   recScale   scale of the U and Q draws (1 = paper)
    % Rows 3H+1:4H of W2, Q and b2 (the phase "output gate") are unused, as in
    % the paper layer; their gradients are zero.
    % Forward and backward are vectorised per time step (CPU or gpuArray),
    % gradients hand-written; checkPhaseCorr compares them with dlgradient of
    % a plain dlarray implementation and the default config with the
    % original layer.

    properties
        NumHiddenUnits
        OutputMode
        VersionId
        Cfg
    end

    properties (Learnable)
        W1
        W2
        U
        Q
        b1
        b2
    end

    properties (State)
        hiddenMagnitude
        hiddenPhase
        memoryReal
        memoryImaginary
    end

    methods
        function layer = phaseAwareLSTMCorr(numHiddenUnits, args)
            arguments
                numHiddenUnits
                args.Name = "";
                args.OutputMode = "sequence";
                args.Version = "port";
                args.Cfg = [];
            end
            layer.NumHiddenUnits = numHiddenUnits;
            layer.Name           = args.Name;
            layer.OutputMode     = args.OutputMode;
            layer.VersionId        = char(args.Version);
            if isempty(args.Cfg)
                layer.Cfg = phaseCorrVersions(layer.VersionId);
            else
                layer.Cfg = args.Cfg;
            end
            layer.Description = "Phase-aware LSTM (" + layer.VersionId + ") with " + numHiddenUnits + " hidden units";
        end

        function layer = initialize(layer, layout)
            % Paper draws first, in the paper's order (same RNG stream); a
            % second Q block for the [2H] feedback modes is drawn last.
            cfg = layer.Cfg;
            H   = layer.NumHiddenUnits;
            C   = layout.Size(finddim(layout, "C"));
            if isempty(layer.W1), layer.W1 = initializeGlorot([4*H, C]); end
            if isempty(layer.W2), layer.W2 = initializeGlorot([4*H, C]); end
            if isempty(layer.U)
                layer.U = cfg.recScale * initializeOrthogonal([4*H, H]);
            end
            newQ = isempty(layer.Q);
            if newQ
                layer.Q = cfg.recScale * initializeOrthogonal([4*H, H]);
            end
            if isempty(layer.b1)
                b = initializeUnitForgetGate(4, H);          % +1 on rows H+1:2H
                if strcmp(cfg.forgetRow, 'f')
                    b = zeros(size(b), 'like', b); b(1:H) = 1;
                end
                layer.b1 = b;
            end
            if isempty(layer.b2)
                b = initializeUnitForgetGate(4, H);
                if strcmp(cfg.phaseBias, 'zero'), b = zeros(size(b), 'like', b); end
                layer.b2 = b;
            end
            if newQ && feedbackWidth(cfg) == 2
                layer.Q = [layer.Q, cfg.recScale * initializeOrthogonal([4*H, H])];
            end
        end

        function [Y, hS, phS, cRS, cIS] = predict(layer, X)
            [Y, hS, phS, cRS, cIS] = layer.run(X, false);
        end

        function [Y, hS, phS, cRS, cIS, memory] = forward(layer, X)
            [Y, hS, phS, cRS, cIS, memory] = layer.run(X, true);
        end

        function varargout = backward(layer, X, varargin)
            % backward(layer, X, Y, dLdY, memory)
            % backward(layer, X, Y, dLdY, dLdS1..dLdS4, memory)
            % backward(layer, X, Y, S1..S4, dLdY, dLdS1..dLdS4, memory)
            % State derivatives are ignored: states restart from zero at
            % every call (as in the paper layer), so they carry no gradient.
            memory = varargin{end};
            rest   = varargin(1:end-1);
            switch numel(rest)
                case {2, 6}, dLdY = rest{2};
                case 10,     dLdY = rest{6};
                otherwise
                    error('phaseAwareLSTMCorr:backwardSyntax', 'Unexpected backward call with %d arguments.', nargin);
            end
            cfg = layer.Cfg;
            H   = layer.NumHiddenUnits;
            Xd  = raw(stripdims(dlarray(X, 'CBT')));
            [C, B, T] = size(Xd, 1:3);
            W1 = raw(layer.W1); W2 = raw(layer.W2); U = raw(layer.U); Q = raw(layer.Q);
            zw = feedbackWidth(cfg) * H;
            Q3 = Q(1:3*H, :);
            dY = raw(stripdims(dLdY));
            clip = cast(cfg.clip, 'like', Xd);
            epsl = cast(1e-9, 'like', Xd);
            isSeq = layer.OutputMode == "sequence";

            Gs = memory.G; Ps = memory.P; Hs = memory.h; Zs = memory.z;
            CRs = memory.cR; CIs = memory.cI;

            dh  = zeros(H, B, 'like', Xd);
            dz  = zeros(zw, B, 'like', Xd);
            dcR = zeros(H, B, 'like', Xd);
            dcI = dcR;
            dGs = zeros(4*H, B, T, 'like', Xd);
            dPs = zeros(3*H, B, T, 'like', Xd);
            top = 1:H; bot = H+1:2*H;

            for t = T:-1:1
                if isSeq
                    dh = dh + dY(:,:,t);
                elseif t == T
                    dh = dh + dY;
                end
                G = Gs(:,:,t); P = Ps(:,:,t);
                sg = 1 ./ (1 + exp(-G));
                f  = sg(top,:); ig = sg(bot,:); o = sg(3*H+1:4*H,:);
                gR = tanh(G(2*H+1:3*H,:));
                gI = tanh(P(2*H+1:3*H,:));
                Pa = P(1:2*H,:);
                cs = 1 ./ sqrt(1 + Pa.^2);
                sn = Pa .* cs;
                cRp = CRs(:,:,t);  cIp = CIs(:,:,t);
                cR  = CRs(:,:,t+1); cI = CIs(:,:,t+1);
                rho  = sqrt(cR.^2 + cI.^2);
                tm   = tanh(max(rho, epsl));

                % h = o tanh(max(|c|, eps))
                do   = dh .* tm;
                dmag = dh .* o .* (1 - tm.^2) .* (rho > epsl);
                rinv = 1 ./ max(rho, realmin('like', rho));
                dcR = dcR + dmag .* cR .* rinv;
                dcI = dcI + dmag .* cI .* rinv;

                % fed-back z
                switch cfg.feedback
                    case 'atan2'
                        r2 = rinv.^2;
                        dcR = dcR - dz .* cI .* r2;
                        dcI = dcI + dz .* cR .* r2;
                    case 'cosSin'
                        r3 = 1 ./ max(rho, 1e-12).^3;      % finite at |c| = 0 (single)
                        dzc = dz(top,:); dzs = dz(bot,:);
                        q = (dzc .* cI - dzs .* cR) .* r3;
                        dcR = dcR + cI .* q;
                        dcI = dcI - cR .* q;
                    case 'complexH'
                        [g, gp] = tanhOverR(rho);            % g = tanh(r)/r, gp = g'(r)/r
                        dzR = dz(top,:); dzI = dz(bot,:);
                        do  = do + g .* (dzR .* cR + dzI .* cI);
                        s   = gp .* (dzR .* cR + dzI .* cI);
                        dcR = dcR + o .* (g .* dzR + s .* cR);
                        dcI = dcI + o .* (g .* dzI + s .* cI);
                end

                % c = sum over [prev; cand] of m e^{i theta} (x + i y)
                m  = [f; ig];
                x  = [cRp; gR];
                y  = [cIp; gI];
                A  = m .* x;
                Bv = m .* y;
                dxN = [dcR; dcR];
                dyN = [dcI; dcI];
                dA = dxN .* cs + dyN .* sn;
                dB = dyN .* cs - dxN .* sn;
                dc = dxN .* A + dyN .* Bv;
                ds = dyN .* A - dxN .* Bv;
                dm = dA .* x + dB .* y;
                dx = dA .* m;
                dy = dB .* m;
                dPa = cs.^3 .* (ds - Pa .* dc);

                dG = [dm(top,:) .* f .* (1 - f); ...
                      dm(bot,:) .* ig .* (1 - ig); ...
                      dx(bot,:) .* (1 - gR.^2); ...
                      do .* o .* (1 - o)];
                dP = [dPa; dy(bot,:) .* (1 - gI.^2)];
                dG = dG .* (abs(G) < clip);
                dP = dP .* (abs(P) < clip);
                dGs(:,:,t) = dG;
                dPs(:,:,t) = dP;

                dh  = U.'  * dG;
                dz  = Q3.' * dP;
                dcR = dx(top,:);
                dcI = dy(top,:);
            end

            dG2 = reshape(dGs, 4*H, B*T);
            dP2 = reshape(dPs, 3*H, B*T);
            X2  = reshape(Xd, C, B*T);
            dW1 = dG2 * X2.';
            dW2 = [dP2 * X2.'; zeros(H, C, 'like', dP2)];
            dU  = dG2 * reshape(Hs(:,:,1:T), H, B*T).';
            dQ  = [dP2 * reshape(Zs(:,:,1:T), zw, B*T).'; zeros(H, zw, 'like', dP2)];
            db1 = sum(dG2, 2);
            db2 = [sum(dP2, 2); zeros(H, 1, 'like', dP2)];
            dX  = pagemtimes(W1.', dGs) + pagemtimes(W2(1:3*H,:).', dPs);

            grads = {dW1, dW2, dU, dQ, db1, db2};
            if isfinite(cfg.gradCap)
                grads = phaseCorrCapGrads(grads, cfg.gradCap);
            else
                phaseCorrCapGrads(grads, Inf);              % record norms only
            end
            out = [{dX}, grads];
            for k = numel(out)+1:nargout
                out{k} = zeros(H, B, 'like', Xd);
            end
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
            cfg = layer.Cfg;
            H  = layer.NumHiddenUnits;
            Xd = raw(stripdims(dlarray(X, 'CBT')));
            [~, B, T] = size(Xd, 1:3);
            W1 = raw(layer.W1); W2 = raw(layer.W2);
            U  = raw(layer.U);  Q  = raw(layer.Q);
            b1 = raw(layer.b1); b2 = raw(layer.b2);
            W2 = W2(1:3*H,:); Q3 = Q(1:3*H,:); b2 = b2(1:3*H);
            zw = feedbackWidth(cfg) * H;
            clip = cast(cfg.clip, 'like', Xd);
            epsl = cast(1e-9, 'like', Xd);
            isSeq = layer.OutputMode == "sequence";
            if ~isSeq && layer.OutputMode ~= "last"
                error("OutputMode must be 'sequence' or 'last'.");
            end

            WX1 = pagemtimes(W1, Xd) + b1;
            WX2 = pagemtimes(W2, Xd) + b2;
            h  = zeros(H, B, 'like', WX1); cR = h; cI = h; ph = h;
            z  = zeros(zw, B, 'like', WX1);
            if keep
                Gs = zeros(4*H, B, T, 'like', WX1);
                Ps = zeros(3*H, B, T, 'like', WX1);
                Hs = zeros(H, B, T+1, 'like', WX1);
                Zs = zeros(zw, B, T+1, 'like', WX1);
                CRs = Hs; CIs = Hs;
            end
            if isSeq, Yd = zeros(H, B, T, 'like', WX1); end
            top = 1:H; bot = H+1:2*H;

            for t = 1:T
                G = max(min(WX1(:,:,t) + U  * h, clip), -clip);
                P = max(min(WX2(:,:,t) + Q3 * z, clip), -clip);
                sg = 1 ./ (1 + exp(-G));
                gR = tanh(G(2*H+1:3*H,:));
                gI = tanh(P(2*H+1:3*H,:));
                cs = 1 ./ sqrt(1 + P(1:2*H,:).^2);
                sn = P(1:2*H,:) .* cs;
                m  = sg(1:2*H,:);
                A  = m .* [cR; gR];
                Bv = m .* [cI; gI];
                xN = A .* cs - Bv .* sn;
                yN = A .* sn + Bv .* cs;
                cR = xN(top,:) + xN(bot,:);
                cI = yN(top,:) + yN(bot,:);
                rho = sqrt(cR.^2 + cI.^2);
                o   = sg(3*H+1:4*H,:);
                h   = o .* tanh(max(rho, epsl));
                ph  = atan2(cI, cR);
                switch cfg.feedback
                    case 'atan2'
                        z = ph;
                    case 'cosSin'
                        ri = 1 ./ max(rho, realmin('like', rho));
                        z = [cR .* ri; cI .* ri];
                        z(top,:) = z(top,:) + (rho == 0);   % angle(0) = 0 -> cos 1, sin 0
                    case 'complexH'
                        g = tanhOverR(rho);
                        z = [o .* g .* cR; o .* g .* cI];
                end
                if isSeq, Yd(:,:,t) = h; end
                if keep
                    Gs(:,:,t) = G; Ps(:,:,t) = P;
                    Hs(:,:,t+1) = h; Zs(:,:,t+1) = z;
                    CRs(:,:,t+1) = cR; CIs(:,:,t+1) = cI;
                end
            end
            hS = dlarray(h); phS = dlarray(ph); cRS = dlarray(cR); cIS = dlarray(cI);
            if isSeq
                Y = dlarray(Yd, 'CBT');
            else
                Y = dlarray(h, 'CB');
            end
            if keep
                memory = struct('G', Gs, 'P', Ps, 'h', Hs, 'z', Zs, 'cR', CRs, 'cI', CIs);
            else
                memory = [];
            end
        end
    end
end

function w = feedbackWidth(cfg)
if strcmp(cfg.feedback, 'atan2'), w = 1; else, w = 2; end
end

function v = raw(v)
if isa(v, 'dlarray'), v = extractdata(v); end
end

function [g, gp] = tanhOverR(r)
% g = tanh(r)/r and gp = g'(r)/r = ((1 - tanh^2 r) r - tanh r)/r^3,
% with their series below r = 1e-2 (g -> 1, gp -> -2/3).
small = r < 1e-2;
rs = max(r, 1e-2);
t  = tanh(rs);
g  = t ./ rs;
r2 = r.^2;
g(small) = 1 - r2(small)/3 + 2*r2(small).^2/15;
if nargout > 1
    gp = ((1 - t.^2) .* rs - t) ./ rs.^3;
    gp(small) = -2/3 + 8*r2(small)/15;
end
end
