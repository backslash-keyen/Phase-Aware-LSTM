classdef peepholeLSTMLayer < nnet.layer.Layer & nnet.layer.Formattable
    % peepholeLSTMLayer  LSTM with peephole connections (Gers, Schraudolph &
    % Schmidhuber 2002), configurable so every tracked version is one field.
    %
    % Rows of W, R and b are in lstmLayer order [i; f; z; o] (H rows each):
    %   a_i = W_i x_t + R_i h_{t-1} + p_i .* c_{t-1} + b_i,   i = sigmoid(a_i)
    %   a_f = W_f x_t + R_f h_{t-1} + p_f .* c_{t-1} + b_f,   f = sigmoid(a_f)
    %   a_z = W_z x_t + R_z h_{t-1}                  + b_z,   z = g(a_z)
    %   c_t = f .* c_{t-1} + i .* z                (CEC; CIFG: i = 1 - f)
    %   a_o = W_o x_t + R_o h_{t-1} + p_o .* c_t   + b_o,   o = sigmoid(a_o)
    %   h_t = o .* hOut(c_t)
    % (paper eqs. 1, 4, 5, 19-21: the output gate is updated after the cell, so
    % its peephole sees c_t; Cfg.outPeep = 'previous' makes it see c_{t-1}).
    % States start at zero for every sequence (paper Sec. 4.1: reset at the
    % start of each stream).
    %
    % Cfg fields (peepholeVersions.m):
    %   grad     'truncated'  paper Sec. 2.2/3.2: error reaches the past only
    %                         through the CEC (c_{t-1} -> c_t via f); nothing
    %                         flows back through R h_{t-1} or through the
    %                         peepholes into the cell (truncated RTRL, written
    %                         here as the equivalent truncated BPTT)
    %            'full'       exact gradient of the forward pass (BPTT)
    %   gFun     'identity' (paper experiments, Sec. 4.1) | 'tanh'
    %   hFun     'identity' (paper eq. 5: output squashing removed) | 'tanh'
    %   peep     true | false (false: traditional LSTM of Sec. 2; p fixed at 0)
    %   outPeep  'current' (paper) | 'previous'
    %   cifg     false | true (coupled input-forget gate, i = 1 - f)
    %   init     'gers'    every weight U[-0.1, 0.1] (peepholes and the cell
    %                      bias included); gate biases i 0, f -2, o +2
    %            'glorot'  lstmLayer defaults: Glorot W, orthogonal R,
    %                      unit-forget b (f +1), peepholes 0
    %   forgetBias  NaN = the init scheme's value, else this number
    % Forward and backward are vectorised per time step (CPU or gpuArray);
    % checkPeephole compares both gradient modes with dlgradient.

    properties
        NumHiddenUnits
        OutputMode
        VersionId
        Cfg
    end

    properties (Learnable)
        W
        R
        b
        p
    end

    methods
        function layer = peepholeLSTMLayer(numHiddenUnits, args)
            arguments
                numHiddenUnits
                args.Name = "";
                args.OutputMode = "sequence";
                args.Version = "gers2002";
                args.Cfg = [];
            end
            layer.NumHiddenUnits = numHiddenUnits;
            layer.Name        = args.Name;
            layer.OutputMode  = args.OutputMode;
            layer.VersionId   = char(args.Version);
            if isempty(args.Cfg)
                layer.Cfg = peepholeVersions(layer.VersionId);
            else
                layer.Cfg = args.Cfg;
            end
            layer.Description = "Peephole LSTM (" + layer.VersionId + ") with " + numHiddenUnits + " hidden units";
        end

        function layer = initialize(layer, layout)
            % Only empty learnables are drawn, so a trained layer is never re-drawn.
            cfg = layer.Cfg;
            H = layer.NumHiddenUnits;
            C = layout.Size(finddim(layout, "C"));
            switch cfg.init
                case 'gers'
                    u = @(sz) single(0.2*rand(sz) - 0.1);
                    if isempty(layer.W), layer.W = u([4*H, C]); end
                    if isempty(layer.R), layer.R = u([4*H, H]); end
                    if isempty(layer.b)
                        bz = u([H, 1]);
                        layer.b = [zeros(H,1,'single'); -2*ones(H,1,'single'); bz; 2*ones(H,1,'single')];
                    end
                    if isempty(layer.p), layer.p = u([3*H, 1]); end
                case 'glorot'
                    if isempty(layer.W), layer.W = single(initializeGlorot([4*H, C])); end
                    if isempty(layer.R), layer.R = single(initializeOrthogonal([4*H, H])); end
                    if isempty(layer.b), layer.b = single(initializeUnitForgetGate(4, H)); end
                    if isempty(layer.p), layer.p = zeros(3*H, 1, 'single'); end
                otherwise
                    error('peepholeLSTMLayer:init', 'Unknown init %s', cfg.init);
            end
            if ~isnan(cfg.forgetBias)
                bb = layer.b; bb(H+1:2*H) = cfg.forgetBias; layer.b = bb;
            end
            if ~cfg.peep, layer.p = zeros(size(layer.p), 'like', layer.p); end
            if cfg.cifg       % the input-gate rows are unused (i = 1 - f)
                WW = layer.W; WW(1:H,:) = 0; layer.W = WW;
                RR = layer.R; RR(1:H,:) = 0; layer.R = RR;
                bb = layer.b; bb(1:H) = 0; layer.b = bb;
                pp = layer.p; pp(1:H) = 0; layer.p = pp;
            end
        end

        function Y = predict(layer, X)
            Y = layer.run(X, false);
        end

        function [Y, memory] = forward(layer, X)
            [Y, memory] = layer.run(X, true);
        end

        function [dX, dW, dR, db, dp] = backward(layer, X, ~, dLdY, memory)
            cfg = layer.Cfg;
            H = layer.NumHiddenUnits;
            Xd = raw(stripdims(dlarray(X, 'CBT')));
            [C, B, T] = size(Xd, 1:3);
            W = raw(layer.W); R = raw(layer.R); p = raw(layer.p);
            pi_ = p(1:H); pf = p(H+1:2*H); po = p(2*H+1:3*H);
            dY = raw(stripdims(dLdY));
            full = strcmp(cfg.grad, 'full');
            isSeq = layer.OutputMode == "sequence";
            curO = strcmp(cfg.outPeep, 'current');
            ri = 1:H; rf = H+1:2*H; rz = 2*H+1:3*H; ro = 3*H+1:4*H;

            As = memory.A; Cs = memory.C; Hs = memory.H;     % A: 4H x B x T, C/H: H x B x (T+1)
            dA  = zeros(4*H, B, T, 'like', Xd);
            dhR = zeros(H, B, 'like', Xd);   % dL/dh_t from step t+1 (full only)
            dcN = zeros(H, B, 'like', Xd);   % dL/dc_t from step t+1
            for t = T:-1:1
                if isSeq
                    dh = dY(:,:,t);
                elseif t == T
                    dh = dY;
                else
                    dh = zeros(H, B, 'like', Xd);
                end
                if full, dh = dh + dhR; end
                a = As(:,:,t);
                f = sig(a(rf,:));
                if cfg.cifg, ig = 1 - f; else, ig = sig(a(ri,:)); end
                o = sig(a(ro,:));
                [z, dzda] = actFun(cfg.gFun, a(rz,:));
                cp = Cs(:,:,t); c = Cs(:,:,t+1);
                [hc, dhc] = actFun(cfg.hFun, c);

                dao = dh .* hc .* o .* (1 - o);
                dc  = dh .* o .* dhc + dcN;
                if full && cfg.peep && curO, dc = dc + dao .* po; end
                daz = dc .* ig .* dzda;
                dig = dc .* z;
                df  = dc .* cp;
                if cfg.cifg
                    df  = df - dig;
                    dai = zeros(H, B, 'like', Xd);
                else
                    dai = dig .* ig .* (1 - ig);
                end
                daf = df .* f .* (1 - f);
                dA(:,:,t) = [dai; daf; daz; dao];

                dcN = dc .* f;                      % the CEC (kept in both modes)
                if full
                    dhR = R.' * dA(:,:,t);
                    if cfg.peep
                        dcN = dcN + dai .* pi_ + daf .* pf;
                        if ~curO, dcN = dcN + dao .* po; end
                    end
                end
            end

            dA2 = reshape(dA, 4*H, B*T);
            dW = dA2 * reshape(Xd, C, B*T).';
            dR = dA2 * reshape(Hs(:,:,1:T), H, B*T).';
            db = sum(dA2, 2);
            cPrev = reshape(Cs(:,:,1:T), H, B*T);
            if curO, cO = reshape(Cs(:,:,2:T+1), H, B*T); else, cO = cPrev; end
            dp = [sum(dA2(ri,:) .* cPrev, 2); sum(dA2(rf,:) .* cPrev, 2); sum(dA2(ro,:) .* cO, 2)];
            if ~cfg.peep, dp(:) = 0; end
            if cfg.cifg
                dW(ri,:) = 0; dR(ri,:) = 0; db(ri) = 0; dp(1:H) = 0;
            end
            dX = pagemtimes(W.', dA);
            if isa(X, 'dlarray') && ~isempty(dims(X))
                dX = dlarray(dX, dims(X));
            else
                dX = dlarray(dX);
            end
            % trainNetwork expects dlarray outputs from a custom backward
            dW = dlarray(dW); dR = dlarray(dR); db = dlarray(db); dp = dlarray(dp);
        end
    end

    methods (Access = private)
        function [Y, memory] = run(layer, X, keep)
            cfg = layer.Cfg;
            H = layer.NumHiddenUnits;
            Xd = raw(stripdims(dlarray(X, 'CBT')));
            [~, B, T] = size(Xd, 1:3);
            W = raw(layer.W); R = raw(layer.R); b = raw(layer.b); p = raw(layer.p);
            pi_ = p(1:H); pf = p(H+1:2*H); po = p(2*H+1:3*H);
            isSeq = layer.OutputMode == "sequence";
            if ~isSeq && layer.OutputMode ~= "last"
                error("OutputMode must be 'sequence' or 'last'.");
            end
            curO = strcmp(cfg.outPeep, 'current');
            ri = 1:H; rf = H+1:2*H; rz = 2*H+1:3*H; ro = 3*H+1:4*H;

            WX = pagemtimes(W, Xd) + b;
            h = zeros(H, B, 'like', WX); c = h;
            if keep
                As = zeros(4*H, B, T, 'like', WX);
                Cs = zeros(H, B, T+1, 'like', WX);
                Hs = Cs;
            end
            if isSeq, Yd = zeros(H, B, T, 'like', WX); end
            for t = 1:T
                a = WX(:,:,t) + R * h;
                if cfg.peep
                    a(ri,:) = a(ri,:) + pi_ .* c;
                    a(rf,:) = a(rf,:) + pf .* c;
                    if ~curO, a(ro,:) = a(ro,:) + po .* c; end
                end
                f = sig(a(rf,:));
                if cfg.cifg, ig = 1 - f; else, ig = sig(a(ri,:)); end
                z = actFun(cfg.gFun, a(rz,:));
                c = f .* c + ig .* z;
                if cfg.peep && curO, a(ro,:) = a(ro,:) + po .* c; end
                o = sig(a(ro,:));
                h = o .* actFun(cfg.hFun, c);
                if isSeq, Yd(:,:,t) = h; end
                if keep
                    As(:,:,t) = a; Cs(:,:,t+1) = c; Hs(:,:,t+1) = h;
                end
            end
            if isSeq, Y = dlarray(Yd, 'CBT'); else, Y = dlarray(h, 'CB'); end
            if keep
                memory = struct('A', As, 'C', Cs, 'H', Hs);
            else
                memory = [];
            end
        end
    end
end

function y = sig(x)
y = 1 ./ (1 + exp(-x));
end

function [y, d] = actFun(name, x)
switch name
    case 'identity'
        y = x;
        if nargout > 1, d = ones(size(x), 'like', x); end
    case 'tanh'
        y = tanh(x);
        if nargout > 1, d = 1 - y.^2; end
    otherwise
        error('peepholeLSTMLayer:act', 'Unknown activation %s', name);
end
end

function v = raw(v)
if isa(v, 'dlarray'), v = extractdata(v); end
end
