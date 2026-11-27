classdef phasedLSTMCfg < nnet.layer.Layer & nnet.layer.Formattable
    % Phased LSTM (Neil, Pfeiffer and Liu 2016) with every difference between the
    % Nov 2025 layer (phasedLSTM.m) and the paper as a setting, and a hand-written
    % backward pass (one matrix product for the input and recurrent weight
    % gradients instead of 500 autodiff steps).
    %
    % Settings (a phasedVersions struct, property Cfg):
    %   gate       'nov2025'  k = phi/r (phi < 0.1 r), 3 - phi/r (phi < r), alpha*phi
    %              'neil'     k = 2 phi/r (phi < r/2), 2 - 2 phi/r (phi < r), alpha*phi  (Eq. 6)
    %   ron        r_on at init (per unit); ronLearn: learned (else learn-rate factor 0)
    %   leakTrain  alpha in forward (training); leakTest: alpha in predict (validation, test)
    %   tauLog     tau = exp(U(tauLog(1), tauLog(2))); s = U(0,1) .* tau
    %   timingGrad true: tau, s (and r_on) get their gradients; false: zero, as in
    %              Nov 2025, where only L2 weight decay moves them
    %   timingL2   L2 factor of tau and s (set by the builder)
    %   peephole   w_ci, w_cf on c_{t-1}, w_co on the proposed c~_t (Neil Eqs. 1, 2, 4)
    %
    % Model, per unit and time step t = 1..T (the sample index; 250 Hz CWL data):
    %   phi = ((t - s) mod tau) / tau,  k = gate(phi)
    %   [i f g o] from W x_t + U h_{t-1} + b (+ peepholes), c~ = f c_{t-1} + i g,
    %   h~ = o tanh(c~), c_t = k c~ + (1 - k) c_{t-1}, h_t = k h~ + (1 - k) h_{t-1}
    % Output: h_t for every t ('CBT'). With the Nov 2025 settings the forward pass,
    % initial weights (same rand calls in the same order) and gradients equal
    % phasedLSTM.m; replication/phased/checkPhased.m tests this and the backward
    % pass of every version against autodiff.
    properties
        NumHiddenUnits
        OutputMode = "sequence"
        Cfg
    end
    properties (Learnable)
        W1    % [4H x C]   gates i; f; g; o
        U     % [4H x H]
        b1    % [4H x 1]
        tau   % [H x 1]    period, in samples
        shi   % [H x 1]    phase shift s
        ron   % [H x 1]    open ratio r_on
        Pi    % [H x 1]    peepholes (zero and frozen unless Cfg.peephole)
        Pf
        Po
    end
    methods
        function layer = phasedLSTMCfg(numHiddenUnits, cfg, args)
            arguments
                numHiddenUnits
                cfg struct
                args.Name = ""
            end
            layer.NumHiddenUnits = numHiddenUnits;
            layer.Cfg = cfg;
            layer.Name = args.Name;
            layer.Description = "Phased-LSTM " + string(cfg.name) + " (" + numHiddenUnits + " units)";
        end

        function layer = initialize(layer, layout)
            H = layer.NumHiddenUnits; c = layer.Cfg;
            inDim = layout.Size(finddim(layout, "C"));
            % same calls, same order as phasedLSTM.m
            if isempty(layer.W1), layer.W1 = initializeGlorot([4*H, inDim]); end
            if isempty(layer.U),  layer.U  = initializeOrthogonal([4*H, H]); end
            if isempty(layer.b1), layer.b1 = initializeUnitForgetGate(4, H); end
            if isempty(layer.tau)
                lo = c.tauLog(1); hi = c.tauLog(2);
                u = lo + (hi - lo) * rand(H, 1, "like", layer.W1);
                layer.tau = exp(u);
            end
            if isempty(layer.shi)
                r = rand(H, 1, "like", layer.W1);
                layer.shi = r .* layer.tau;
            end
            if isempty(layer.ron), layer.ron = c.ron * ones(H, 1, "like", layer.W1); end
            if isempty(layer.Pi),  layer.Pi = zeros(H, 1, "like", layer.W1); end
            if isempty(layer.Pf),  layer.Pf = zeros(H, 1, "like", layer.W1); end
            if isempty(layer.Po),  layer.Po = zeros(H, 1, "like", layer.W1); end
            if canUseGPU
                for p = ["W1","U","b1","tau","shi","ron","Pi","Pf","Po"]
                    layer.(p) = gpuArray(layer.(p));
                end
            end
        end

        function Z = predict(layer, X)
            Z = layer.run(X, layer.Cfg.leakTest, false);
        end

        function [Z, mem] = forward(layer, X)
            [Z, mem] = layer.run(X, layer.Cfg.leakTrain, true);
        end

        function [dX, dW1, dU, db1, dtau, dshi, dron, dPi, dPf, dPo] = backward(layer, X, ~, dZ, mem)
            c = layer.Cfg; H = layer.NumHiddenUnits;
            Xn = numericOf(X); dY = numericOf(dZ);
            [C, B, T] = size(Xn, 1:3);
            W = numericOf(layer.W1); Uw = numericOf(layer.U);
            Pi_ = numericOf(layer.Pi); Pf_ = numericOf(layer.Pf); Po_ = numericOf(layer.Po);
            peep = c.peephole;
            iI = 1:H; iF = H+1:2*H; iG = 2*H+1:3*H; iO = 3*H+1:4*H;
            dGall = zeros(4*H, B, T, 'like', Xn);
            dK = zeros(H, T, 'like', Xn);
            dPi = zeros(H, 1, 'like', Xn); dPf = dPi; dPo = dPi;
            dhc = zeros(H, B, 'like', Xn); dcc = dhc;
            for t = T:-1:1
                k = mem.K(:, t);
                dh = dY(:, :, t) + dhc;
                dc = dcc;
                hp = mem.Hp(:, :, t); cp = mem.Cp(:, :, t);
                ht = mem.Ht(:, :, t); ct = mem.Ct(:, :, t);
                dK(:, t) = sum(dh .* (ht - hp), 2) + sum(dc .* (ct - cp), 2);
                dht = k .* dh; dct = k .* dc;
                dhp = (1 - k) .* dh; dcp = (1 - k) .* dc;
                o = mem.O(:, :, t); tc = mem.Tc(:, :, t);
                dct = dct + dht .* o .* (1 - tc.^2);
                dao = (dht .* tc) .* o .* (1 - o);
                if peep
                    dct = dct + dao .* Po_;
                    dPo = dPo + sum(dao .* ct, 2);
                end
                i = mem.I(:, :, t); f = mem.F(:, :, t); g = mem.G(:, :, t);
                dcp = dcp + dct .* f;
                dai = (dct .* g) .* i .* (1 - i);
                daf = (dct .* cp) .* f .* (1 - f);
                dag = (dct .* i) .* (1 - g.^2);
                if peep
                    dcp = dcp + dai .* Pi_ + daf .* Pf_;
                    dPi = dPi + sum(dai .* cp, 2);
                    dPf = dPf + sum(daf .* cp, 2);
                end
                dGt = zeros(4*H, B, 'like', Xn);
                dGt(iI, :) = dai; dGt(iF, :) = daf; dGt(iG, :) = dag; dGt(iO, :) = dao;
                dGall(:, :, t) = dGt;
                dhc = dhp + Uw' * dGt;
                dcc = dcp;
            end
            G2 = reshape(dGall, 4*H, B*T);
            dW1 = G2 * reshape(Xn, C, B*T)';
            dU  = G2 * reshape(mem.Hp, H, B*T)';
            db1 = sum(G2, 2);
            dXn = reshape(W' * G2, C, B, T);
            % timing parameters
            dtau = zeros(H, 1, 'like', Xn); dshi = dtau; dron = dtau;
            if c.timingGrad
                dphi = dK .* mem.dKdphi;                         % [H x T]
                tauE = mem.tauE;
                dshi = sum(dphi, 2) .* (-1 ./ tauE);
                dtau = sum(dphi .* mem.u, 2) .* (-1 ./ tauE) .* mem.tauMask;
            end
            if c.ronLearn
                dron = sum(dK .* mem.dKdr, 2) .* mem.ronMask;
            end
            if ~peep, dPi(:) = 0; dPf(:) = 0; dPo(:) = 0; end
            if isa(X, 'dlarray') && ~isempty(dims(X)), dX = dlarray(dXn, dims(X)); else, dX = dlarray(dXn); end
            dW1 = dlarray(dW1); dU = dlarray(dU); db1 = dlarray(db1);
            dtau = dlarray(dtau); dshi = dlarray(dshi); dron = dlarray(dron);
            dPi = dlarray(dPi); dPf = dlarray(dPf); dPo = dlarray(dPo);
        end
    end

    methods (Access = private)
        function [Z, mem] = run(layer, X, alpha, keep)
            assert(layer.OutputMode == "sequence", 'phased:outputMode', 'phasedLSTMCfg returns the whole sequence only.');
            c = layer.Cfg; H = layer.NumHiddenUnits;
            Xn = numericOf(X);
            [C, B, T] = size(Xn, 1:3);
            W = numericOf(layer.W1); Uw = numericOf(layer.U); b = numericOf(layer.b1);
            Pi_ = numericOf(layer.Pi); Pf_ = numericOf(layer.Pf); Po_ = numericOf(layer.Po);
            peep = c.peephole;
            % time gate for every unit and step: depends on t only
            [K, dKdphi, dKdr, u, tauE, tauMask, ronMask] = layer.timeGate(T, alpha);
            WX = reshape(W * reshape(Xn, C, B*T), 4*H, B, T) + b;
            iI = 1:H; iF = H+1:2*H; iG = 2*H+1:3*H; iO = 3*H+1:4*H;
            h = zeros(H, B, 'like', Xn); cc = h;
            Y = zeros(H, B, T, 'like', Xn);
            if keep
                mem.Hp = Y; mem.Cp = Y; mem.Ht = Y; mem.Ct = Y; mem.Tc = Y;
                mem.I = Y; mem.F = Y; mem.G = Y; mem.O = Y;
            else
                mem = [];
            end
            for t = 1:T
                Gt = WX(:, :, t) + Uw * h;
                ai = Gt(iI, :); af = Gt(iF, :);
                if peep, ai = ai + Pi_ .* cc; af = af + Pf_ .* cc; end
                i = sigm(ai); f = sigm(af); g = tanh(Gt(iG, :));
                ct = f .* cc + i .* g;
                ao = Gt(iO, :);
                if peep, ao = ao + Po_ .* ct; end
                o = sigm(ao);
                tc = tanh(ct);
                ht = o .* tc;
                k = K(:, t);
                if keep
                    mem.Hp(:, :, t) = h; mem.Cp(:, :, t) = cc; mem.Ht(:, :, t) = ht; mem.Ct(:, :, t) = ct;
                    mem.Tc(:, :, t) = tc; mem.I(:, :, t) = i; mem.F(:, :, t) = f; mem.G(:, :, t) = g; mem.O(:, :, t) = o;
                end
                cc = k .* ct + (1 - k) .* cc;
                h  = k .* ht + (1 - k) .* h;
                Y(:, :, t) = h;
            end
            if keep
                mem.K = K; mem.dKdphi = dKdphi; mem.dKdr = dKdr; mem.u = u;
                mem.tauE = tauE; mem.tauMask = tauMask; mem.ronMask = ronMask;
            end
            Z = dlarray(Y, 'CBT');
        end
    end

    methods
        function [K, dKdphi, dKdr, u, tauE, tauMask, ronMask] = timeGate(layer, T, alpha)
            % K [H x T]: the gate of every unit at steps 1..T, and its derivatives
            c = layer.Cfg;
            tau = numericOf(layer.tau); s = numericOf(layer.shi); r = numericOf(layer.ron);
            tauE = tau; tauMask = ones(size(tau), 'like', tau);
            if c.timingGrad                                 % learned periods stay >= 0.5 samples
                tauMask = cast(tau > 0.5, 'like', tau);
                tauE = max(tau, 0.5);
            end
            ronMask = ones(size(r), 'like', r);
            if c.ronLearn                                   % learned open ratio stays in [1e-3, 1]
                ronMask = cast(r > 1e-3 & r < 1, 'like', r);
                r = min(max(r, 1e-3), 1);
            end
            tt = cast(1:T, 'like', tau);                    % [1 x T]
            u = (tt - s) ./ tauE;                           % [H x T]
            phi = u - floor(u);                             % = mod((t - s)/tau, 1) as in phasedLSTM.m
            switch c.gate
                case 'nov2025', a = 0.1; c1 = 1; c2 = 3; c3 = 1;
                case 'neil',    a = 0.5; c1 = 2; c2 = 2; c3 = 2;
                otherwise, error('phased:gate', 'Unknown gate %s.', c.gate);
            end
            m1 = phi < a * r;
            m2 = ~m1 & (phi < r);
            m3 = ~(m1 | m2);
            K = zeros(size(phi), 'like', phi);
            p1 = phi ./ r;
            K(m1) = c1 * p1(m1);
            K(m2) = c2 - c3 * p1(m2);
            K(m3) = alpha * phi(m3);
            rr = repmat(r, 1, T);
            dKdphi = zeros(size(phi), 'like', phi);
            dKdphi(m1) = c1 ./ rr(m1);
            dKdphi(m2) = -c3 ./ rr(m2);
            dKdphi(m3) = alpha;
            dKdr = zeros(size(phi), 'like', phi);
            dKdr(m1) = -c1 * p1(m1) ./ rr(m1);
            dKdr(m2) =  c3 * p1(m2) ./ rr(m2);
        end
    end
end

function A = numericOf(A)
if isa(A, 'dlarray'), A = extractdata(stripdims(A)); end
end
function y = sigm(x)
y = 1 ./ (1 + exp(-x));
end
% ===== initialisers, identical to phasedLSTM.m =====
function X = initializeGlorot(sz)
fanIn  = sz(2);
fanOut = sz(1);
limit  = sqrt(6/(fanIn+fanOut));
X = (rand(sz,'single')*2 - 1) * limit;
end
function X = initializeOrthogonal(sz)
A = randn(sz,'single');
[U,~,V] = svd(A,'econ');
X = U * V';
X = single(X);
end
function b = initializeUnitForgetGate(numGates, H)
b = zeros(numGates*H,1,'single');
fIx = (H+1):(2*H);
b(fIx) = 1.0;
end
