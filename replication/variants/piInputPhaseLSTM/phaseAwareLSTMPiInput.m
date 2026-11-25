classdef phaseAwareLSTMPiInput < nnet.layer.Layer & nnet.layer.Formattable
    % phaseAwareLSTMPiInput  (replication variant 'piInputPhaseLSTM', opt-in)
    %
    % The released phaseAwareLSTM (fused version 5, same model and speed) with
    % ONE initialisation line changed: the phase bias b2 is pi on the
    % input-rotation rows H+1:2H (unit-forget pattern x pi) instead of 0.
    % Clip +-20 and b1 exactly as released. Paired with the shipped
    % CWL C2 phase-aware options (get_trainingOptions_CWL_C2_PhaseAware,
    % 20 target epochs, Shuffle never), as in the Nov 2025 runs.
    % Fold-2 screen and 10-fold result: replication/results/phaseaware_2026-11-25.md

    properties
        NumHiddenUnits
        OutputMode
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
        function layer = phaseAwareLSTMPiInput(numHiddenUnits, args)
            arguments
                numHiddenUnits
                args.Name = "";
                args.OutputMode = "sequence";
            end
            layer.NumHiddenUnits = numHiddenUnits;
            layer.Name           = args.Name;
            layer.OutputMode     = args.OutputMode;
            layer.Description    = "Phase-Aware LSTM with " + numHiddenUnits + " hidden units";
        end

        function layer = initialize(layer, layout)
            % Identical to the original, call for call (same RNG draws).
            numHiddenUnits = layer.NumHiddenUnits;
            idx            = finddim(layout, "C");
            numChannels    = layout.Size(idx);
            if isempty(layer.W1), layer.W1 = initializeGlorot([4 * numHiddenUnits, numChannels]);end
            if isempty(layer.W2), layer.W2 = initializeGlorot([4 * numHiddenUnits, numChannels]);end
            if isempty(layer.U), layer.U  = initializeOrthogonal([4 * numHiddenUnits, numHiddenUnits]);end
            if isempty(layer.Q), layer.Q  = initializeOrthogonal([4 * numHiddenUnits, numHiddenUnits]);end
            if isempty(layer.b1), layer.b1 = initializeUnitForgetGate(4, numHiddenUnits);layer.b1(2*numHiddenUnits:end,:) = layer.b1(2*numHiddenUnits:end,:)*0;end
            if isempty(layer.b2), layer.b2 = initializeUnitForgetGate(4, numHiddenUnits);layer.b2 = layer.b2 * pi;end   % pi on the input-rotation rows H+1:2H (released: * 0)
        end

        function [Y, hiddenState,hiddenPhase,memoryReal,memoryImaginary] = predict(layer, X)
            [Y, hiddenState,hiddenPhase,memoryReal,memoryImaginary] = layer.run(X, false);
        end

        function [Y, hiddenState,hiddenPhase,memoryReal,memoryImaginary,memory] = forward(layer, X)
            [Y, hiddenState,hiddenPhase,memoryReal,memoryImaginary,memory] = layer.run(X, true);
        end

        function varargout = backward(layer, X, varargin)
            % Accepts the documented custom-layer backward syntaxes for a
            % layer with one output and four state outputs:
            %   backward(layer, X, Y, dLdY, memory)
            %   backward(layer, X, Y, dLdY, dLdS1..dLdS4, memory)
            %   backward(layer, X, Y, S1..S4, dLdY, dLdS1..dLdS4, memory)
            % Returns dLdX, dLdW1, dLdW2, dLdU, dLdQ, dLdb1, dLdb2, and
            % zeros for any state-input derivatives the caller asks for
            % (the states start from zero at every call).
            memory = varargin{end};
            rest   = varargin(1:end-1);
            dS     = {[], [], [], []};
            switch numel(rest)
                case 2,  dLdY = rest{2};
                case 6,  dLdY = rest{2}; dS = rest(3:6);
                case 10, dLdY = rest{6}; dS = rest(7:10);
                otherwise
                    error('phaseAwareLSTMPiInput:backwardSyntax', ...
                          'Unexpected backward call with %d arguments.', nargin);
            end

            H  = layer.NumHiddenUnits;
            Xd = raw(stripdims(dlarray(X, 'CBT')));
            W1 = raw(layer.W1); W2 = raw(layer.W2);
            U  = raw(layer.U);  Q3 = raw(layer.Q); Q3 = Q3(1:3*H,:);
            dY = raw(stripdims(dLdY));

            if isfield(memory, 'gpu')
                out = gpuBackward(layer, X, Xd, W1, W2, U, Q3, dY, dS, memory);
                for k = numel(out)+1:nargout
                    out{k} = zeros(H, size(Xd, 2), 'like', Xd);
                end
                varargout = wrapGrads(out(1:nargout), X);
                return
            end
            [C, B, T] = size(Xd, 1:3);

            Gs  = memory.G;  Ps  = memory.P;      % clipped pre-activations
            Hs  = memory.h;  Phs = memory.ph;     % [H x B x T+1], slot 1 = t0
            CRs = memory.cR; CIs = memory.cI;
            epsilon = cast(1e-9, 'like', Gs);
            isSequence = layer.OutputMode == "sequence";

            dh  = zerosIfEmpty(dS{1}, H, B, Gs);
            dph = zerosIfEmpty(dS{2}, H, B, Gs);
            dcR = zerosIfEmpty(dS{3}, H, B, Gs);
            dcI = zerosIfEmpty(dS{4}, H, B, Gs);

            dGs = zeros(4*H, B, T, 'like', Gs);
            dPs = zeros(3*H, B, T, 'like', Gs);
            top = 1:H; bot = H+1:2*H;

            for t = T:-1:1
                if isSequence
                    dh = dh + dY(:,:,t);
                elseif t == T
                    dh = dh + dY;
                end

                G = Gs(:,:,t); P = Ps(:,:,t);
                sg = 1 ./ (1 + exp(-G));                 % f i . o (row 3 unused)
                f  = sg(1:H,:); ig = sg(H+1:2*H,:); o = sg(3*H+1:4*H,:);
                gR = tanh(G(2*H+1:3*H,:));
                gI = tanh(P(2*H+1:3*H,:));
                u  = tanh(0.5 * P(1:2*H,:));
                a   = u .* u;
                inv = 1 ./ (a + (1 + epsilon));
                c   = (1 - a) .* inv;
                s   = (2 * u) .* inv;

                cRp = CRs(:,:,t);  cIp = CIs(:,:,t);     % memory before step t
                cR  = CRs(:,:,t+1); cI = CIs(:,:,t+1);   % memory after step t
                rho2 = cR.^2 + cI.^2;
                rho  = sqrt(rho2);
                tm   = tanh(max(rho, epsilon));

                % h = o .* tanh(max(|c|, eps)),  phase = atan2(cI, cR)
                do   = dh .* tm;
                dmag = dh .* o .* (1 - tm.^2) .* (rho > epsilon);
                rinv = 1 ./ max(rho, realmin('like', rho));
                r2inv = rinv.^2;
                dcR = dcR + dmag .* cR .* rinv - dph .* cI .* r2inv;
                dcI = dcI + dmag .* cI .* rinv + dph .* cR .* r2inv;

                % cR = xN(top)+xN(bot), xN = A.*c - B.*s, yN = A.*s + B.*c
                m  = [f; ig];
                x  = [cRp; gR];
                y  = [cIp; gI];
                A  = m .* x;
                Bv = m .* y;
                dxN = [dcR; dcR];
                dyN = [dcI; dcI];
                dA = dxN .* c + dyN .* s;
                dB = dyN .* c - dxN .* s;
                dc = dxN .* A + dyN .* Bv;
                ds = dyN .* A - dxN .* Bv;
                dm = dA .* x + dB .* y;
                dx = dA .* m;
                dy = dB .* m;

                % c = (1-t^2)/(1+t^2+eps), s = 2t/(1+t^2+eps), t = tanh(0.5*P)
                du  = dc .* (-2 * u .* inv .* (1 + c)) + ds .* (2 * inv .* (1 - u .* s));
                dPa = du .* (1 - a) * 0.5;

                dG = [dm(top,:) .* f .* (1 - f); ...
                      dm(bot,:) .* ig .* (1 - ig); ...
                      dx(bot,:) .* (1 - gR.^2); ...
                      do .* o .* (1 - o)];
                dP = [dPa; dy(bot,:) .* (1 - gI.^2)];
                dG = dG .* (abs(G) < 20);               % clip at +-20
                dP = dP .* (abs(P) < 20);

                dGs(:,:,t) = dG;
                dPs(:,:,t) = dP;

                dh  = U.'  * dG;
                dph = Q3.' * dP;
                dcR = dx(top,:);
                dcI = dy(top,:);
            end

            dG2 = reshape(dGs, 4*H, B*T);
            dP2 = reshape(dPs, 3*H, B*T);
            X2  = reshape(Xd, C, B*T);
            dW1 = dG2 * X2.';
            dW2 = [dP2 * X2.'; zeros(H, C, 'like', dP2)];
            dU  = dG2 * reshape(Hs(:,:,1:T),  H, B*T).';
            dQ  = [dP2 * reshape(Phs(:,:,1:T), H, B*T).'; zeros(H, H, 'like', dP2)];
            db1 = sum(dG2, 2);
            db2 = [sum(dP2, 2); zeros(H, 1, 'like', dP2)];
            dX  = pagemtimes(W1.', dGs) + pagemtimes(W2(1:3*H,:).', dPs);

            if isa(X, 'dlarray') && ~isempty(dims(X))
                dX = dlarray(dX, dims(X));
            end
            out = {dX, dW1, dW2, dU, dQ, db1, db2};
            for k = numel(out)+1:nargout
                out{k} = zeros(H, B, 'like', Gs);
            end
            varargout = wrapGrads(out(1:nargout), X);
        end
    end

    methods (Access = private)
        function [Y, hiddenState,hiddenPhase,memoryReal,memoryImaginary,memory] = run(layer, X, keep)
            epsilon = 1e-9;
            H  = layer.NumHiddenUnits;
            Xd = raw(stripdims(dlarray(X, 'CBT')));
            [C, B, T] = size(Xd, 1:3); %#ok<ASGLU>
            W1 = raw(layer.W1); W2 = raw(layer.W2);
            U  = raw(layer.U);  Q  = raw(layer.Q);
            b1 = raw(layer.b1); b2 = raw(layer.b2);
            W2 = W2(1:3*H,:); Q3 = Q(1:3*H,:); b2 = b2(1:3*H);   % phase output-gate rows are unused

            if isa(Xd, 'gpuArray')
                [Y, hiddenState,hiddenPhase,memoryReal,memoryImaginary,memory] = ...
                    gpuForward(layer, Xd, W1, W2, b1, b2, U, Q3, keep);
                return
            end

            isSequence = layer.OutputMode == "sequence";
            if ~isSequence && layer.OutputMode ~= "last"
                error("OutputMode must be 'sequence' or 'last'.");
            end

            WX1 = pagemtimes(W1, Xd) + b1;      % [4H x B x T]
            WX2 = pagemtimes(W2, Xd) + b2;      % [3H x B x T]

            h  = zeros(H, B, 'like', WX1); ph = h; cR = h; cI = h;
            if keep
                Gs  = zeros(4*H, B, T, 'like', WX1);
                Ps  = zeros(3*H, B, T, 'like', WX1);
                Hs  = zeros(H, B, T+1, 'like', WX1);
                Phs = Hs; CRs = Hs; CIs = Hs;
            end
            if isSequence, Yd = zeros(H, B, T, 'like', WX1); end
            top = 1:H; bot = H+1:2*H;

            for t = 1:T
                G = max(min(WX1(:,:,t) + U  * h,  20), -20);
                P = max(min(WX2(:,:,t) + Q3 * ph, 20), -20);

                sg = 1 ./ (1 + exp(-G));
                gR = tanh(G(2*H+1:3*H,:));
                gI = tanh(P(2*H+1:3*H,:));
                u  = tanh(0.5 * P(1:2*H,:));
                a   = u .* u;
                inv = 1 ./ (a + (1 + epsilon));
                c   = (1 - a) .* inv;
                s   = (2 * u) .* inv;

                m  = sg(1:2*H,:);
                x  = [cR; gR];
                y  = [cI; gI];
                A  = m .* x;
                Bv = m .* y;
                xN = A .* c - Bv .* s;
                yN = A .* s + Bv .* c;
                cR = xN(top,:) + xN(bot,:);
                cI = yN(top,:) + yN(bot,:);

                mag = max(sqrt(cR.^2 + cI.^2), epsilon);
                h   = sg(3*H+1:4*H,:) .* tanh(mag);
                ph  = atan2(cI, cR);

                if isSequence, Yd(:,:,t) = h; end
                if keep
                    Gs(:,:,t) = G;  Ps(:,:,t) = P;
                    Hs(:,:,t+1) = h; Phs(:,:,t+1) = ph;
                    CRs(:,:,t+1) = cR; CIs(:,:,t+1) = cI;
                end
            end

            hiddenState = dlarray(h);  hiddenPhase = dlarray(ph);
            memoryReal  = dlarray(cR); memoryImaginary = dlarray(cI);
            if isSequence
                Y = dlarray(Yd, 'CBT');
            else
                Y = dlarray(h, 'CB');
            end
            if keep
                memory = struct('G', Gs, 'P', Ps, 'h', Hs, 'ph', Phs, 'cR', CRs, 'cI', CIs);
            else
                memory = [];
            end
        end
    end
end

function v = raw(v)
if isa(v, 'dlarray'), v = extractdata(v); end
end

function out = wrapGrads(out, X)
% trainNetwork's legacy custom-layer path (CustomLayerLegacyStrategy) calls
% extractdata on every output of backward(), so return dlarrays; dlnetwork
% accepts them too. dLdX keeps the format of X.
for k = 1:numel(out)
    if isa(out{k}, 'dlarray'), continue; end
    if k == 1 && isa(X, 'dlarray') && ~isempty(dims(X))
        out{k} = dlarray(out{k}, dims(X));
    else
        out{k} = dlarray(out{k});
    end
end
end

function z = zerosIfEmpty(v, H, B, likeThis)
if isempty(v)
    z = zeros(H, B, 'like', likeThis);
else
    z = raw(stripdims(dlarray(v)));
end
end

% =====================================================================
% GPU path: one fused kernel per time step
% =====================================================================
function [Y, hS, phS, cRS, cIS, memory] = gpuForward(layer, Xd, W1, W2, b1, b2, U, Q3, keep)
H = layer.NumHiddenUnits;
[C, B, T] = size(Xd, 1:3); %#ok<ASGLU>
Wst = [W1; W2];                                   % [7H x C]: f i g o | theta_f theta_i g_im
bst = [b1; b2];
R   = [U, zeros(4*H, H, 'like', U); zeros(3*H, H, 'like', U), Q3];   % [7H x 2H] on [h; phase]
WX  = pagemtimes(Wst, Xd) + bst;                  % [7H x B x T]

hp = zeros(2*H, B, 'like', WX);                   % [h; phase]
cc = zeros(2*H, B, 'like', WX);                   % [Re; Im] memory
HPs = cell(1, T+1); CCs = cell(1, T+1); Ms = cell(1, T);
HPs{1} = hp; CCs{1} = cc;
I2 = reshape(gpuArray(1:2*H*B), 2*H, B);
for t = 1:T
    M = R * hp;
    [hp, cc] = fusedForwardStep(I2, M, WX, cc, t, H, B);
    HPs{t+1} = hp; CCs{t+1} = cc;
    if keep, Ms{t} = M; end
end
HP = cat(3, HPs{:});                              % [2H x B x T+1]
if layer.OutputMode == "sequence"
    Y = dlarray(HP(1:H,:,2:end), 'CBT');
else
    Y = dlarray(hp(1:H,:), 'CB');
end
hS  = dlarray(hp(1:H,:));    phS = dlarray(hp(H+1:end,:));
cRS = dlarray(cc(1:H,:));    cIS = dlarray(cc(H+1:end,:));
if keep
    memory = struct('gpu', true, 'WX', WX, 'M', {Ms}, 'HP', HP, 'CC', cat(3, CCs{:}), 'R', R, 'Wst', Wst);
else
    memory = [];
end
end

function [hp, cc] = fusedForwardStep(I2, M, WX, ccPrev, t, H, B)
% One time step for every unit and batch element. Element r of the
% [2H x B] output is row j, column b; rows 1..H give h and Re(c),
% rows H+1..2H give phase and Im(c) (each thread computes its unit fully).
epsilon = 1e-9;
H7 = 7*H; H2 = 2*H;
wOff = (t-1)*H7*B;
[hp, cc] = arrayfun(@kernel, I2);
    function [o1, o2] = kernel(r)
        rr = r - 1;
        j  = mod(rr, H2) + 1;
        b  = floor(rr / H2);
        u  = j;
        if j > H, u = j - H; end
        base = b*H7 + u;
        wb   = wOff + base;
        zf = min(max(WX(wb)       + M(base),       -20), 20);
        zi = min(max(WX(wb + H)   + M(base + H),   -20), 20);
        zg = min(max(WX(wb + 2*H) + M(base + 2*H), -20), 20);
        zo = min(max(WX(wb + 3*H) + M(base + 3*H), -20), 20);
        pf = min(max(WX(wb + 4*H) + M(base + 4*H), -20), 20);
        pn = min(max(WX(wb + 5*H) + M(base + 5*H), -20), 20);
        pg = min(max(WX(wb + 6*H) + M(base + 6*H), -20), 20);
        f  = 1 / (1 + exp(-zf));
        ig = 1 / (1 + exp(-zi));
        o  = 1 / (1 + exp(-zo));
        gR = tanh(zg);
        gI = tanh(pg);
        uf = tanh(0.5 * pf); af = uf * uf; vf = 1 / (af + (1 + epsilon));
        cf = (1 - af) * vf;  sf = (2 * uf) * vf;
        un = tanh(0.5 * pn); an = un * un; vn = 1 / (an + (1 + epsilon));
        cn = (1 - an) * vn;  sn = (2 * un) * vn;
        cb  = b*H2 + u;
        cRp = ccPrev(cb);
        cIp = ccPrev(cb + H);
        A1 = f * cRp; B1 = f * cIp; A2 = ig * gR; B2 = ig * gI;
        cR = (A1 * cf - B1 * sf) + (A2 * cn - B2 * sn);
        cI = (A1 * sf + B1 * cf) + (A2 * sn + B2 * cn);
        mag = max(sqrt(cR * cR + cI * cI), epsilon);
        if j <= H
            o1 = o * tanh(mag);
            o2 = cR;
        else
            o1 = atan2(cI, cR);
            o2 = cI;
        end
    end
end

function out = gpuBackward(layer, X, Xd, W1, W2, U, Q3, dY, dS, memory) %#ok<INUSL>
H = layer.NumHiddenUnits;
[C, B, T] = size(Xd, 1:3);
WX = memory.WX; Ms = memory.M; HP = memory.HP; CC = memory.CC; R = memory.R; Wst = memory.Wst;
isSequence = layer.OutputMode == "sequence";

z = zeros(H, B, 'like', WX);
s = cell(1, 4);
for k = 1:4
    if isempty(dS{k}), s{k} = z; else, s{k} = raw(stripdims(dlarray(dS{k}))); end
end
dhp = [s{1}; s{2}];                               % d[h; phase] flowing into step T
dcp = zeros(7*H, B, 'like', WX);                  % rows 1..H dRe(c), H+1..2H dIm(c)
dcp(1:H,:) = s{3}; dcp(H+1:2*H,:) = s{4};

I7 = reshape(gpuArray(1:7*H*B), 7*H, B);
dSs = cell(1, T);
Rt  = R.';
for t = T:-1:1
    if isSequence
        yOff = (t-1)*H*B; yOn = 1;
    else
        yOff = 0; yOn = double(t == T);
    end
    [dSt, dcp] = fusedBackwardStep(I7, Ms{t}, WX, CC, dY, dhp, dcp, t, H, B, yOff, yOn);
    dSs{t} = dSt;
    dhp = Rt * dSt;
end
dSall = cat(3, dSs{:});                           % [7H x B x T]
dS2   = reshape(dSall, 7*H, B*T);
X2    = reshape(Xd, C, B*T);
dWst  = dS2 * X2.';
dbst  = sum(dS2, 2);
hPrev = reshape(HP(1:H,:,1:T),     H, B*T);
pPrev = reshape(HP(H+1:2*H,:,1:T), H, B*T);
dW1 = dWst(1:4*H,:);
dW2 = [dWst(4*H+1:end,:); zeros(H, C, 'like', dWst)];
dU  = dS2(1:4*H,:) * hPrev.';
dQ  = [dS2(4*H+1:end,:) * pPrev.'; zeros(H, H, 'like', dWst)];
db1 = dbst(1:4*H);
db2 = [dbst(4*H+1:end); zeros(H, 1, 'like', dWst)];
dX  = pagemtimes(Wst.', dSall);
if isa(X, 'dlarray') && ~isempty(dims(X))
    dX = dlarray(dX, dims(X));
end
out = {dX, dW1, dW2, dU, dQ, db1, db2};
end

function [dSt, dcpOut] = fusedBackwardStep(I7, M, WX, CC, dY, dhp, dcp, t, H, B, yOff, yOn)
% Chain rule of one time step. Element r of the [7H x B] output is the
% gradient of pre-activation row k (f i g o theta_f theta_i g_im) of unit u;
% the second output carries dRe/dIm of the previous memory in rows 1..2H.
epsilon = 1e-9;
H7 = 7*H; H2 = 2*H;
wOff  = (t-1)*H7*B;
cPrev = (t-1)*H2*B;
cNow  = t*H2*B;
[dSt, dcpOut] = arrayfun(@kernel, I7);
    function [v, v2] = kernel(r)
        rr  = r - 1;
        row = mod(rr, H7) + 1;
        b   = floor(rr / H7);
        k   = floor((row - 1) / H) + 1;
        u   = row - (k - 1)*H;
        base = b*H7 + u;
        wb   = wOff + base;
        qf = WX(wb)       + M(base);
        qi = WX(wb + H)   + M(base + H);
        qg = WX(wb + 2*H) + M(base + 2*H);
        qo = WX(wb + 3*H) + M(base + 3*H);
        qpf = WX(wb + 4*H) + M(base + 4*H);
        qpn = WX(wb + 5*H) + M(base + 5*H);
        qpg = WX(wb + 6*H) + M(base + 6*H);
        f  = 1 / (1 + exp(-min(max(qf, -20), 20)));
        ig = 1 / (1 + exp(-min(max(qi, -20), 20)));
        o  = 1 / (1 + exp(-min(max(qo, -20), 20)));
        gR = tanh(min(max(qg, -20), 20));
        gI = tanh(min(max(qpg, -20), 20));
        uf = tanh(0.5 * min(max(qpf, -20), 20)); af = uf * uf; vf = 1 / (af + (1 + epsilon));
        cf = (1 - af) * vf;  sf = (2 * uf) * vf;
        un = tanh(0.5 * min(max(qpn, -20), 20)); an = un * un; vn = 1 / (an + (1 + epsilon));
        cn = (1 - an) * vn;  sn = (2 * un) * vn;
        cRp = CC(cPrev + b*H2 + u);
        cIp = CC(cPrev + b*H2 + H + u);
        cR  = CC(cNow  + b*H2 + u);
        cI  = CC(cNow  + b*H2 + H + u);
        A1 = f * cRp; B1 = f * cIp; A2 = ig * gR; B2 = ig * gI;

        dh  = dhp(b*H2 + u) + yOn * dY(yOff + b*H + u);
        dph = dhp(b*H2 + H + u);
        dcR = dcp(b*H7 + u);
        dcI = dcp(b*H7 + H + u);

        rho  = sqrt(cR * cR + cI * cI);
        tm   = tanh(max(rho, epsilon));
        dout = dh * tm;
        dmag = dh * o * (1 - tm * tm);
        if rho <= epsilon, dmag = 0 * dmag; end
        rinv = 1 / max(rho, 1e-30);
        r2   = rinv * rinv;
        dcR = dcR + dmag * cR * rinv - dph * cI * r2;
        dcI = dcI + dmag * cI * rinv + dph * cR * r2;

        dA1 = dcR * cf + dcI * sf;  dB1 = dcI * cf - dcR * sf;
        dA2 = dcR * cn + dcI * sn;  dB2 = dcI * cn - dcR * sn;

        v2 = 0 * dcR;
        if k == 1
            v  = (dA1 * cRp + dB1 * cIp) * f * (1 - f);  q = qf;
            v2 = dA1 * f;                                 % dRe(c_prev)
        elseif k == 2
            v  = (dA2 * gR + dB2 * gI) * ig * (1 - ig);  q = qi;
            v2 = dB1 * f;                                 % dIm(c_prev)
        elseif k == 3
            v  = dA2 * ig * (1 - gR * gR);  q = qg;
        elseif k == 4
            v  = dout * o * (1 - o);        q = qo;
        elseif k == 5
            dc1 = dcR * A1 + dcI * B1;  ds1 = dcI * A1 - dcR * B1;
            du  = dc1 * (-2 * uf * vf * (1 + cf)) + ds1 * (2 * vf * (1 - uf * sf));
            v   = du * (1 - af) * 0.5;      q = qpf;
        elseif k == 6
            dc2 = dcR * A2 + dcI * B2;  ds2 = dcI * A2 - dcR * B2;
            du  = dc2 * (-2 * un * vn * (1 + cn)) + ds2 * (2 * vn * (1 - un * sn));
            v   = du * (1 - an) * 0.5;      q = qpn;
        else
            v  = dB2 * ig * (1 - gI * gI);  q = qpg;
        end
        if abs(q) >= 20, v = 0 * v; end                   % clip at +-20
    end
end
