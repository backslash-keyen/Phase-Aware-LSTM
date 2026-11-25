classdef phaseAwareLSTMBackward < nnet.layer.Layer & nnet.layer.Formattable
    % phaseAwareLSTMBackward  (replication variant 'backwardPhaseLSTM', opt-in)
    %
    % Version 4: the version-3 forward pass plus a hand-written backward
    % pass (backpropagation through time), so MATLAB no longer records and
    % replays ~40 autodiff operations per time step. Copy of
    % utils/modelstuff/layerdefinitions/phaseAwareLSTM.m, which is not changed.
    %
    % Same model and parameters as the original (W1 W2 U Q b1 b2, same
    % sizes, same initialize), same gates, clipping, polar memory and
    % tan-half-angle rotation. The forward maths is that of version 3
    % (phaseAwareLSTMFastest): cos/sin = (1-t^2, 2t)/(1+t^2+eps) without
    % the redundant renormalization, which changes values by ~1e-9.
    %
    % forward() runs on plain numeric arrays (CPU or gpuArray) and keeps
    % what backward() needs: clipped pre-activations and the per-step
    % h, phase, Re/Im memory. backward() recomputes the cheap elementwise
    % values from those and applies the chain rule step by step, then forms
    % all weight gradients with a few large matrix products over B*T.
    % Memory kept per mini-batch: 11*H*B*T values (H 30, B 197, T 500,
    % single: about 130 MB).
    %
    % Gradients match autodiff of the original to floating-point rounding.
    % replication/checkPhaseLSTMVersions.m measures the difference.

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
        function layer = phaseAwareLSTMBackward(numHiddenUnits, args)
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
            if isempty(layer.b2), layer.b2 = initializeUnitForgetGate(4, numHiddenUnits);layer.b2 = layer.b2 * 0;end
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
                    error('phaseAwareLSTMBackward:backwardSyntax', ...
                          'Unexpected backward call with %d arguments.', nargin);
            end

            H  = layer.NumHiddenUnits;
            Xd = raw(stripdims(dlarray(X, 'CBT')));
            W1 = raw(layer.W1); W2 = raw(layer.W2);
            U  = raw(layer.U);  Q3 = raw(layer.Q); Q3 = Q3(1:3*H,:);
            dY = raw(stripdims(dLdY));
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
