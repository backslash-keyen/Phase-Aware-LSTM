classdef phaseAwareLSTMFastest < nnet.layer.Layer & nnet.layer.Formattable & nnet.layer.Acceleratable
    % phaseAwareLSTMFastest  (replication variant 'fastestPhaseLSTM', opt-in)
    %
    % Version 3: the fastest forward pass of the same model, gradients still
    % by MATLAB autodiff. Copy of utils/modelstuff/layerdefinitions/
    % phaseAwareLSTM.m, which is not changed.
    %
    % Same model, same parameters (W1 W2 U Q b1 b2, same sizes, same
    % initialize), same gates, same clipping, same polar memory, same
    % tan-half-angle (Weierstrass) rotation, same magnitude/phase outputs.
    % What changes is only how the arithmetic is laid out per time step:
    %
    %   1. One matmul per step instead of two: the magnitude and phase
    %      branches are stacked into one [7H x 2H] block matrix acting on
    %      [h; phase]. Rows are ordered so each nonlinearity reads one
    %      contiguous block: [f i o | g_re g_im | theta_f theta_i].
    %   2. The phase branch's output-gate rows (H(idx4,:) in the original)
    %      are never used by the original, so they are not computed. Their
    %      gradient is zero in both versions.
    %   3. tanh(0.5*clip(z,20)) = tanh(clip(0.5*z,10)): the 0.5 is folded
    %      into the stacked weights once per call (exact in floating point),
    %      and one clip with a per-row bound handles both branches.
    %   4. cos/sin from t = tan(theta/2) as (1-t^2)/(1+t^2+eps) and
    %      2t/(1+t^2+eps) without the follow-up division by
    %      r = sqrt(cos^2+sin^2+eps). For these formulas
    %      cos^2+sin^2 = ((1+t^2)/(1+t^2+eps))^2, so r differs from 1 by
    %      about eps = 1e-9, far below single-precision resolution (1.2e-7).
    %      The rotation stays unit-norm to that accuracy.
    %   5. Outputs are collected in a cell and joined once (as in version 2).
    %
    % Results match the original to floating-point rounding, not bit for
    % bit. replication/checkPhaseLSTMVersions.m measures the difference.

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
        function layer = phaseAwareLSTMFastest(numHiddenUnits, args)
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
            epsilon = 1e-9;
            H  = layer.NumHiddenUnits;
            B  = size(X, finddim(X, "B"));
            T  = size(X, finddim(X, "T"));
            X  = stripdims(dlarray(X, 'CBT'));

            isSequence = layer.OutputMode == "sequence";
            if ~isSequence && layer.OutputMode ~= "last"
                error("OutputMode must be 'sequence' or 'last'.");
            end

            % ---- stacked, reordered parameters (once per call) -----------
            g  = @(k) (k-1)*H + (1:H);            % rows of gate k in a 4H block
            rg = [g(1) g(2) g(4) g(3)];           % magnitude rows: f i o g_re
            rp = [g(3) g(1) g(2)];                % phase rows:     g_im theta_f theta_i
            half = [ones(5*H,1,'like',X); 0.5*ones(2*H,1,'like',X)];   % fold the 0.5 of tanh(0.5*.)
            bnd  = [20*ones(5*H,1,'like',X); 10*ones(2*H,1,'like',X)];

            Win = [layer.W1(rg,:); layer.W2(rp,:)] .* half;           % [7H x C]
            bin = [layer.b1(rg);   layer.b2(rp)]   .* half;           % [7H x 1]
            Z   = zeros(4*H, H, 'like', X); Z3 = zeros(3*H, H, 'like', X);
            R   = [layer.U(rg,:), Z; Z3, layer.Q(rp,:)] .* half;      % [7H x 2H]

            WX = pagemtimes(Win, X) + bin;                            % [7H x B x T]

            hp = zeros(2*H, B, 'like', X);        % [h; phase]
            cR = zeros(H, B, 'like', X);
            cI = zeros(H, B, 'like', X);
            if isSequence, Ys = cell(1, T); end

            iS  = 1:3*H;          % sigmoid block: f i o
            iT  = 3*H+1:5*H;      % tanh block:    g_re g_im
            iA  = 5*H+1:7*H;      % half-angle:    theta_f theta_i
            top = 1:H; bot = H+1:2*H;

            for t = 1:T
                S  = WX(:,:,t) + R * hp;
                S  = max(min(S, bnd), -bnd);

                sg = sigmoid(S(iS,:));            % [f; i; o]
                gc = tanh(S(iT,:));               % [g_re; g_im]
                u  = tanh(S(iA,:));               % tan(theta/2), [2H x B]

                a   = u .* u;
                inv = 1 ./ (a + (1 + epsilon));
                c   = (1 - a) .* inv;             % cos(theta)
                s   = (2 * u) .* inv;             % sin(theta)

                m  = sg(1:2*H,:);                 % [f; i]
                x  = cat(1, cR, gc(top,:));
                y  = cat(1, cI, gc(bot,:));
                A  = m .* x;
                Bv = m .* y;
                xN = A .* c - Bv .* s;
                yN = A .* s + Bv .* c;

                cR = xN(top,:) + xN(bot,:);
                cI = yN(top,:) + yN(bot,:);

                mag = max(sqrt(cR.^2 + cI.^2), epsilon);
                h   = sg(2*H+1:3*H,:) .* tanh(mag);
                ph  = atan2(cI, cR);
                hp  = cat(1, h, ph);

                if isSequence, Ys{t} = h; end
            end

            hiddenState     = h;
            hiddenPhase     = ph;
            memoryReal      = cR;
            memoryImaginary = cI;
            if isSequence
                Y = dlarray(cat(3, Ys{:}), 'CBT');
            else
                Y = dlarray(h, 'CB');
            end
        end

        function [Y, hiddenState,hiddenPhase,memoryReal,memoryImaginary] = forward(layer,X)
            [Y, hiddenState,hiddenPhase,memoryReal,memoryImaginary] = predict(layer, X);
        end
    end
end
