classdef phaseAwareLSTM_halfAngV2_fast < nnet.layer.Layer & nnet.layer.Formattable
    % phaseAwareLSTM_halfAngV2_fast  FAST port of phase-layer version 'halfAngV2' (variant 'pv_halfAngV2', opt-in)
    %
    % Original : M:\Earth\archive\phaseAwareLSTM_halfAng_v2.m (2025-11-06)
    %            verbatim copy: replication/versions/originals/27_halfAngV2/phaseAwareLSTM_halfAng_v2.m
    % Parent   : halfAng
    % Changes  : tanh(0.6 P) instead of tanh(P/2); new init (glorot-uniform, block-orthogonal, W2 and Q x 0.5, b1 forget +1 on rows 1:H, input -0.5)
    % Note: draft: the original reads its zero states from initialize, so it errors for mini-batches larger than 1; the port runs the intended recurrence. Same file as the M:\Earth toolbox PhaseAwareLSTM_halfAngle (Mar 2026)
    %
    % Same model, learnables and initialize as the original; the recurrence
    % runs in the shared fused engine (replication/versions/fast/common/
    % phaseVersionEngine.m) with the settings in config() below. Checked
    % against the original: replication/versions/checkPhaseVersions.m.

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

    methods (Static)
        function cfg = config()
            cfg = phaseVersionEngine.config('map','halfAngle','k',0.6,'clip',20,'magFloor',1e-12);
        end
    end

    methods
        function layer = phaseAwareLSTM_halfAngV2_fast(numHiddenUnits, args)
            arguments
                numHiddenUnits
                args.Name = "";
                args.OutputMode = "sequence";
            end
            layer.NumHiddenUnits = numHiddenUnits;
            layer.Name           = args.Name;
            layer.OutputMode     = args.OutputMode;
            layer.Description    = "Phase-aware LSTM (halfAngV2, fast) with " + numHiddenUnits + " hidden units";
        end

        function layer = initialize(layer, layout)
            % Learnable initialisation copied from the original, line for line (same RNG draws).
            numChannels = layout.Size(finddim(layout, "C"));
            % ---- Sizes & indices ----
            H = layer.NumHiddenUnits;
            idx_f = 1:H;                          % forget
            idx_i = H+1:2*H;                      % input
            idx_g = 2*H+1:3*H;                    % candidate
            idx_o = 3*H+1:4*H;     

            % after init (or in your init)
            b = zeros(4*H,1,'like',layer.W1);
            b(1:H)           = 1.0;   % forget +1   => σ≈0.73
            b(H+1:2*H)       = -0.5;  % input  -0.5 => σ≈0.38
            b(3*H+1:4*H)     = 0.0;   % output
            layer.b1 = b;
            % output

            % ---- Input-to-hidden weights ----
            % Glorot-uniform for both magnitude and phase branches; phase is quieter.
            if isempty(layer.W1), layer.W1 = glorotUniform([4*H, numChannels]); end
            if isempty(layer.W2), layer.W2 = 0.5 * glorotUniform([4*H, numChannels]); end   % phase branch gain ↓

            % ---- Recurrent weights ----
            % Block-orthogonal (per gate) for stability; phase branch a bit quieter.
            if isempty(layer.U), layer.U = blockOrthogonal(4, H, 1.0); end      % magnitude branch (gain=1)
            if isempty(layer.Q), layer.Q = 0.5 * blockOrthogonal(4, H, 1.0); end % phase branch (gain=0.5)

            % ---- Biases ----
            % Only forget gets +bias. Others start at 0. (Optionally set i to a small negative.)
            if isempty(layer.b1)
                layer.b1 = zeros(4*H,1,'like',layer.W1);
                layer.b1(idx_f) = 1.0;           % unit-forget (σ≈0.73), not 0.99
                % layer.b1(idx_i) = -0.5;        % OPTIONAL: gentle brake on input gate
            end

            % Phase branch biases all zero (don't pre-tilt phase)
            if isempty(layer.b2)
                layer.b2 = zeros(4*H,1,'like',layer.W2);
            end
        end

        function [Y, hiddenState, hiddenPhase, memoryReal, memoryImaginary] = predict(layer, X)
            [Y, hiddenState, hiddenPhase, memoryReal, memoryImaginary] = phaseVersionLayerRun(layer, X, false);
        end

        function [Y, hiddenState, hiddenPhase, memoryReal, memoryImaginary, memory] = forward(layer, X)
            [Y, hiddenState, hiddenPhase, memoryReal, memoryImaginary, memory] = phaseVersionLayerRun(layer, X, true);
        end

        function varargout = backward(layer, X, varargin)
            % Returns dLdX then the learnable gradients in declaration order
            % (W1, W2, U, Q, b1, b2), then zeros for any state-input derivatives.
            g = phaseVersionLayerGrads(layer, X, varargin);
            out = {g.X, g.W1, g.W2, g.U, g.Q, g.b1, g.b2};
            varargout = phaseVersionFinishGrads(out, g, X, nargout);
        end
    end
end
