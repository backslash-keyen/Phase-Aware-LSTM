classdef phaseAwareLSTM_halfAng_fast < nnet.layer.Layer & nnet.layer.Formattable
    % phaseAwareLSTM_halfAng_fast  FAST port of phase-layer version 'halfAng' (variant 'pv_halfAng', opt-in)
    %
    % Original : L:\plstm\helperLibrary\modelframe\phaseAwareLSTM_halfAng.m (2025-11-06)
    %            verbatim copy: replication/versions/originals/26_halfAng/phaseAwareLSTM_halfAng.m
    % Parent   : notrig
    % Changes  : theta via tan-half-angle t = tanh(P/2); clip +-20; b1 zeroed from row 2H; b2 = 0
    % Note: = the released layer (utils/.../phaseAwareLSTM.m). Run as 'phase-lstm-halfang'. Fast version already exists: variant fusedPhaseLSTM
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
            cfg = phaseVersionEngine.config('map','halfAngle','k',0.5,'clip',20,'magFloor',1e-12);
        end
    end

    methods
        function layer = phaseAwareLSTM_halfAng_fast(numHiddenUnits, args)
            arguments
                numHiddenUnits
                args.Name = "";
                args.OutputMode = "sequence";
            end
            layer.NumHiddenUnits = numHiddenUnits;
            layer.Name           = args.Name;
            layer.OutputMode     = args.OutputMode;
            layer.Description    = "Phase-aware LSTM (halfAng, fast) with " + numHiddenUnits + " hidden units";
        end

        function layer = initialize(layer, layout)
            % Learnable initialisation copied from the original, line for line (same RNG draws).
            numHiddenUnits = layer.NumHiddenUnits;
            numChannels = layout.Size(finddim(layout, "C"));
            if isempty(layer.W1),layer.W1 = initializeGlorot([4 * numHiddenUnits, numChannels]);end
            if isempty(layer.W2),layer.W2 = initializeGlorot([4 * numHiddenUnits, numChannels]);end
            if isempty(layer.U),layer.U = initializeOrthogonal([4 * numHiddenUnits, numHiddenUnits]);end
            if isempty(layer.Q),layer.Q = initializeOrthogonal([4 *numHiddenUnits, numHiddenUnits]);end
            if isempty(layer.b1), layer.b1 = initializeUnitForgetGate(4,numHiddenUnits);layer.b1(2*numHiddenUnits:end,:) = layer.b1(2*numHiddenUnits:end,:)*0;end
            if isempty(layer.b2), layer.b2 = initializeUnitForgetGate(4,numHiddenUnits);layer.b2=layer.b2*0;end
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
