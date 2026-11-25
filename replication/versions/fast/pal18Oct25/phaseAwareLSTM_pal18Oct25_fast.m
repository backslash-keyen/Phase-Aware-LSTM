classdef phaseAwareLSTM_pal18Oct25_fast < nnet.layer.Layer & nnet.layer.Formattable
    % phaseAwareLSTM_pal18Oct25_fast  FAST port of phase-layer version 'pal18Oct25' (variant 'pv_pal18Oct25', opt-in)
    %
    % Original : M:\Earth\helperLibrary\modelframe\architectures\layerfunctions\phaseAwareLSTM.m (2025-10-18)
    %            verbatim copy: replication/versions/originals/19_pal18Oct25/phaseAwareLSTM.m
    % Parent   : pal12Oct25
    % Changes  : memory carried between steps (the no-carry bug fixed)
    % Note: M: copy only
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
            cfg = phaseVersionEngine.config('map','atanEps','gateEps',true,'imag','angle','carry',true,'sign','sum','hEps',true,'phMemEps',true,'phase','OpTimesMem','phOutEps',true);
        end
    end

    methods
        function layer = phaseAwareLSTM_pal18Oct25_fast(numHiddenUnits, args)
            arguments
                numHiddenUnits
                args.Name = "";
                args.OutputMode = "sequence";
            end
            layer.NumHiddenUnits = numHiddenUnits;
            layer.Name           = args.Name;
            layer.OutputMode     = args.OutputMode;
            layer.Description    = "Phase-aware LSTM (pal18Oct25, fast) with " + numHiddenUnits + " hidden units";
        end

        function layer = initialize(layer, layout)
            % Learnable initialisation copied from the original, line for line (same RNG draws).
            numHiddenUnits = layer.NumHiddenUnits;
            numChannels = layout.Size(finddim(layout, "C"));
            if isempty(layer.W1),layer.W1 = initializeGlorot([4 * numHiddenUnits, numChannels]);end
            if isempty(layer.W2),layer.W2 = initializeGlorot([4 * numHiddenUnits, numChannels]);end
            if isempty(layer.U),layer.U = initializeOrthogonal([4 * numHiddenUnits, numHiddenUnits]);end
            if isempty(layer.Q),layer.Q = initializeOrthogonal([4 *numHiddenUnits, numHiddenUnits]);end
            if isempty(layer.b1), layer.b1 = initializeUnitForgetGate(4,numHiddenUnits);end
            if isempty(layer.b2), layer.b2 = initializeUnitForgetGate(4,numHiddenUnits);layer.b2=layer.b2*pi;end
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
