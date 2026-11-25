classdef phaseAwareLSTM_atan2v4_fast < nnet.layer.Layer & nnet.layer.Formattable
    % phaseAwareLSTM_atan2v4_fast  FAST port of phase-layer version 'atan2v4' (variant 'pv_atan2v4', opt-in)
    %
    % Original : L:\plstm\helperLibrary\modelframe\architectures\layerfunctions\phaseAwareLSTMs_versions\phaseAwareLSTM_atan2_ver4.m (2025-10-20)
    %            verbatim copy: replication/versions/originals/17_atan2v4/phaseAwareLSTM_atan2_ver4.m
    % Parent   : atan2v1
    % Changes  : eps floors removed; fed-back phase = phase
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
        W3
        U
        Q
        b1
        b2
        b3
    end

    properties (State)
        hiddenMagnitude
        hiddenPhase
        memoryReal
        memoryImaginary
    end

    methods (Static)
        function cfg = config()
            cfg = phaseVersionEngine.config('map','atan2QW','imag','angle','carry',false,'sign','sum');
        end
    end

    methods
        function layer = phaseAwareLSTM_atan2v4_fast(numHiddenUnits, args)
            arguments
                numHiddenUnits
                args.Name = "";
                args.OutputMode = "sequence";
            end
            layer.NumHiddenUnits = numHiddenUnits;
            layer.Name           = args.Name;
            layer.OutputMode     = args.OutputMode;
            layer.Description    = "Phase-aware LSTM (atan2v4, fast) with " + numHiddenUnits + " hidden units";
        end

        function layer = initialize(layer, layout)
            % Learnable initialisation copied from the original, line for line (same RNG draws).
            numHiddenUnits = layer.NumHiddenUnits;
            numChannels = layout.Size(finddim(layout, "C"));
            if isempty(layer.W1),layer.W1 = initializeGlorot([4 * numHiddenUnits, numChannels]);end
            if isempty(layer.W2),layer.W2 = initializeGlorot([4 * numHiddenUnits, numChannels]);end
            if isempty(layer.W3),layer.W3 = initializeGlorot([4 * numHiddenUnits, numChannels]);end
            if isempty(layer.U),layer.U = initializeOrthogonal([4 * numHiddenUnits, numHiddenUnits]);end
            if isempty(layer.Q),layer.Q = initializeOrthogonal([4 *numHiddenUnits, numHiddenUnits]);end
            if isempty(layer.b1), layer.b1 = initializeUnitForgetGate(4,numHiddenUnits);end
            if isempty(layer.b2), layer.b2 = initializeUnitForgetGate(4,numHiddenUnits);layer.b2=layer.b2*pi;end
            if isempty(layer.b3), layer.b3 = layer.b2; end   % original: b3 = gpuArray(b2) in initializeGPUlayer (unused)
        end

        function [Y, hiddenState, hiddenPhase, memoryReal, memoryImaginary] = predict(layer, X)
            [Y, hiddenState, hiddenPhase, memoryReal, memoryImaginary] = phaseVersionLayerRun(layer, X, false);
        end

        function [Y, hiddenState, hiddenPhase, memoryReal, memoryImaginary, memory] = forward(layer, X)
            [Y, hiddenState, hiddenPhase, memoryReal, memoryImaginary, memory] = phaseVersionLayerRun(layer, X, true);
        end

        function varargout = backward(layer, X, varargin)
            % Returns dLdX then the learnable gradients in declaration order
            % (W1, W2, W3, U, Q, b1, b2, b3), then zeros for any state-input derivatives.
            g = phaseVersionLayerGrads(layer, X, varargin);
            out = {g.X, g.W1, g.W2, g.W3, g.U, g.Q, g.b1, g.b2, g.b3};
            varargout = phaseVersionFinishGrads(out, g, X, nargout);
        end
    end
end
