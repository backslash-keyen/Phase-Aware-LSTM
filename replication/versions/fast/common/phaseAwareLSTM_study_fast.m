classdef phaseAwareLSTM_study_fast < nnet.layer.Layer & nnet.layer.Formattable
    % phaseAwareLSTM_study_fast  One layer for every cell of the pipeline study (variants 'pv_ps_<cell>', opt-in)
    %
    % Same learnables, states and engine as the other fast ports
    % (replication/versions/fast/common/phaseVersionEngine.m). The cell
    % (replication/study/pipelineStudyCells.m) sets the engine options and
    % the b1 / b2 / Q initialisation; with cell 'base' the layer equals
    % pv_piSign_none (released half-angle + pi on the input rotation).

    properties
        NumHiddenUnits
        OutputMode
        StudyCell
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
        function cfg = config(layer)
            c = pipelineStudyCells(layer.StudyCell);
            cfg = phaseVersionEngine.config(c.eng{:});
        end

        function layer = phaseAwareLSTM_study_fast(numHiddenUnits, cellId, args)
            arguments
                numHiddenUnits
                cellId
                args.Name = "";
                args.OutputMode = "sequence";
            end
            layer.NumHiddenUnits = numHiddenUnits;
            layer.StudyCell      = char(cellId);
            layer.Name           = args.Name;
            layer.OutputMode     = args.OutputMode;
            layer.Description    = "Phase-aware LSTM (study cell " + cellId + ", fast) with " + numHiddenUnits + " hidden units";
        end

        function layer = initialize(layer, layout)
            % Same RNG draws, in the same order, as the released layer.
            c = pipelineStudyCells(layer.StudyCell);
            H = layer.NumHiddenUnits;
            numChannels = layout.Size(finddim(layout, "C"));
            if isempty(layer.W1), layer.W1 = initializeGlorot([4*H, numChannels]); end
            if isempty(layer.W2), layer.W2 = initializeGlorot([4*H, numChannels]); end
            if isempty(layer.U),  layer.U  = initializeOrthogonal([4*H, H]); end
            if isempty(layer.Q),  layer.Q  = initializeOrthogonal([4*H, H]) * c.qScale; end
            if isempty(layer.b1)
                b = initializeUnitForgetGate(4, H);
                switch c.b1
                    case 'released', b(2*H:end, :) = b(2*H:end, :) * 0;     % as released (+1 on rows H+1:2H-1)
                    case 'forget',   b = b * 0; b(1:H, :) = 1;              % +1 on the forget gate rows
                    case 'zero',     b = b * 0;
                    otherwise, error('study:b1', 'unknown b1 mode %s', c.b1);
                end
                layer.b1 = b;
            end
            if isempty(layer.b2)
                b = initializeUnitForgetGate(4, H) * 0;
                for k = c.b2blk, b((k-1)*H+1:k*H, :) = c.b2val; end
                layer.b2 = b;
            end
        end

        function [Y, hiddenState, hiddenPhase, memoryReal, memoryImaginary] = predict(layer, X)
            [Y, hiddenState, hiddenPhase, memoryReal, memoryImaginary] = phaseVersionLayerRun(layer, X, false);
        end

        function [Y, hiddenState, hiddenPhase, memoryReal, memoryImaginary, memory] = forward(layer, X)
            [Y, hiddenState, hiddenPhase, memoryReal, memoryImaginary, memory] = phaseVersionLayerRun(layer, X, true);
        end

        function varargout = backward(layer, X, varargin)
            g = phaseVersionLayerGrads(layer, X, varargin);
            out = {g.X, g.W1, g.W2, g.U, g.Q, g.b1, g.b2};
            varargout = phaseVersionFinishGrads(out, g, X, nargout);
        end
    end
end
