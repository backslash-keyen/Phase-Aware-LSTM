classdef phaseAwareLSTM < nnet.layer.Layer & nnet.layer.Formattable & nnet.layer.Acceleratable
    % phaseAwareLSTM
    %
    % A custom recurrent layer implementing a Phase-Aware Long Short-Term Memory
    % (LSTM) unit. The layer decomposes the recurrent state into:
    %   - a magnitude pathway (real-valued, gated like a standard LSTM), and
    %   - a phase pathway (angular dynamics using a half-angle Weierstrass
    %     parameterization to generate norm-preserving rotations).
    %
    % INPUT FORMAT
    %   X : dlarray with format 'CBT'
    %       C — number of input channels
    %       B — mini-batch size
    %       T — number of time steps
    %
    % OUTPUT FORMAT
    %   If OutputMode = "sequence":
    %        Y : dlarray 'CBT'  (C = NumHiddenUnits)
    %            returns hidden state at every time step
    %
    %   If OutputMode = "last":
    %        Y : dlarray 'CB'   (C = NumHiddenUnits)
    %            returns only the final hidden state
    %
    % PURPOSE
    %   The layer models oscillatory or phase-sensitive temporal structure
    %   by embedding recurrent memory in a 2-D (real, imaginary) representation.
    %   Magnitude is controlled via LSTM-style gating, while phase evolves through
    %   stable, differentiable rotations without trig functions in the inner loop.
    %
    % APPLICATIONS
    %   Suitable for time-series tasks where phase information is meaningful:
    %   EEG, speech, radar, biomedical oscillatory signals, or any model
    %   requiring complex-valued recurrent behavior in a purely real-domain
    %   implementation.



    properties
        NumHiddenUnits        % Number of hidden units H
        OutputMode            % "sequence" or "last"
    end

    properties (Learnable)
        W1 % Input-to-hidden magnitude weights: [4H, inputSize]        
        W2 % Input-to-hidden phase weights: [4H, inputSize]        
        U  % Hidden-to-hidden magnitude weights: [4H, H]        
        Q  % Hidden-to-hidden phase weights: [4H, H]        
        b1 
        b2 % Bias terms for magnitude and phase branches: [4H, 1]
    end

    properties (State)
        % (Stored for interpretability)        

        hiddenMagnitude    % Final hidden magnitude 
        hiddenPhase        % Final hidden phase (stored for interpretability)        
        memoryReal         % Final real part of internal memory state        
        memoryImaginary    % Final imaginary part of internal memory state
    end

    methods
        function layer = phaseAwareLSTM(numHiddenUnits, args)
            % Constructor for Phase-Aware LSTM layer.
            %
            % phaseAwareLSTM(H, Name=name, OutputMode=mode)
            %
            % numHiddenUnits : number of hidden units H
            % Name           : layer name in layerGraph
            % OutputMode     : "sequence" (default) or "last"

            arguments
                numHiddenUnits
                args.Name = "";
                args.OutputMode = "sequence";
            end

            % Set layer properties
            layer.NumHiddenUnits = numHiddenUnits;
            layer.Name           = args.Name;
            layer.OutputMode     = args.OutputMode;

            % Description shown in network summary
            layer.Description = "Phase-Aware LSTM with " + numHiddenUnits + " hidden units";
        end

        function layer = initialize(layer, layout)
            % INITIALIZE
            % Called by dlnetwork/trainNetwork for parameter initialization.
            % Uses Glorot and orthogonal initializations for weights and a
            % unit-forget-gate initialization for b1/b2.

            % Determine input channel dimension C from layout
            numHiddenUnits = layer.NumHiddenUnits;
            idx            = finddim(layout, "C");
            numChannels    = layout.Size(idx);

            % Initialize learnable parameters (magnitude & phase branches)
            if isempty(layer.W1), layer.W1 = initializeGlorot([4 * numHiddenUnits, numChannels]);end
            if isempty(layer.W2), layer.W2 = initializeGlorot([4 * numHiddenUnits, numChannels]);end
            if isempty(layer.U), layer.U  = initializeOrthogonal([4 * numHiddenUnits, numHiddenUnits]);end
            if isempty(layer.Q), layer.Q  = initializeOrthogonal([4 * numHiddenUnits, numHiddenUnits]);end

            % Magnitude bias: unit-forget on first block, zero elsewhere
            if isempty(layer.b1), layer.b1 = initializeUnitForgetGate(4, numHiddenUnits);layer.b1(2*numHiddenUnits:end,:) = layer.b1(2*numHiddenUnits:end,:)*0;end

            % Phase bias: initialized to zero
            if isempty(layer.b2), layer.b2 = initializeUnitForgetGate(4, numHiddenUnits);layer.b2 = layer.b2 * 0;end
        end

        function [Y, hiddenState,hiddenPhase,memoryReal,memoryImaginary] = predict(layer, X)
            % PREDICT
            %
            % Forward pass through the Phase-Aware LSTM.
            %
            % Inputs:
            %   X : dlarray with format 'CBT' (channels, batch, time)
            %
            % Outputs:
            %   Y               : dlarray (CBT for "sequence", CB for "last")
            %   hiddenState     : final hidden magnitude [H x B]
            %   hiddenPhase     : final hidden phase     [H x B]
            %   memoryReal      : final memory real      [H x B]
            %   memoryImaginary : final memory imaginary [H x B]

            epsilon        = 1e-9; % Constant used for numerical precision
            numHiddenUnits = layer.NumHiddenUnits;
            miniBatchSize  = size(X, finddim(X, "B"));
            numTimeSteps   = size(X, finddim(X, "T"));

            % Ensure formatted dlarray
            X = dlarray(X, 'CBT');

            % Strip labels for internal matmul ops: X → [C x B x T]
            X   = stripdims(X);
            WX1 = pagemtimes(layer.W1, X) + layer.b1;  % [4H x B x T]
            WX2 = pagemtimes(layer.W2, X) + layer.b2;  % [4H x B x T]

            % -----------------------------------------------------------------
            % Allocate per-call state (recurrent across time)
            % -----------------------------------------------------------------
            hiddenState     = zeros(numHiddenUnits, miniBatchSize, 'like', X);
            hiddenPhase     = zeros(numHiddenUnits, miniBatchSize, 'like', X);
            memoryReal      = zeros(numHiddenUnits, miniBatchSize, 'like', X);
            memoryImaginary = zeros(numHiddenUnits, miniBatchSize, 'like', X);

            % Allocate output tensor depending on OutputMode
            if layer.OutputMode == "sequence"
                % Return all time steps: [H x B x T]
                Y = zeros(numHiddenUnits, miniBatchSize, numTimeSteps, 'like', X);
            elseif layer.OutputMode == "last"
                % Return only final time step: [H x B]
                Y = zeros(numHiddenUnits, miniBatchSize, 'like', X);
            else
                error("OutputMode must be 'sequence' or 'last'.");
            end

            % Gate index ranges
            idx1 = 1:numHiddenUnits;
            idx2 = 1*numHiddenUnits+1:2*numHiddenUnits;
            idx3 = 2*numHiddenUnits+1:3*numHiddenUnits;
            idx4 = 3*numHiddenUnits+1:4*numHiddenUnits;

            % -----------------------------------------------------------------
            % Time-step recurrence
            % -----------------------------------------------------------------
            for t = 1:numTimeSteps
                % Recurrent contributions from previous hidden state & phase
                Wx = WX1(:,:,t);             % [4H x B]
                Wp = WX2(:,:,t);             % [4H x B]
                Uh = layer.U * hiddenState;  % [4H x B]
                Qx = layer.Q * hiddenPhase;  % [4H x B]

                % Magnitude and phase pre-activations
                G = Wx + Uh;                 % magnitude branch pre-acts
                H = Wp + Qx;                 % phase branch pre-acts

                % Clip pre-activations to keep sigmoids/tanh/rotations stable
                G = max(min(G, 20), -20);
                H = max(min(H, 20), -20);

                % -----------------------------------------------------------------
                % Gate magnitudes (LSTM-style) and candidate memory
                % -----------------------------------------------------------------
                sigmoidG            = sigmoid(G);
                realMemory_hat      = tanh(G(idx3,:)); % candidate real part
                imaginaryMemory_hat = tanh(H(idx3,:)); % candidate imag part

                % Output gate magnitude (acts on memory magnitude)
                Om = sigmoidG(idx4,:);       % [H x B]

                % -----------------------------------------------------------------
                % Phase-aware rotation: build rotation magnitudes and angles
                % -----------------------------------------------------------------
                rotationMag = cat(1, sigmoidG(idx1,:), sigmoidG(idx2,:)); % [2H x B]

                % Phase preactivation for forget & input gates: [2H x B]
                Htheta = cat(1, H(idx1,:), H(idx2,:));

                % Half-angle mapping via Weierstrass parameterization
                %   t = tan(theta/2) ≈ tanh(0.5 * Htheta)
                Htheta = tanh(0.5 * Htheta);
                denom    = 1 + Htheta.^2 + epsilon;

                % cos(theta) and sin(theta) from t = tan(theta/2)
                costheta = (1 - Htheta.^2) ./ denom;
                sintheta = (2 .* Htheta)    ./ denom;

                % Optional re-normalization to enforce exact unit norm
                r        = sqrt(costheta.^2 + sintheta.^2 + epsilon);
                costheta = costheta ./ r;
                sintheta = sintheta ./ r;

                % -----------------------------------------------------------------
                % Rotate and scale complex memory: z_new = r * e^{iθ} ⊙ z
                % -----------------------------------------------------------------
                x = cat(1, memoryReal,      realMemory_hat);      % [2H x B]
                y = cat(1, memoryImaginary, imaginaryMemory_hat); % [2H x B]

                [xNew, yNew] = layer.rotate_and_scale_notrig(x, y, rotationMag, costheta, sintheta);

                realMemory      = xNew(1:layer.NumHiddenUnits,:) + xNew(layer.NumHiddenUnits+1:end,:);
                imaginaryMemory = yNew(1:layer.NumHiddenUnits,:) + yNew(layer.NumHiddenUnits+1:end,:);

                % Carry memory state forward
                memoryReal      = realMemory;
                memoryImaginary = imaginaryMemory;

                % -----------------------------------------------------------------
                % Convert complex memory -> (magnitude, phase)
                % -----------------------------------------------------------------
                memoryMagnitude = sqrt(realMemory.^2 + imaginaryMemory.^2);
                memoryMagnitude = max(memoryMagnitude, epsilon);  % avoid exact zeros
                memoryPhase     = atan2(imaginaryMemory, realMemory);

                % Hidden output: magnitude + phase
                hiddenState = Om .* tanh(memoryMagnitude);   % [H x B]
                hiddenPhase = memoryPhase;                   % [H x B]

                % -----------------------------------------------------------------
                % Store outputs according to OutputMode
                % -----------------------------------------------------------------
                if layer.OutputMode == "sequence"
                    % Store per time-step output
                    Y(:,:,t) = hiddenState;
                elseif layer.OutputMode == "last" && t == numTimeSteps
                    % Overwrite Y only at final time step
                    Y(:,:) = hiddenState;
                end
            end

            % Wrap outputs back into dlarray with appropriate format
            if layer.OutputMode == "sequence"
                % [C x B x T]
                Y = dlarray(Y, 'CBT');
            elseif layer.OutputMode == "last" && t == numTimeSteps
                % [C x B]
                Y = dlarray(Y, 'CB');
            end
        end

        function [Y, hiddenState,hiddenPhase,memoryReal,memoryImaginary] = forward(layer,X)
            % FORWARD
            % Wrapper used by dlnetwork / trainNetwork.
            % - Calls predict(...) for actual computation.
            % - Writes final states into layer.State properties for
            %   interpretability / inspection (not for recurrence).

            [Y, hiddenState,hiddenPhase,memoryReal,memoryImaginary] = predict(layer, X);
        end

        function [xNew, yNew] = rotate_and_scale_notrig(~, x, y, magnitudeScale, costheta, sintheta)
            % ROTATE_AND_SCALE_NOTRIG
            %
            % Compute:
            %   (x_new + i y_new) = magnitudeScale .* (cosθ + i sinθ) ⊙ (x + i y)
            %
            % Inputs:
            %   x, y           : [2H x B] real & imag parts of stacked states
            %   magnitudeScale : [2H x B] radial scaling r
            %   costheta       : [2H x B] cosθ
            %   sintheta       : [2H x B] sinθ
            %
            % Outputs:
            %   x_new, y_new   : rotated & scaled real/imag parts [2H x B]

            xMag = magnitudeScale .* x;
            yMag = magnitudeScale .* y;

            % Re{ r e^{iθ} z } and Im{ r e^{iθ} z }
            xNew = xMag .* costheta - yMag .* sintheta;
            yNew = xMag .* sintheta + yMag .* costheta;
        end

    end
end
