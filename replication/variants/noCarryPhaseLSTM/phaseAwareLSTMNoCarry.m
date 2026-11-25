classdef phaseAwareLSTMNoCarry < nnet.layer.Layer & nnet.layer.Formattable & nnet.layer.Acceleratable
    % phaseAwareLSTMNoCarry  (replication variant 'noCarryPhaseLSTM', opt-in DIAGNOSTIC)
    %
    % NOT the released model. Copy of phaseAwareLSTMFast (bit-identical to the
    % released phaseAwareLSTM.m) with the two June 2025 cstm_arch1 behaviours
    % that bound the backward pass:
    %   1. No memory carry: the complex memory is NOT passed to the next step
    %      (cstm_arch1 never reassigns memoryReal/memoryImaginary in its loop),
    %      so every step rotates [0; candidate]. Recurrence only via U*h, Q*phase.
    %   2. Clamps at epsilon as in cstm_arch1: gates max(sigmoid(G), eps) and the
    %      fed-back phase max(atan2(cI, cR), eps).
    % Everything else (initialisation, +-20 clip, tan-half-angle rotation,
    % unsigned output h = o.*tanh(|c|)) is as released. Gradients by autodiff.

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
        function layer = phaseAwareLSTMNoCarry(numHiddenUnits, args)
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
            epsilon        = 1e-9;
            numHiddenUnits = layer.NumHiddenUnits;
            miniBatchSize  = size(X, finddim(X, "B"));
            numTimeSteps   = size(X, finddim(X, "T"));

            X   = dlarray(X, 'CBT');
            X   = stripdims(X);
            WX1 = pagemtimes(layer.W1, X) + layer.b1;
            WX2 = pagemtimes(layer.W2, X) + layer.b2;

            hiddenState     = zeros(numHiddenUnits, miniBatchSize, 'like', X);
            hiddenPhase     = zeros(numHiddenUnits, miniBatchSize, 'like', X);
            memoryReal      = zeros(numHiddenUnits, miniBatchSize, 'like', X);
            memoryImaginary = zeros(numHiddenUnits, miniBatchSize, 'like', X);

            isSequence = layer.OutputMode == "sequence";
            if ~isSequence && layer.OutputMode ~= "last"
                error("OutputMode must be 'sequence' or 'last'.");
            end
            if isSequence
                Ys = cell(1, numTimeSteps);
            end

            U = layer.U;
            Q = layer.Q;
            H1 = numHiddenUnits;
            idx12 = 1:2*H1;               % = [idx1 idx2]
            idx3  = 2*H1+1:3*H1;
            idx4  = 3*H1+1:4*H1;
            top   = 1:H1;
            bot   = H1+1:2*H1;            % = H+1:end of a [2H x B] array

            for t = 1:numTimeSteps
                Wx = WX1(:,:,t);
                Wp = WX2(:,:,t);
                Uh = U * hiddenState;
                Qx = Q * hiddenPhase;

                G = Wx + Uh;
                H = Wp + Qx;

                G = max(min(G, 20), -20);
                H = max(min(H, 20), -20);

                sigmoidG            = max(sigmoid(G), epsilon);        % June clamp
                realMemory_hat      = tanh(G(idx3,:));
                imaginaryMemory_hat = tanh(H(idx3,:));

                Om = sigmoidG(idx4,:);

                rotationMag = sigmoidG(idx12,:);
                Htheta      = H(idx12,:);

                Htheta = tanh(0.5 * Htheta);
                denom    = 1 + Htheta.^2 + epsilon;

                costheta = (1 - Htheta.^2) ./ denom;
                sintheta = (2 .* Htheta)    ./ denom;

                r        = sqrt(costheta.^2 + sintheta.^2 + epsilon);
                costheta = costheta ./ r;
                sintheta = sintheta ./ r;

                x = cat(1, memoryReal,      realMemory_hat);
                y = cat(1, memoryImaginary, imaginaryMemory_hat);

                % rotate_and_scale_notrig, inlined (same expressions)
                xMag = rotationMag .* x;
                yMag = rotationMag .* y;
                xNew = xMag .* costheta - yMag .* sintheta;
                yNew = xMag .* sintheta + yMag .* costheta;

                realMemory      = xNew(top,:) + xNew(bot,:);
                imaginaryMemory = yNew(top,:) + yNew(bot,:);

                % June: no memory carry (memoryReal/memoryImaginary stay 0)

                memoryMagnitude = sqrt(realMemory.^2 + imaginaryMemory.^2);
                memoryMagnitude = max(memoryMagnitude, epsilon);
                memoryPhase     = max(atan2(imaginaryMemory, realMemory), epsilon);   % June clamp

                hiddenState = Om .* tanh(memoryMagnitude);
                hiddenPhase = memoryPhase;

                if isSequence
                    Ys{t} = hiddenState;
                end
            end

            memoryReal      = realMemory;        % state outputs: last step's memory
            memoryImaginary = imaginaryMemory;
            if isSequence
                Y = dlarray(cat(3, Ys{:}), 'CBT');
            else
                Y = dlarray(hiddenState, 'CB');
            end
        end

        function [Y, hiddenState,hiddenPhase,memoryReal,memoryImaginary] = forward(layer,X)
            [Y, hiddenState,hiddenPhase,memoryReal,memoryImaginary] = predict(layer, X);
        end
    end
end
