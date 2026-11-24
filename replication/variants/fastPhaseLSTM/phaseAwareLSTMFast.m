classdef phaseAwareLSTMFast < nnet.layer.Layer & nnet.layer.Formattable & nnet.layer.Acceleratable
    % phaseAwareLSTMFast  (replication variant 'fastPhaseLSTM', opt-in)
    %
    % SPEED-ONLY copy of utils/modelstuff/layerdefinitions/phaseAwareLSTM.m.
    % The original file is not changed. Same properties, same initialize(),
    % same arithmetic in the same order at every time step. Only three
    % bookkeeping changes in predict():
    %
    %   1. Outputs are collected per step in a cell and joined once with
    %      cat(3, ...) at the end. The original writes Y(:,:,t) = h into a
    %      traced [H x B x T] dlarray at every step; under autodiff each of
    %      those writes is a full-size copy, in the forward and again in the
    %      backward pass, so the cost grows with T^2 (T = 500 for CWL).
    %   2. cat(1, A(idx1,:), A(idx2,:)) is written as A(1:2H,:): the same
    %      rows in the same order, two indexing ops and a cat fewer per step.
    %   3. Loop-invariant work (U, Q, the OutputMode string test) is done
    %      once before the loop instead of at every step.
    %
    % None of these changes a single value. replication/checkPhaseLSTMVersions.m
    % compares outputs and gradients against the original layer bit for bit
    % and times both.

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
        function layer = phaseAwareLSTMFast(numHiddenUnits, args)
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

                sigmoidG            = sigmoid(G);
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

                memoryReal      = realMemory;
                memoryImaginary = imaginaryMemory;

                memoryMagnitude = sqrt(realMemory.^2 + imaginaryMemory.^2);
                memoryMagnitude = max(memoryMagnitude, epsilon);
                memoryPhase     = atan2(imaginaryMemory, realMemory);

                hiddenState = Om .* tanh(memoryMagnitude);
                hiddenPhase = memoryPhase;

                if isSequence
                    Ys{t} = hiddenState;
                end
            end

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
