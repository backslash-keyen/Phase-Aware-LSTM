classdef phaseAwareLSTM_ver1 < nnet.layer.Layer & nnet.layer.Formattable & nnet.layer.Acceleratable

    properties
        NumHiddenUnits
        OutputMode        
    end
    properties (Learnable)
        W1 % Input-to-hidden-magnitude weights: [4H, inputSize]        
        W2 % Input-to-hidden-phase weights: [4H, inputSize]        
        U % Hidden-to-hidden-magnitude weights: [4H, H]
        Q % Hidden-to-hidden-phase weights: [4H, H]
        b1% magnitude bias
        b2% Phase bias
        % alpha
    end
    properties (State)
        hiddenMagnitude
        hiddenPhase
        memoryReal
        memoryImaginary        
    end
    methods
        function layer = phaseAwareLSTM_ver1(numHiddenUnits, args)
            % Constructor for Euler-LSTM layer
            arguments
                numHiddenUnits
                args.Name = "";
                args.OutputMode = "sequence";
            end

            % Set layer properties
            layer.NumHiddenUnits = numHiddenUnits;
            layer.Name = args.Name;
            layer.OutputMode = args.OutputMode;

            % Set layer description
            layer.Description = "Euler-LSTM with " + numHiddenUnits + " hidden units";
        end

        function layer = initialize(layer, layout)
            % Determine input and ouput size
            numHiddenUnits = layer.NumHiddenUnits;
            idx = finddim(layout, "C");
            numChannels = layout.Size(idx);

            % Initialize learnable parameters
            if isempty(layer.W1),layer.W1 = initializeGlorot([4 * numHiddenUnits, numChannels]);end
            if isempty(layer.W2),layer.W2 = initializeGlorot([4 * numHiddenUnits, numChannels]);end
            if isempty(layer.U),layer.U = initializeOrthogonal([4 * numHiddenUnits, numHiddenUnits]);end
            if isempty(layer.Q),layer.Q = initializeOrthogonal([4 *numHiddenUnits, numHiddenUnits]);end
            if isempty(layer.b1), layer.b1 = initializeUnitForgetGate(4,numHiddenUnits);end
            if isempty(layer.b2), layer.b2 = initializeUnitForgetGate(4,numHiddenUnits);layer.b2=layer.b2*pi;end
            
            % Initialize states
            if isempty(layer.hiddenMagnitude),layer.hiddenMagnitude = zeros(numHiddenUnits, 1, 1, "like", layer.W1);end
            if isempty(layer.hiddenPhase),layer.hiddenPhase = zeros(numHiddenUnits, 1, 1, "like", layer.U);end
            if isempty(layer.memoryReal), layer.memoryReal = zeros(numHiddenUnits, 1, 1, "like", layer.W1);end
            if isempty(layer.memoryImaginary), layer.memoryImaginary = zeros(numHiddenUnits, 1, 1, "like", layer.W2);end

            if canUseGPU,layer = layer.initializeGPUlayer();end



         end
    
        function [Y, hiddenState,hiddenPhase,memoryReal,memoryImaginary] = predict(layer, X)

            numHiddenUnits = layer.NumHiddenUnits;
            miniBatchSize = size(X, finddim(X, "B"));
            numTimeSteps = size(X, finddim(X, "T"));
            
            epsilon = 1e-9;

            
            X = dlarray(X, 'CBT');
            % X = gpuArray(X);
            % layer = initializeGPUlayer(layer)
            

            % Compute WX + b
            X = stripdims(X);
            WX1 = pagemtimes(layer.W1, X) + layer.b1;
            WX2 = pagemtimes(layer.W2, X) + layer.b2;
            expectedBatchSize = size(WX1, 2);
            
            % Define gate indices
            idx1 = 1:numHiddenUnits;
            idx2 = 1*numHiddenUnits+1:2*numHiddenUnits;
            idx3 = 2*numHiddenUnits+1:3*numHiddenUnits;
            idx4 = 3*numHiddenUnits+1:4*numHiddenUnits;

            layer.hiddenMagnitude = zeros(numHiddenUnits, expectedBatchSize, "like", X);
            layer.hiddenPhase = zeros(numHiddenUnits, expectedBatchSize, "like", X);
            layer.memoryReal = zeros(numHiddenUnits, expectedBatchSize, "like", X);
            layer.memoryImaginary = zeros(numHiddenUnits, expectedBatchSize, "like", X);
            Y = zeros(numHiddenUnits, miniBatchSize,numTimeSteps, "like", X);

             % Initialize
            hiddenState = layer.hiddenMagnitude;
            hiddenPhase = layer.hiddenPhase;
            memoryReal = layer.memoryReal;
            memoryImaginary = layer.memoryImaginary;

            for t = 1:numTimeSteps
                % Compute recurrent contributions
                Wx = WX1(:,:,t);
                Wp = WX2(:,:,t);
                Uh = layer.U * hiddenState; 
                Qx = layer.Q * hiddenPhase;

                minBatchSize = min(size(Wx,2), size(Uh,2));
                Uh = Uh(:,1:minBatchSize);
                Wx = Wx(:,1:minBatchSize);
                Wp = Wp(:,1:minBatchSize);
                Qx = Qx(:,1:minBatchSize);

                G = Wx + Uh; % [4H, B]  → Precompute sum for all gates
                H = Wp + Qx; % [4H, B]  → Precompute sum for all gates
               

                sigmoidG = sigmoid(G);
                atanH = max(atan(H),epsilon);
                tanhG = tanh(G(idx3,:));
                tanhH = atanH(idx3,:);

% Candidate Memory
realMemory_hat = tanhG;
imaginaryMemory_hat = tanhH;


% Output Gate
Om = sigmoidG(idx4,:);
Op = atanH(idx4,:);


    rotationMag = cat(1,sigmoidG(idx1,:),sigmoidG(idx2,:));    
    theta = cat(1,atanH(idx1,:),atanH(idx2,:));
    
    x = cat(1,memoryReal,realMemory_hat);
    y = cat(1,memoryImaginary,imaginaryMemory_hat);

    [x_rot,y_rot] = layer.rotate_and_scale(x,y,rotationMag,theta);
    
    realMemory = x_rot(1:layer.NumHiddenUnits,:) + x_rot(layer.NumHiddenUnits+1:end,:);
    imaginaryMemory = y_rot(1:layer.NumHiddenUnits,:) + y_rot(layer.NumHiddenUnits+1:end,:);


               
                memoryMagnitude = sqrt(realMemory.^2 + imaginaryMemory.^2);               
                memoryPhase = atan2(imaginaryMemory, realMemory);
                
                
                outputSign = sign(realMemory+imaginaryMemory);
                
                hiddenState = Om.* tanh(outputSign.*memoryMagnitude);
                hiddenPhase = Op.*memoryPhase;


                            % Store outputs

                            % Store outputs
                if layer.OutputMode == "sequence"
                     Y(:,:,t) = hiddenState;                     
                end

            end            
            Y = dlarray(Y, 'CBT');
            

        end

        function layer = initializeGPUlayer(layer)
        layer.W1 = gpuArray(layer.W1);
        layer.W2 = gpuArray(layer.W2);
        layer.U = gpuArray(layer.U);
        layer.Q = gpuArray(layer.Q);
        layer.b1 = gpuArray(layer.b1);
        layer.b2 = gpuArray(layer.b2);
        
    
        layer.hiddenMagnitude = gpuArray(layer.hiddenMagnitude);
        layer.hiddenPhase = gpuArray(layer.hiddenPhase);
        layer.memoryReal = gpuArray(layer.memoryReal);
        layer.memoryImaginary = gpuArray(layer.memoryImaginary);
    end
    

function [x_new, y_new] = rotate_and_scale(layer, x, y, r, theta)
    % Fully vectorized and GPU-optimized rotation & scaling using pagemtimes
    % Inputs:
    %   x, y   - Matrices containing real and imaginary parts [H, B]
    %   theta  - Rotation angles in radians [H, B]
    %   r      - Scaling factors [H, B]
    % Outputs:
    %   x_new, y_new - Rotated and scaled values [H, B]
    
    
    trigMatrix = cat(3,times(r,cos(theta)),times(r,sin(theta)));
    
    xdash = times(trigMatrix,x);
    ydash = times(trigMatrix,y);

    x_new = xdash(:,:,1) - ydash(:,:,2);
    y_new = xdash(:,:,2) + ydash(:,:,1);

end

    end


end