classdef cstm_arch2 < nnet.layer.Layer & nnet.layer.Formattable & nnet.layer.Acceleratable

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
        function layer = cstm_arch2(numHiddenUnits, args)
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
            if isempty(layer.b2), layer.b2 = initializeUnitForgetGate(4,numHiddenUnits);layer.b2=layer.b2*pi/2;end
            
            % Initialize states
            if isempty(layer.hiddenMagnitude),layer.hiddenMagnitude = zeros(numHiddenUnits, 1, 1, "like", layer.W1);end
            if isempty(layer.hiddenPhase),layer.hiddenPhase = zeros(numHiddenUnits, 1, 1, "like", layer.U);end
            if isempty(layer.memoryReal), layer.memoryReal = zeros(numHiddenUnits, 1, 1, "like", layer.W1);end
            if isempty(layer.memoryImaginary), layer.memoryImaginary = zeros(numHiddenUnits, 1, 1, "like", layer.W2);end

            if canUseGPU,layer = layer.initializeGPUlayer();end



         end
    
        function [Y, hiddenState, hiddenPhase, memoryMagnitude, memoryPhase] = predict(layer, X)

    numHiddenUnits = layer.NumHiddenUnits;
    miniBatchSize = size(X, finddim(X, "B"));
    numTimeSteps = size(X, finddim(X, "T"));
    epsilon = 1e-9;

    X = dlarray(X, 'CBT');
    X = stripdims(X);

    WX1 = pagemtimes(layer.W1, X) + layer.b1;
    WX2 = pagemtimes(layer.W2, X) + layer.b2;
    expectedBatchSize = size(WX1, 2);

    % Gate indices
    idx1 = 1:numHiddenUnits;
    idx2 = numHiddenUnits+1:2*numHiddenUnits;
    idx3 = 2*numHiddenUnits+1:3*numHiddenUnits;
    idx4 = 3*numHiddenUnits+1:4*numHiddenUnits;

    % Initialize hidden and memory states (polar form)
    hiddenState = zeros(numHiddenUnits, expectedBatchSize, "like", X);
    hiddenPhase = zeros(numHiddenUnits, expectedBatchSize, "like", X);
    memoryMagnitude = zeros(numHiddenUnits, expectedBatchSize, "like", X);
    memoryPhase = zeros(numHiddenUnits, expectedBatchSize, "like", X);

    Y = zeros(numHiddenUnits, miniBatchSize, numTimeSteps, "like", X);

    for t = 1:numTimeSteps
        Wx = WX1(:,:,t);
        Wp = WX2(:,:,t);
        Uh = layer.U * hiddenState;
        Qx = layer.Q * hiddenPhase;

        % Ensure matching batch size
        minBatchSize = min(size(Wx,2), size(Uh,2));
        Wx = Wx(:,1:minBatchSize);
        Wp = Wp(:,1:minBatchSize);
        Uh = Uh(:,1:minBatchSize);
        Qx = Qx(:,1:minBatchSize);

        G = Wx + Uh;
        H = Wp + Qx;

        % Gate activations
        f_mag = max(sigmoid(G(idx1,:)), epsilon);
        f_phase = max(atan(H(idx1,:)), epsilon);
        i_mag = max(sigmoid(G(idx2,:)), epsilon);
        i_phase = max(atan(H(idx2,:)), epsilon);
        c_mag = max(tanh(G(idx3,:)), epsilon);
        c_phase = max(atan(H(idx3,:)), epsilon);
        o_mag = max(sigmoid(G(idx4,:)), epsilon);
        o_phase = max(atan(H(idx4,:)), epsilon);  % Optional: learnable phase gate

        % Memory update: polar multiplication followed by Cartesian addition
        % Forget branch: F_t * Z_{t-1}
        realF = f_mag .* memoryMagnitude .* cos(f_phase + memoryPhase);
        imagF = f_mag .* memoryMagnitude .* sin(f_phase + memoryPhase);

        % Input branch: I_t * \tilde{Z}_t
        realI = i_mag .* c_mag .* cos(i_phase + c_phase);
        imagI = i_mag .* c_mag .* sin(i_phase + c_phase);

        % Complex addition
        realMem = realF + realI;
        imagMem = imagF + imagI;

        % Update memory in polar form
        memoryMagnitude = sqrt(realMem.^2 + imagMem.^2);
        memoryPhase = atan2(imagMem, realMem + epsilon); % epsilon to avoid zero denominator

        % Output
        outputSign = sign(realMem + imagMem);
        hiddenState = max(o_mag .* tanh(outputSign .* memoryMagnitude), epsilon);
        hiddenPhase = max(o_phase .* memoryPhase, epsilon);

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
 

    end


end