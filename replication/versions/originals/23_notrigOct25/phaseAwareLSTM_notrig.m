classdef phaseAwareLSTM_ver10 < nnet.layer.Layer & nnet.layer.Formattable & nnet.layer.Acceleratable

    properties
        NumHiddenUnits
        OutputMode        
    end
    properties (Learnable)
        W1 % Input-to-hidden-magnitude weights: [4H, inputSize]        
        W2 % Input-to-hidden-phase weights: [4H, inputSize]
        W3
        U % Hidden-to-hidden-magnitude weights: [4H, H]
        Q % Hidden-to-hidden-phase weights: [4H, H]
        b1% magnitude bias
        b2% Phase bias
        b3
        % alpha
    end
    properties (State)
        hiddenMagnitude
        hiddenPhase
        memoryReal
        memoryImaginary        
    end
    methods
        function layer = phaseAwareLSTM_ver10(numHiddenUnits, args)
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
    
        function [Y, hiddenState, hiddenPhase, memoryReal, memoryImaginary] = predict(layer, X)
    % --- 1) Put X on the same device as params, then label it
    onGPU = isa(layer.W1,'gpuArray');
    if onGPU && ~isa(X,'gpuArray'), X = gpuArray(X); end
    X = dlarray(X,'CBT');

    % --- 2) Sizes AFTER labeling
    numHiddenUnits = layer.NumHiddenUnits;
    miniBatchSize  = size(X, finddim(X,"B"));
    numTimeSteps   = size(X, finddim(X,"T"));

    % --- 3) Scalar eps with correct type/device
    epsilon = cast(1e-9,'like',X);

    % --- 4) Precompute WX + b (all on same device)
    Xplain = stripdims(X);                   % [C B T]
    WX1    = pagemtimes(layer.W1, Xplain) + layer.b1;  % [4H B T]
    WX2    = pagemtimes(layer.W2, Xplain) + layer.b2;  % [4H B T]
    expectedBatchSize = size(WX1,2);

    % --- 5) Gate indices
    H  = numHiddenUnits;
    g1 = 1:H; g2 = H+1:2*H; g3 = 2*H+1:3*H; g4 = 3*H+1:4*H;

    % --- 6) Init states/Y on same device/type as X
    hiddenState     = zeros(H, expectedBatchSize, 'like', X);
    hiddenPhase     = zeros(H, expectedBatchSize, 'like', X);
    memoryReal      = zeros(H, expectedBatchSize, 'like', X);
    memoryImaginary = zeros(H, expectedBatchSize, 'like', X);
    Y               = zeros(H, miniBatchSize, numTimeSteps, 'like', X);

    % also update layer.State so trainNetwork can snapshot if needed
    layer.hiddenMagnitude  = hiddenState;
    layer.hiddenPhase      = hiddenPhase;
    layer.memoryReal       = memoryReal;
    layer.memoryImaginary  = memoryImaginary;

    % --- 7) Time loop
    for t = 1:numTimeSteps
        Wx = WX1(:,:,t);           % [4H B]
        Wp = WX2(:,:,t);           % [4H B]
        Uh = layer.U * hiddenState;
        Qx = layer.Q * hiddenPhase;

        B = min([size(Wx,2), size(Uh,2), size(Wp,2), size(Qx,2)]);
        Wx = Wx(:,1:B);  Wp = Wp(:,1:B);
        Uh = Uh(:,1:B);  Qx = Qx(:,1:B);

     G  = Wx + Uh;                 % magnitude preacts [4H B]
Hh = Wp + Qx;                 % phase preacts     [4H B]

% Gates
sG = sigmoid(G);              % [4H B], unitless in (0,1)

% Trig-free cos/sin for rows of Hh
den       = sqrt(1 + Hh.^2) + epsilon;   % [4H B]
cos_part  = 1 ./ den;                    % cos(atan(Hh))
sin_part  = Hh .* cos_part;              % sin(atan(Hh))

% Candidate memory (rectangular, unit-consistent)
real_hat = tanh(G(g3,:));     % unitless
imag_hat = tanh(Hh(g3,:));    % unitless

% Output gates
Om = sG(g4,:);                % unitless magnitude gate
Op = sigmoid(Hh(g4,:));       % unitless phase gate (safe to scale a phase)

% Rotation + scale (forget & input packed as two rows each) — trig-free
rotMag_f = sG(g1,:);   c_f = cos_part(g1,:);   s_f = sin_part(g1,:);
rotMag_i = sG(g2,:);   c_i = cos_part(g2,:);   s_i = sin_part(g2,:);

% Apply forget rotation to current memory
rx_f = rotMag_f .* memoryReal;      ry_f = rotMag_f .* memoryImaginary;
x_f  = rx_f .* c_f - ry_f .* s_f;   y_f  = rx_f .* s_f + ry_f .* c_f;

% Apply input rotation to candidate
rx_i = rotMag_i .* real_hat;        ry_i = rotMag_i .* imag_hat;
x_i  = rx_i .* c_i - ry_i .* s_i;   y_i  = rx_i .* s_i + ry_i .* c_i;

% Update memory
memoryReal      = x_f + x_i;
memoryImaginary = y_f + y_i;

        memMag   = sqrt(memoryReal.^2 + memoryImaginary.^2);
        memPhase = atan2(memoryImaginary, memoryReal);        

        hiddenState = Om .* tanh(memMag);
        hiddenPhase = memPhase+Op;


        if layer.OutputMode == "sequence"
            Y(:,1:B,t) = hiddenState(:,1:B);
        end
    end

    Y = dlarray(Y,'CBT');
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