classdef phaseAwareLSTM < nnet.layer.Layer & nnet.layer.Formattable & nnet.layer.Acceleratable

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
        function layer = phaseAwareLSTM(numHiddenUnits, args)
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

        G = Wx + Uh;               % magnitude preacts [4H B]
        Hh = Wp + Qx;              % phase preacts     [4H B]

        % gates (stable + strictly positive where you need them)
        sG   = max(sigmoid(G), epsilon);
        aH   = max(atan(Hh),   epsilon);

        % candidate memory
        real_hat = max(tanh(G(g3,:)), epsilon);
        imag_hat = aH(g3,:);

        % output gates
        Om = sG(g4,:);  Op = aH(g4,:);

        % rotation + scale (forget & input packed as two rows each)
        rotMag = [sG(g1,:); sG(g2,:)];
        theta  = [aH(g1,:); aH(g2,:)];

        x = [memoryReal;      real_hat];
        y = [memoryImaginary; imag_hat];

        [xrot, yrot] = layer.rotate_and_scale(x,y,rotMag,theta);

        memoryReal      = xrot(1:H,:) + xrot(H+1:end,:);
        memoryImaginary = yrot(1:H,:) + yrot(H+1:end,:);

        memMag   = sqrt(memoryReal.^2 + memoryImaginary.^2);
        memPhase = max(atan2(memoryImaginary, memoryReal), epsilon);
        outSign  = sign(memoryReal + memoryImaginary);

        hiddenState = max(Om .* tanh(outSign .* memMag), epsilon);
        hiddenPhase = max(Op .* memPhase, epsilon);

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
    

function [x_new, y_new] = rotate_and_scale(layer, x, y, r, theta)
% x,y,r,theta: [2H, B], same class/device as X (dlarray/gpuArray ok)
% Apply per-element complex rotation+scaling:
% z_new = r .* (cos(theta) + i*sin(theta)) .* (x + i*y)
    c = cos(theta);           % [2H,B]
    s = sin(theta);           % [2H,B]
    rx = r .* x;              % [2H,B]
    ry = r .* y;              % [2H,B]

    % Real:  r*(c*x - s*y)
    x_new = rx .* c - ry .* s;

    % Imag:  r*(s*x + c*y)
    y_new = rx .* s + ry .* c;
end

    end


end