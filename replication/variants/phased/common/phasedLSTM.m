classdef phasedLSTM < nnet.layer.Layer & nnet.layer.Formattable & nnet.layer.Acceleratable
    % Phased LSTM baseline with the same IO contract as your cstm_arch1.
    % X: [C B T], step through X(:,:,t). Outputs:
    %   Y (CBT), hiddenState(H,B), hiddenPhase(H,B)=zeros,
    %   memoryReal(H,B)=cell state, memoryImaginary(H,B)=zeros.

    properties
        NumHiddenUnits
        OutputMode
        Ron (1,1) double = 0.05        % open ratio
        LeakAlpha (1,1) double = 1e-3   % training leak (set 0 for test)
        DefaultLeakAlpha (1,1) double = 1e-3
    end

    properties (Learnable)
        W1   % [4H, inputSize]
        U    % [4H, H]
        b1   % [4H, 1]
        tau  % [H, 1]  period (learnable, >0)
        shi  % [H, 1]  phase shift (learnable, in [0, tau] at init)
    end

    properties (State)
        hiddenMagnitude
        hiddenPhase
        memoryReal
        memoryImaginary
    end

    methods
        function layer = phasedLSTM(numHiddenUnits, args)
            arguments
                numHiddenUnits
                args.Name = ""
                args.OutputMode = "sequence"
                args.Ron (1,1) double = 0.01
                args.LeakAlpha (1,1) double = 0
            end
            layer.NumHiddenUnits = numHiddenUnits;
            layer.Name = args.Name;
            layer.OutputMode = args.OutputMode;
            layer.Ron = args.Ron;
            layer.LeakAlpha = args.LeakAlpha;
            layer.DefaultLeakAlpha = args.LeakAlpha;
            layer.Description = "Phased-LSTM (" + numHiddenUnits + " units; τ log-uniform init)";
        end

        function layer = initialize(layer, layout)
            H = layer.NumHiddenUnits;
            inDim = layout.Size(finddim(layout, "C"));

            % Learnables (core LSTM)
            if isempty(layer.W1), layer.W1 = initializeGlorot([4*H, inDim]); end
            if isempty(layer.U),  layer.U  = initializeOrthogonal([4*H, H]); end
            if isempty(layer.b1), layer.b1 = initializeUnitForgetGate(4, H); end

            % ---- Phased-LSTM timing params (log-uniform τ) ----
            if isempty(layer.tau)
                u = rand(H,1,"like",layer.W1) * 1.0;   % U(0,3)
                layer.tau = exp(u);                    % log-uniform in steps
            end
            if isempty(layer.shi)
                r = rand(H,1,"like",layer.W1);
                layer.shi = r .* layer.tau;            % s ∈ [0, τ]
            end

            % States
            if isempty(layer.hiddenMagnitude),  layer.hiddenMagnitude  = zeros(H,1, "like", layer.W1); end
            if isempty(layer.hiddenPhase),      layer.hiddenPhase      = zeros(H,1, "like", layer.W1); end
            if isempty(layer.memoryReal),       layer.memoryReal       = zeros(H,1, "like", layer.W1); end
            if isempty(layer.memoryImaginary),  layer.memoryImaginary  = zeros(H,1, "like", layer.W1); end

            if canUseGPU, layer = layer.initializeGPUlayer(); end
        end

        function [Y, hiddenState, hiddenPhase, memoryReal, memoryImaginary] = predict(layer, X)
            epsilon = 1e-9;
            H = layer.NumHiddenUnits;

            % Expect X as [C B T]
            X = dlarray(X,'CBT');   % keep dlarray for pagemtimes
            X = stripdims(X);

            B = size(X,2);
            T = size(X,3);

            % Precompute WX + b across time
            WX = pagemtimes(layer.W1, X) + layer.b1;  % [4H, B, T]

            % Gate slices: [i; f; g; o]
            idx1 = 1:H;              % i
            idx2 = H+1:2*H;          % f
            idx3 = 2*H+1:3*H;        % g (candidate)
            idx4 = 3*H+1:4*H;        % o

            % Init states per-batch (like tau to be safe on GPU/single)
            h = zeros(H, B, 'like', layer.tau);
            c = zeros(H, B, 'like', layer.tau);

            % Outputs
            Y = zeros(H, B, T, 'like', layer.tau);

            for t = 1:T
                % Core LSTM affine
                w  = WX(:,:,t);                      
                Uh = layer.U * h;                   
                G  = w + Uh;                        

                % Gates
                i = sigmoid(G(idx1,:));   % sigmoid clamp
                f = sigmoid(G(idx2,:));
                g = tanh(G(idx3,:));
                o = sigmoid(G(idx4,:));

                % Proposed updates
                c_prop = f .* c + i .* g;
                h_prop = o .* tanh(c_prop);

                % Time gate k_t (per-unit), using step index t
                k = layer.computeTimeGate(t, B); % [H,B], like tau

                % Phased mix
                c = k .* c_prop + (1 - k) .* c;
                h = k .* h_prop + (1 - k) .* h;

                if layer.OutputMode == "sequence"
                    Y(:,:,t) = h;
                end
            end

            hiddenState      = h;                               % H×B
            hiddenPhase      = zeros(H, B, "like", layer.tau);  % unused
            memoryReal       = c;                               % cell state
            memoryImaginary  = zeros(H, B, "like", layer.tau);  % unused

            Y = dlarray(Y,'CBT');
        end

        function layer = setTestMode(layer)
            layer.LeakAlpha = 0;
        end
        function layer = setTrainMode(layer)
            layer.LeakAlpha = layer.DefaultLeakAlpha;
        end

        function layer = initializeGPUlayer(layer)
            layer.W1 = gpuArray(layer.W1);
            layer.U  = gpuArray(layer.U);
            layer.b1 = gpuArray(layer.b1);
            layer.tau = gpuArray(layer.tau);
            layer.shi = gpuArray(layer.shi);

            layer.hiddenMagnitude = gpuArray(layer.hiddenMagnitude);
            layer.hiddenPhase     = gpuArray(layer.hiddenPhase);
            layer.memoryReal      = gpuArray(layer.memoryReal);
            layer.memoryImaginary = gpuArray(layer.memoryImaginary);
        end
    end

    methods (Access=private)
    function k = computeTimeGate(layer, t, B)
        % Triangular time gate with leak, using step index t.
        % k: [H,B], built 'like' tau to be GPU/precision-safe and avoid dlarray ops.

        H = layer.NumHiddenUnits;

        % --- Get plain numeric (or gpuArray) copies of learnables ---
        tau = layer.tau;
        if isa(tau,'dlarray'), tau = extractdata(tau); end
        shi = layer.shi;
        if isa(shi,'dlarray'), shi = extractdata(shi); end

        % Guards & constants in the SAME type/device as tau
        onev = ones(1,'like',tau);           % scalar 1 on CPU/GPU, single/double
        tiny = eps(onev);                    % eps in that type
        % tau  = max(abs(tau), tiny);          % τ > 0
        % shi  = min(max(shi, 0), tau);        % clamp s ∈ [0, τ]

        % φ_t = ((t - s) mod τ) / τ ∈ [0,1)
        tt   = cast(t, 'like', tau);         % scalar t as same type/device
        phi  = mod((tt - shi)./tau, onev);   % [H,1]    (avoid dlarray 'mod')

        % Triangular gate with leak α
        ron  = layer.Ron;
        half = 0.1*ron;

        kvec = zeros(H,1,'like',tau);
        m1   = (phi < half);
        m2   = (~m1) & (phi < ron);
        kvec(m1)        = phi(m1)/ron;
        kvec(m2)        = 3 - phi(m2)/ron;
        kvec(~(m1|m2))  = layer.LeakAlpha * phi(~(m1|m2));

        % replicate across batch
        k = repmat(kvec, 1, B);              % [H,B]
    end
end

end

% ===== helpers (keep identical to your project) =====
function X = initializeGlorot(sz)
fanIn  = sz(2);
fanOut = sz(1);
limit  = sqrt(6/(fanIn+fanOut));
X = (rand(sz,'single')*2 - 1) * limit;
end

function X = initializeOrthogonal(sz)
A = randn(sz,'single');
[U,~,V] = svd(A,'econ');
X = U * V';
X = single(X);
end

function b = initializeUnitForgetGate(numGates, H)
b = zeros(numGates*H,1,'single');
fIx = (H+1):(2*H);
b(fIx) = 1.0;  % forget gate bias
end
