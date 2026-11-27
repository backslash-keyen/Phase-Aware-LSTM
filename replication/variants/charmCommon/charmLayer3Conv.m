classdef charmLayer3Conv < nnet.layer.Layer & nnet.layer.Formattable & nnet.layer.Acceleratable
    % CHARM-base with the paper's three-layer mconv (Saeed et al. 2020, Sec. 2.3.1:
    % "three convolutional layers with 32 feature maps and a filter size of 8
    % with a stride of 1"). Everything else is charmLayer (the ported CHARM-base):
    % per-channel keys by global max-pool over time, canonical embeddings Q [M x d],
    % p = softmax over input channels of Q*K, Z = p*X at every time step.
    %
    % Choices the paper leaves open, made here:
    %   * ReLU after conv 1 and conv 2; conv 3 is linear, then the max-pool
    %     (the same "linear filter, then max" as the one-layer charmLayer).
    %     Without activations three convolutions collapse to one linear filter.
    %   * 'same' padding and Glorot-style uniform init for every conv, written
    %     like charmLayer's: limit sqrt(6/(F*Cin + d)); biases 0.
    % With NumConv = 1 the layer is charmLayer exactly (same arithmetic and the
    % same order of rand calls), which replication/variants/charmCommon/checkCharmLayers.m tests.
    % I/O: 'CBT' (Channels x Batch x Time)

    properties
        NumCanonical   % M
        EmbDim         % d
        FilterSize     % F
        NumConv        % 1 or 3
    end

    properties (Learnable)
        Wk             % [F x 1 x d]
        bk             % [1 x 1 x d]
        Wk2            % [F x d x d]  (NumConv = 3)
        bk2
        Wk3            % [F x d x d]  (NumConv = 3)
        bk3
        Q              % [M x d]
    end

    methods
        function layer = charmLayer3Conv(M, d, args)
            arguments
                M (1,1) double {mustBePositive,mustBeInteger}
                d (1,1) double {mustBePositive,mustBeInteger}
                args.FilterSize (1,1) double {mustBePositive,mustBeInteger} = 8
                args.NumConv (1,1) double {mustBeMember(args.NumConv,[1 3])} = 3
                args.Name = ""
            end
            layer.NumCanonical = M;
            layer.EmbDim = d;
            layer.FilterSize = args.FilterSize;
            layer.NumConv = args.NumConv;
            layer.Name = args.Name;
            layer.Description = sprintf("CHARM-base, %d-layer mconv (Conv1D x%d + global max-pool; canonical embeddings)", ...
                args.NumConv, args.NumConv);
        end

        function layer = initialize(layer, layout) %#ok<INUSD>
            if ~isempty(layer.Wk) && ~isempty(layer.Q), return; end   % keep trained weights
            F = layer.FilterSize; d = layer.EmbDim; M = layer.NumCanonical;

            sW = sqrt(6/(F + d));
            W  = single((2*sW*rand(F,1,d) - sW));
            b  = single(zeros(1,1,d));

            sQ = sqrt(6/(d + M));
            Q  = single((2*sQ*rand(M,d) - sQ));

            if layer.NumConv == 3
                s2 = sqrt(6/(F*d + d));
                W2 = single((2*s2*rand(F,d,d) - s2));
                W3 = single((2*s2*rand(F,d,d) - s2));
            else
                W2 = single(zeros(1,1,1)); W3 = W2;   % unused
            end
            b2 = single(zeros(1,1,d)); b3 = b2;

            if canUseGPU
                W = gpuArray(W); b = gpuArray(b); Q = gpuArray(Q);
                W2 = gpuArray(W2); b2 = gpuArray(b2); W3 = gpuArray(W3); b3 = gpuArray(b3);
            end
            layer.Wk = W; layer.bk = b; layer.Q = Q;
            layer.Wk2 = W2; layer.bk2 = b2; layer.Wk3 = W3; layer.bk3 = b3;
        end

        function Z = predict(layer, X)
            C = size(X, finddim(X,"C"));
            B = size(X, finddim(X,"B"));

            X_BCT = permute(stripdims(X), [2 1 3]);      % [B x C x T]
            X_B1T = reshape(X_BCT, B*C, 1, []);          % [B*C x 1 x T]

            W = dlarray(layer.Wk, 'TCU');
            b = dlarray(layer.bk, '');
            Y = dlconv(X_B1T, W, b, 'DataFormat','BCT', 'Stride',1, 'Padding','same');  % [B*C x d x T]
            if layer.NumConv == 3
                Y = relu(Y);
                Y = dlconv(Y, dlarray(layer.Wk2,'TCU'), dlarray(layer.bk2,''), 'DataFormat','BCT', 'Stride',1, 'Padding','same');
                Y = relu(Y);
                Y = dlconv(Y, dlarray(layer.Wk3,'TCU'), dlarray(layer.bk3,''), 'DataFormat','BCT', 'Stride',1, 'Padding','same');
            end

            Ymax = max(Y, [], 3);                        % [B*C x d]
            K = reshape(Ymax, B, C, layer.EmbDim);
            K = permute(K, [3 2 1]);                     % [d x C x B]

            logits = pagemtimes(layer.Q, K);             % [M x C x B]
            L = stripdims(logits);
            L = L - max(L, [], 2);
            p = exp(L) ./ sum(exp(L), 2);

            X_CTB = permute(stripdims(X), [1 3 2]);
            Z_MTB = pagemtimes(p, X_CTB);
            Z = dlarray(permute(Z_MTB, [1 3 2]), 'CBT');
        end
    end
end
