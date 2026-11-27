classdef charmLayer < nnet.layer.Layer & nnet.layer.Formattable & nnet.layer.Acceleratable
    % CHARM-base (Saeed et al. 2020, Sec. 2.1.1, Eq. 3): Conv1D + global max-pool
    % keys per input channel, learned canonical embeddings, soft remap.
    % I/O: 'CBT' (Channels x Batch x Time)
    %
    % Ported unchanged from the Nov 2025 project (L:\plstm helperLibrary, 30 Oct 2025)
    % behind the paper's CHARM rows. Only this comment and the Description string
    % changed: the original called it CHARM-CQ, but it has no values, no attention
    % and no residual MLP, so it is CHARM-base. One conv layer (the paper's
    % CHARM-base has three, see charmLayer3Conv) and no input normalisation
    % (see the charmInstNorm variant).

    properties
        NumCanonical   % M
        EmbDim         % d
        FilterSize     % F (paper uses 8)
    end

    properties (Learnable)
        % Shared Conv1D bank over time, applied independently to each channel
        % Stored as plain arrays; we add labels at call-site:
        %   Wk: [F x 1 x d]  (format 'TCU': Time x inChannels x outChannels)
        %   bk: [1 x 1 x d]  (format 'U':    outChannels)
        Wk
        bk

        % Canonical queries: [M x d]
        Q
    end

    methods
        function layer = charmLayer(M, d, args)
            arguments
                M (1,1) double {mustBePositive,mustBeInteger}
                d (1,1) double {mustBePositive,mustBeInteger}
                args.FilterSize (1,1) double {mustBePositive,mustBeInteger} = 8
                args.Name = ""
            end
            layer.NumCanonical = M;
            layer.EmbDim = d;
            layer.FilterSize = args.FilterSize;
            layer.Name = args.Name;
            layer.Description = "CHARM-base (per-channel Conv1D + global max-pool; canonical embeddings)";
        end

        function layer = initialize(layer, layout)
            F = layer.FilterSize; d = layer.EmbDim; M = layer.NumCanonical;

            % Glorot-ish init (inChannels=1, outChannels=d)
            sW = sqrt(6/(F + d));
            W  = single((2*sW*rand(F,1,d) - sW));   % [F x 1 x d]
            b  = single(zeros(1,1,d));              % [1 x 1 x d]

            sQ = sqrt(6/(d + M));
            Q  = single((2*sQ*rand(M,d) - sQ));     % [M x d]

            if canUseGPU
                W = gpuArray(W); b = gpuArray(b); Q = gpuArray(Q);
            end
            layer.Wk = W; layer.bk = b; layer.Q = Q;
        end

        function Z = predict(layer, X)
            % X: dlarray 'CBT' [C x B x T]  -> Z: 'CBT' [M x B x T]
            C = size(X, finddim(X,"C"));
            B = size(X, finddim(X,"B"));
            T = size(X, finddim(X,"T")); %#ok<NASGU>
            M = layer.NumCanonical; %#ok<NASGU>

            % ---------- 1) mconv: per-channel Conv1D + GLOBAL MAX-POOL ----------
            % Fold channels into batch: [B*C x 1 x T]
            X_BCT = permute(stripdims(X), [2 1 3]);      % [B x C x T]
            X_B1T = reshape(X_BCT, B*C, 1, []);          % [B*C x 1 x T]

            % Convolve over time: give weights/bias proper formats
            W = dlarray(layer.Wk, 'TCU');                % [F x 1 x d]
            b = dlarray(layer.bk, '');                  % [1 x 1 x d]
            Y = dlconv(X_B1T, W, b, 'DataFormat','BCT', 'Stride',1, 'Padding','same'); % [B*C x d x T]

            % Global max-pool over time -> [B*C x d]
            Ymax = max(Y, [], 3);

            % Unfold to keys K: [d x C x B]
            K = reshape(Ymax, B, C, layer.EmbDim);       % [B x C x d]
            K = permute(K, [3 2 1]);                     % [d x C x B]

            % ---------- 2) Canonical queries & logits ----------
            logits = pagemtimes(layer.Q, K);             % Q[M x d] * K[d x C x B] -> [M x C x B]

            % ---------- 3) Softmax over channels (dim=2) ----------
            L = stripdims(logits);                       % numeric [M x C x B]
            L = L - max(L, [], 2);
            p = exp(L) ./ sum(exp(L), 2);                % [M x C x B]

            % ---------- 4) Mix raw channels uniformly across time ----------
            X_CTB = permute(stripdims(X), [1 3 2]);      % [C x T x B]
            Z_MTB = pagemtimes(p, X_CTB);                % [M x T x B]

            Z = dlarray(permute(Z_MTB, [1 3 2]), 'CBT'); % [M x B x T]
        end
    end
end
