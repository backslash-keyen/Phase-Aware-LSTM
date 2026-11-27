classdef charmLayerLowMem < nnet.layer.Layer & nnet.layer.Formattable
    % CHARM-base (the ported charmLayer), same model and parameters, with a
    % hand-written backward pass that never holds the full conv output.
    %
    % Why: charmLayer runs its conv on every channel of every trial and keeps the
    % output [B*C x d x T] for autodiff. With the 61-channel source folds and the
    % source mini-batch (B = 197, d = 32, T = 500) that is 0.77 GB per copy, and the
    % backward pass needs several, so the charmSourceTrained source run ran this PC
    % out of memory. Only the max over time is used, so:
    %   forward   conv in chunks of rows, keep the max and its time index only
    %   backward  the max routes the key gradient to one time step per (row, filter),
    %             so dWk, dbk and dX need only the 8-sample input window there
    % Same rand calls in initialize as charmLayer, so the same seed gives the same
    % weights. checkCharm compares output and gradients with charmLayer (autodiff).
    % I/O: 'CBT' (Channels x Batch x Time)

    properties
        NumCanonical   % M
        EmbDim         % d
        FilterSize     % F
        ChunkRows = 2000   % rows (trial x channel) per chunk (2000 x 500 x 40 floats = 160 MB)
    end

    properties (Learnable)
        Wk             % [F x 1 x d]
        bk             % [1 x 1 x d]
        Q              % [M x d]
    end

    methods
        function layer = charmLayerLowMem(M, d, args)
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
            layer.Description = "CHARM-base (charmLayer with a memory-light backward pass)";
        end

        function layer = initialize(layer, layout) %#ok<INUSD>
            % Only a fresh layer is initialised. charmLayer re-draws its weights on
            % every initialize call, which is harmless for a new CHARM but would wipe
            % the source-trained CHARM when trainNetwork starts the target run.
            if ~isempty(layer.Wk) && ~isempty(layer.Q), return; end
            F = layer.FilterSize; d = layer.EmbDim; M = layer.NumCanonical;
            sW = sqrt(6/(F + d));
            W  = single((2*sW*rand(F,1,d) - sW));
            b  = single(zeros(1,1,d));
            sQ = sqrt(6/(d + M));
            Q  = single((2*sQ*rand(M,d) - sQ));
            if canUseGPU
                W = gpuArray(W); b = gpuArray(b); Q = gpuArray(Q);
            end
            layer.Wk = W; layer.bk = b; layer.Q = Q;
        end

        function Z = predict(layer, X)
            [Z, ~] = forward(layer, X);
        end

        function [Z, mem] = forward(layer, X)
            Xn = numericOf(X);                          % [C x B x T]
            [C, B, T] = size(Xn, 1:3);
            Xr = reshape(permute(Xn, [2 1 3]), B*C, T); % row r = b + (c-1)*B, as in charmLayer
            [Kr, idx] = layer.keys(Xr);                 % [B*C x d]
            K = permute(reshape(Kr, B, C, layer.EmbDim), [3 2 1]);   % [d x C x B]
            p = softmaxC(pagemtimes(numericOf(layer.Q), K));         % [M x C x B]
            Z = pagemtimes(p, permute(Xn, [1 3 2]));                 % [M x T x B]
            Z = dlarray(permute(Z, [1 3 2]), 'CBT');
            mem = struct('idx', idx, 'K', K, 'p', p, 'T', T);
        end

        function [dX, dW, db, dQ] = backward(layer, X, ~, dZ, mem)
            Xn = numericOf(X); dZn = numericOf(dZ);
            [C, B, T] = size(Xn, 1:3);
            d = layer.EmbDim; F = layer.FilterSize;
            Qn = numericOf(layer.Q); Wn = reshape(numericOf(layer.Wk), F, d);
            p = mem.p; K = mem.K;

            % mixing Z = p * X
            X_CTB  = permute(Xn, [1 3 2]);              % [C x T x B]
            dZ_MTB = permute(dZn, [1 3 2]);             % [M x T x B]
            dp  = pagemtimes(dZ_MTB, 'none', X_CTB, 'transpose');    % [M x C x B]
            dXm = pagemtimes(p, 'transpose', dZ_MTB, 'none');        % [C x T x B]

            % softmax over channels, logits = Q * K
            dL = p .* (dp - sum(p .* dp, 2));
            dQ = sum(pagemtimes(dL, 'none', K, 'transpose'), 3);     % [M x d]
            dK = pagemtimes(Qn, 'transpose', dL, 'none');            % [d x C x B]
            dKr = reshape(permute(dK, [3 2 1]), B*C, d);             % [B*C x d]

            % keys: K(r,k) = bk(k) + sum_f W(f,k) * X(r, t*(r,k) + f - 1 - padL)
            Xr = reshape(permute(Xn, [2 1 3]), B*C, T);
            R = B*C; padL = floor((F - 1)/2);
            dW = zeros(F, d, 'like', Wn);
            dXr = zeros(R, T, 'like', Xr);
            rows = (1:R)';
            for k = 1:d
                g = dKr(:, k);
                for f = 1:F
                    s = mem.idx(:, k) + (f - 1 - padL);
                    ok = s >= 1 & s <= T;
                    lin = rows(ok) + (s(ok) - 1)*R;
                    dW(f, k) = sum(g(ok) .* Xr(lin));
                    dXr(lin) = dXr(lin) + g(ok) * Wn(f, k);   % each (r,t) at most once per (k,f)
                end
            end
            db = reshape(sum(dKr, 1), 1, 1, d);
            dW = reshape(dW, F, 1, d);
            dXk = permute(reshape(dXr, B, C, T), [2 1 3]);           % [C x B x T]
            dX = permute(dXm, [1 3 2]) + dXk;                         % [C x B x T]
            % trainNetwork expects dlarray outputs; dX keeps the input's format
            if isa(X, 'dlarray') && ~isempty(dims(X)), dX = dlarray(dX, dims(X)); else, dX = dlarray(dX); end
            dW = dlarray(dW); db = dlarray(db); dQ = dlarray(dQ);
        end
    end

    methods (Access = private)
        function [Kr, idx] = keys(layer, Xr)
            % conv as a matrix product over the F shifted copies of each row
            % ('same' padding, padL = floor((F-1)/2) zeros on the left, as dlconv),
            % so no convolution library is loaded (cuDNN's conv kernels cost
            % about 1.9 GB of host memory on this PC)
            [R, T] = size(Xr); d = layer.EmbDim; F = layer.FilterSize;
            padL = floor((F - 1)/2); padR = F - 1 - padL;
            W = reshape(numericOf(layer.Wk), F, d);
            b = reshape(numericOf(layer.bk), 1, d);
            Kr = zeros(R, d, 'like', Xr); idx = zeros(R, d);          % double indices on the CPU
            for r0 = 1:layer.ChunkRows:R
                r = r0:min(R, r0 + layer.ChunkRows - 1); n = numel(r);
                Xp = [zeros(n, padL, 'like', Xr), Xr(r, :), zeros(n, padR, 'like', Xr)];
                S = zeros(n, T, F, 'like', Xr);
                for f = 1:F, S(:, :, f) = Xp(:, f:f+T-1); end
                Y = reshape(reshape(S, n*T, F) * W, n, T, d) + reshape(b, 1, 1, d);   % [n x T x d]
                [m, i] = max(Y, [], 2);
                Kr(r, :) = reshape(m, n, d); idx(r, :) = gather(double(reshape(i, n, d)));
            end
        end
    end
end

function A = numericOf(A)
if isa(A, 'dlarray'), A = extractdata(stripdims(A)); end
end

function p = softmaxC(L)
L = L - max(L, [], 2);
p = exp(L) ./ sum(exp(L), 2);
end
