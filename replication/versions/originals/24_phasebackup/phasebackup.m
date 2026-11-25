classdef phaseAwareLSTM_ver5 < nnet.layer.Layer & nnet.layer.Formattable & nnet.layer.Acceleratable
    % Phase-aware LSTM (ver5) — single-output predict/forward (returns Y only).
    % Input X must be dlarray with labels 'CBT'  [C x B x T]
    % Output Y is dlarray 'CBT'                   [H x B x T] or [H x B x 1]

    properties
        NumHiddenUnits
        OutputMode    % "sequence" or "last"
    end

    properties (Learnable)
        W1  % [4H, C]  input-to-gates (magnitude)
        W2  % [4H, C]  input-to-gates (phase proxy)
        U   % [4H, H]  hidden-to-gates (magnitude)
        Q   % [4H, H]  hidden-to-gates (phase proxy)
        b1  % [4H x 1 x 1] biases for W1/U
        b2  % [4H x 1 x 1] biases for W2/Q
    end

    methods
        function layer = phaseAwareLSTM_ver5(numHiddenUnits, args)
            arguments
                numHiddenUnits (1,1) double {mustBePositive,mustBeInteger}
                args.Name = ""
                args.OutputMode = "sequence"
            end
            layer.NumHiddenUnits = numHiddenUnits;
            layer.Name           = args.Name;
            layer.OutputMode     = string(args.OutputMode);
            layer.Description    = "Phase-aware LSTM (ver5, single-output)";
        end

        function layer = initialize(layer, layout)
            H = layer.NumHiddenUnits;
            C = layout.Size(finddim(layout,"C"));

            % Weights
            if isempty(layer.W1), layer.W1 = initializeGlorot([4*H, C]); end
            if isempty(layer.W2), layer.W2 = initializeGlorot([4*H, C]); end
            if isempty(layer.U),  layer.U  = initializeOrthogonal([4*H, H]); end
            if isempty(layer.Q),  layer.Q  = initializeOrthogonal([4*H, H]); end

            % Biases -> [4H x 1 x 1] (keep simple/robust; no accidental idx typos)
            if isempty(layer.b1)
                b1 = single(initializeUnitForgetGate(4, H));   % [4H x 1]
                b1 = reshape(b1, [], 1, 1);                    % [4H x 1 x 1]
                layer.b1 = b1;
            end
            if isempty(layer.b2)
                b2 = single(zeros(4*H,1));                     % neutral for phase path
                b2 = reshape(b2, [], 1, 1);                    % [4H x 1 x 1]
                layer.b2 = b2;
            end

            if canUseGPU, layer = layer.initializeGPUlayer(); end
        end

   function Y = predict(layer, X)
    % X : dlarray('CBT')  [C x B x T]
    H = layer.NumHiddenUnits;
    B = size(X, finddim(X,"B"));
    T = size(X, finddim(X,"T"));
    idx1 = 1:H; idx2 = H+1:2*H; idx3 = 2*H+1:3*H; idx4 = 3*H+1:4*H;

    % ---------- toggles ----------
    debug      = true;             % set false when training for speed
    log_t      = unique([1, max(1,floor(T/2)), T]);
    gateNorm   = true;             % per-gate batch-centering/normalization
    epsLike    = cast(1e-6,'like',X);
    tauAngle   = cast(1e-3,'like',X);  % floor for unit-vector norm

    % ---------- helpers (dlarray-safe) ----------
    clamp      = @(A,lo,hi) min(max(A,lo),hi);
    sig_stable = @(Z) 1 ./ (1 + exp(-clamp(Z,-8,8)));     % stable sigmoid
    sanitize   = @(A) localReplaceNonfinite(A);           % zero NaN/Inf
    unitCS     = @(qx,wp) localSafeUnit(qx,wp,tauAngle);  % (cos,sin)

    % unlabeled view for all math to avoid label overhead
    Xu = sanitize(stripdims(X));                          % [C x B x T]

    % precompute drives over time
    WX1 = sanitize(pagemtimes(layer.W1, Xu) + layer.b1);  % [4H x B x T]
    WX2 = sanitize(pagemtimes(layer.W2, Xu) + layer.b2);  % [4H x B x T]

    % recurrent state buffers (like X)
    h   = zeros(H, B, 'like', Xu);  % hiddenState (magnitude output)
    ph  = zeros(H, B, 'like', Xu);  % hiddenPhase (kept internal)
    mr  = zeros(H, B, 'like', Xu);  % memory real
    mi  = zeros(H, B, 'like', Xu);  % memory imag

    % output allocation
    if layer.OutputMode == "sequence"
        Ynum = zeros(H, B, T, 'like', Xu);
    else
        Ynum = zeros(H, B, 1, 'like', Xu);
    end

    % optional one-line device/type banner
    if debug
        fprintf('[phaseLSTM] device=%s | H=%d B=%d T=%d\n', underlyingType(X), H, B, T);
    end

    % ------- time loop -------
    for t = 1:T
        Wx = sanitize(WX1(:,:,t));                  % [4H x B]
        Wp = sanitize(WX2(:,:,t));                  % [4H x B]
        Uh = sanitize(layer.U * h);                 % [4H x B]
        Qx = sanitize(layer.Q * ph);                % [4H x B]

        G = sanitize(Wx + Uh);                      % [4H x B]
        if gateNorm
            mu  = mean(G, 2);
            sg  = std(G, 0, 2) + epsLike;
            G   = (G - mu) ./ sg;
        end

        sigG  = sig_stable(G);                      % [4H x B]
        m_hat = sanitize(tanh(G(idx3,:)));          % [H x B]

        % unit directions from (Qx, Wp)
        [c1,s1] = unitCS(Qx(idx1,:), Wp(idx1,:));   % memory rotation
        [c2,s2] = unitCS(Qx(idx2,:), Wp(idx2,:));   % candidate rotation
        [c3,s3] = unitCS(Qx(idx3,:), Wp(idx3,:));   % candidate orientation

        % candidate (Cartesian)
        rhat = sanitize(m_hat .* c3);               % [H x B]
        ihat = sanitize(m_hat .* s3);               % [H x B]

        % rotate+scale memory and candidate
        g1 = sigG(idx1,:); g2 = sigG(idx2,:);
        mr = sanitize( g1 .* (mr .* c1 - mi .* s1) );
        mi = sanitize( g1 .* (mr .* s1 + mi .* c1) );  % uses updated mr; OK (in-place)

        cr = sanitize( g2 .* (rhat .* c2 - ihat .* s2) );
        ci = sanitize( g2 .* (rhat .* s2 + ihat .* c2) );

        % update memory
        mr = sanitize(mr + cr);
        mi = sanitize(mi + ci);

        % output gate + readout
        Om = sigG(idx4,:);
        mag = sanitize( sqrt(mr.^2 + mi.^2) );
        h   = sanitize( Om .* (mag ./ (1 + abs(mag))) );

        % optionally keep a clamped phase (not used for output)
        % phi = atan2(mi, mr);
        % ph  = 0.5 * atan2( sin(2*phi), cos(2*phi) ); % clamp to [-pi/2, pi/2]
        % For now keep phase neutral so Qx stays bounded initially:
        ph = sanitize(ph);

        % minimal debug snapshots (scalar gathers only)
        if debug && any(t == log_t)
            gmean = @(A) double(gather(mean(A,"all")));
            gstd  = @(A) double(gather(std(A,0,"all")));
            gmin  = @(A) double(gather(min(A,[],"all")));
            gmax  = @(A) double(gather(max(A,[],"all")));
            gnan  = @(A) double(gather(sum(~isfinite(A),"all")));
            fprintf('  t=%4d | sigG1 μ=%.3f σ=%.3f [% .3f,% .3f] | bad=%g\n', ...
                t, gmean(sigG(idx1,:)), gstd(sigG(idx1,:)), gmin(sigG(idx1,:)), gmax(sigG(idx1,:)), gnan(sigG(idx1,:)));
            fprintf('         sigG2 μ=%.3f σ=%.3f | sigG3 μ=%.3f σ=%.3f | sigG4 μ=%.3f σ=%.3f\n', ...
                gmean(sigG(idx2,:)), gstd(sigG(idx2,:)), gmean(sigG(idx3,:)), gstd(sigG(idx3,:)), gmean(sigG(idx4,:)), gstd(sigG(idx4,:)));
            fprintf('         m_hat μ=% .3f σ=%.3f [% .3f,% .3f] | Om μ=%.3f σ=%.3f\n', ...
                gmean(m_hat), gstd(m_hat), gmin(m_hat), gmax(m_hat), gmean(Om), gstd(Om));
            fprintf('         |mem|  μ=%.3f σ=%.3f [% .3f,% .3f]\n', gmean(mag), gstd(mag), gmin(mag), gmax(mag));
        end

        if layer.OutputMode == "sequence"
            Ynum(:,:,t) = h;
        elseif t == T
            Ynum(:,:,1) = h;
        end
    end

    % wrap once
    Y = dlarray(Ynum,'CBT');

    % -------- local helpers --------
    function A = localReplaceNonfinite(A)
        mask = ~isfinite(A);
        if any(mask(:)), A(mask) = 0; end
    end
    function [c,s] = localSafeUnit(qx, wp, tau)
        % normalized (c,s) ~ (qx,wp)/norm, with default (1,0) if norm < tau
        den = sqrt(qx.*qx + wp.*wp);
        m   = den >= tau;
        c   = zeros(size(qx),'like',qx);
        s   = zeros(size(wp),'like',wp);
        if any(m(:))
            d = den(m);
            c(m) = qx(m)./d;
            s(m) = wp(m)./d;
        end
        if any(~m(:))
            c(~m) = 1; s(~m) = 0;
        end
    end
end


            Y = dlarray(Y, 'CBT');

            % ---- local helpers ----
            function A = replaceNonfinite(A)
                mask = ~isfinite(A);
                if any(mask(:)), A(mask) = 0; end
            end
            function [c,s] = localSafeUnit(x,y,th)
                % Returns unit (c,s) ~ (x,y)/norm with default (1,0) when norm<th
                den = sqrt(x.*x + y.*y);
                m   = den >= th;
                c   = zeros(size(x),'like',x);
                s   = zeros(size(y),'like',y);
                if any(m(:))
                    d = den(m);
                    c(m) = x(m)./d;
                    s(m) = y(m)./d;
                end
                if any(~m(:))
                    c(~m) = 1; s(~m) = 0;  % default no-rotation
                end
            end
        end

        function Y = forward(layer, X)
            % Same as predict (single output for trainNetwork)
            Y = predict(layer, X);
        end

        function layer = initializeGPUlayer(layer)
            if canUseGPU
                layer.W1 = gpuArray(layer.W1);
                layer.W2 = gpuArray(layer.W2);
                layer.U  = gpuArray(layer.U);
                layer.Q  = gpuArray(layer.Q);
                layer.b1 = gpuArray(layer.b1);
                layer.b2 = gpuArray(layer.b2);
            end
        end
    end
end
