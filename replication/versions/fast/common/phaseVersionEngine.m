classdef phaseVersionEngine
    % phaseVersionEngine  Shared fused forward/backward for every fast phase-layer version.
    %
    % Every phase-aware LSTM version found in Keyen's folders (see
    % replication/versions/README.md) is the same recurrence with different
    % choices. A version is described by a config struct (see
    % phaseVersionEngine.config) and its own small layer class in
    % replication/versions/fast/<id>/ calls run() and grads() below.
    %
    % Per time step, for unit u and batch element b (gate rows f i c o):
    %   g  = W1 x + b1 + U h            (clipped at +-clip)
    %   w  = W2 x + b2,  q = Q phase,  p = w + q (clipped at +-clip)
    %   xp = W3 x + b3                  (only the atan2 'phantom' version)
    %   f, i, o = sigmoid(g_f,i,o), gR = tanh(g_c)      (gateEps: max(., eps))
    %   A_k = angle map of row k        (atan, max(atan,eps), atan(p+eps),
    %         pi*tanh, pi*sigmoid, p, half-angle 2*atan(tanh(k p)),
    %         atan2(q,w), atan2(w,q), atan2(w+q, xp+q))
    %   candidate = gR + 1i*A_c  |  gR + 1i*tanh(p_c)  |  gR*exp(1i*A_c)
    %   c = carry*f*exp(1i*A_f)*c_prev + i*exp(1i*A_i)*candidate
    %   h = o .* tanh(s .* |c|), s = 1 | sign(Re c + Im c) | sign(Re c * Im c)
    %   phase = atan2(Im c, Re c [+xoff])  (phMemEps: max(., eps)), then
    %           phase | A_o.*phase | phase + A_o | phase + sigmoid(p_o)
    %           (phOutEps: max(., eps)); h may be floored at eps (hEps)
    % Speed: one matmul [U 0; 0 Q]*[h; phase] and one GPU arrayfun kernel per
    % step forward; one kernel and one matmul per step backward; weight
    % gradients as a few large matmuls at the end. The same kernels run on
    % the CPU (arrayfun), so a CPU check validates the GPU code.
    % Gradients are hand-written; they were checked against finite
    % differences for every version config (replication/versions/README.md).

    methods (Static)
        function cfg = config(varargin)
            % Defaults = released half-angle layer except the bias init.
            p = inputParser;
            addParameter(p, 'map', 'atan');        % see mapId
            addParameter(p, 'k', 0.5);             % half-angle factor
            addParameter(p, 'clip', Inf);
            addParameter(p, 'gateEps', false);
            addParameter(p, 'imag', 'tanh');       % 'angle' | 'tanh' | 'polar'
            addParameter(p, 'carry', true);
            addParameter(p, 'sign', 'none');       % 'none' | 'sum' | 'prod'
            addParameter(p, 'hEps', false);
            addParameter(p, 'phMemEps', false);
            addParameter(p, 'phase', 'mem');       % 'mem' | 'OpTimesMem' | 'memPlusOp' | 'memPlusSigmoid'
            addParameter(p, 'phOutEps', false);
            addParameter(p, 'magFloor', 0);
            addParameter(p, 'xoff', 0);
            parse(p, varargin{:}); r = p.Results;
            maps   = {'atan','atanEps','atanPlusEps','piTanh','piSigmoid','direct','halfAngle','atan2QW','atan2WQ','atan2Phantom'};
            imags  = {'angle','tanh','polar'};
            signs  = {'none','sum','prod'};
            phases = {'mem','OpTimesMem','memPlusOp','memPlusSigmoid'};
            cfg.mapName = r.map;
            cfg.v = [find(strcmp(maps, r.map)), r.k, r.clip, r.gateEps, ...
                     find(strcmp(imags, r.imag)), r.carry, find(strcmp(signs, r.sign)) - 1, ...
                     r.hEps, r.phMemEps, find(strcmp(phases, r.phase)), r.phOutEps, r.magFloor, r.xoff];
            assert(numel(cfg.v) == 13, 'phaseVersionEngine:config', 'Unknown option value.');
            cfg.v = double(cfg.v);
        end

        function [Y, hS, phS, cRS, cIS, memory] = run(cfg, outputMode, X, W1, W2, W3, b1, b2, b3, U, Q, keep)
            % X [C x B x T] numeric; parameters numeric; W3/b3 may be empty.
            H = size(U, 2);
            [C, B, T] = size(X, 1:3);
            if isempty(W3), W3 = zeros(4*H, C, 'like', W1); end
            if isempty(b3), b3 = zeros(4*H, 1, 'like', W1); end
            Wst = [W1; W2; W3];                              % [12H x C]
            bst = [b1(:); b2(:); b3(:)];
            R   = [U, zeros(4*H, H, 'like', U); zeros(4*H, H, 'like', U), Q];   % [8H x 2H]
            WX  = pagemtimes(Wst, X) + bst;                  % [12H x B x T]
            hp  = zeros(2*H, B, 'like', WX);                 % [h; phase]
            cR  = zeros(H, B, 'like', WX); cI = cR;
            HPs = cell(1, T+1); CRs = cell(1, T+1); CIs = cell(1, T+1); Ms = cell(1, T);
            HPs{1} = hp; CRs{1} = cR; CIs{1} = cI;
            c = num2cell(cfg.v);
            if ~isa(WX, 'gpuArray')   % CPU arrayfun does not expand scalars (GPU arrayfun does)
                c = cellfun(@(s) repmat(cast(s, 'like', WX), H, B), c, 'UniformOutput', false);
            end
            r = @(A, k) A((k-1)*H+1:k*H, :);
            for t = 1:T
                M  = R * hp;
                Wt = WX(:,:,t);
                G  = Wt(1:4*H,:) + M(1:4*H,:);
                Wp = Wt(4*H+1:8*H,:); Qx = M(4*H+1:8*H,:); Xp = Wt(8*H+1:12*H,:);
                [h, ph, cR, cI] = arrayfun(@fwdK, r(G,1), r(G,2), r(G,3), r(G,4), ...
                    r(Wp,1), r(Wp,2), r(Wp,3), r(Wp,4), r(Qx,1), r(Qx,2), r(Qx,3), r(Qx,4), ...
                    r(Xp,1), r(Xp,2), r(Xp,3), r(Xp,4), cR, cI, c{:});
                hp = [h; ph];
                HPs{t+1} = hp; CRs{t+1} = cR; CIs{t+1} = cI;
                if keep, Ms{t} = M; end
            end
            HP = cat(3, HPs{:});
            if outputMode == "sequence"
                Y = HP(1:H,:,2:end);
            else
                Y = hp(1:H,:);
            end
            hS = hp(1:H,:); phS = hp(H+1:end,:); cRS = cR; cIS = cI;
            if keep
                memory = struct('WX', WX, 'M', {Ms}, 'HP', HP, 'CR', cat(3, CRs{:}), ...
                                'CI', cat(3, CIs{:}), 'R', R, 'Wst', Wst);
            else
                memory = [];
            end
        end

        function g = grads(cfg, outputMode, X, dY, dS, memory)
            % dY: [H x B x T] (sequence) or [H x B] (last); dS: {dh, dphase, dRe, dIm} (may be empty)
            WX = memory.WX; HP = memory.HP; CR = memory.CR; CI = memory.CI; R = memory.R;
            H = size(R, 2) / 2;
            [C, B, T] = size(X, 1:3);
            z = zeros(H, B, 'like', WX);
            for k = 1:4, if isempty(dS{k}), dS{k} = z; end, end
            dh = dS{1}; dph = dS{2}; dcR = dS{3}; dcI = dS{4};
            isSeq = outputMode == "sequence";
            c = num2cell(cfg.v);
            if ~isa(WX, 'gpuArray')   % CPU arrayfun does not expand scalars (GPU arrayfun does)
                c = cellfun(@(s) repmat(cast(s, 'like', WX), H, B), c, 'UniformOutput', false);
            end
            r = @(A, k) A((k-1)*H+1:k*H, :);
            dWs = cell(1, T); dRs = cell(1, T);
            Rt = R.';
            for t = T:-1:1
                if isSeq
                    dh = dh + dY(:,:,t);
                elseif t == T
                    dh = dh + dY;
                end
                M  = memory.M{t};
                Wt = WX(:,:,t);
                G  = Wt(1:4*H,:) + M(1:4*H,:);
                Wp = Wt(4*H+1:8*H,:); Qx = M(4*H+1:8*H,:); Xp = Wt(8*H+1:12*H,:);
                [dG1, dG2, dG3, dG4, dW1, dW2, dW3, dW4, dQ1, dQ2, dQ3, dQ4, ...
                 dX1, dX2, dX3, dX4, dcRp, dcIp] = arrayfun(@bwdK, r(G,1), r(G,2), r(G,3), r(G,4), ...
                    r(Wp,1), r(Wp,2), r(Wp,3), r(Wp,4), r(Qx,1), r(Qx,2), r(Qx,3), r(Qx,4), ...
                    r(Xp,1), r(Xp,2), r(Xp,3), r(Xp,4), CR(:,:,t), CI(:,:,t), dh, dph, dcR, dcI, c{:});
                dG = [dG1; dG2; dG3; dG4];
                dQ = [dQ1; dQ2; dQ3; dQ4];
                dWs{t} = [dG; dW1; dW2; dW3; dW4; dX1; dX2; dX3; dX4];   % [12H x B] rows of Wst
                dRs{t} = [dG; dQ];                                      % [8H x B] rows of R
                dhp = Rt * dRs{t};
                dh = dhp(1:H,:); dph = dhp(H+1:end,:);
                dcR = dcRp; dcI = dcIp;
            end
            dWall = cat(3, dWs{:});                      % [12H x B x T]
            dW2d  = reshape(dWall, 12*H, B*T);
            dR2d  = reshape(cat(3, dRs{:}), 8*H, B*T);
            X2    = reshape(X, C, B*T);
            dWst  = dW2d * X2.';
            dbst  = sum(dW2d, 2);
            hPrev = reshape(HP(1:H,:,1:T),     H, B*T);
            pPrev = reshape(HP(H+1:2*H,:,1:T), H, B*T);
            g.W1 = dWst(1:4*H,:);       g.W2 = dWst(4*H+1:8*H,:);   g.W3 = dWst(8*H+1:12*H,:);
            g.b1 = dbst(1:4*H);         g.b2 = dbst(4*H+1:8*H);     g.b3 = dbst(8*H+1:12*H);
            g.U  = dR2d(1:4*H,:) * hPrev.';
            g.Q  = dR2d(4*H+1:8*H,:) * pPrev.';
            g.X  = pagemtimes(memory.Wst.', dWall);
        end
    end
end

% =====================================================================
% Scalar kernels (GPU arrayfun compiles each into one CUDA kernel)
% =====================================================================
function [a, dp, dw, dq, dx] = amapK(m, k, p, w, q, x)
% angle map and its derivatives w.r.t. the summed p, or the parts w, q, x
EPS = 1e-9;
dp = 0 * p; dw = dp; dq = dp; dx = dp;
if m == 1
    a = atan(p); dp = 1 / (1 + p * p);
elseif m == 2
    a = atan(p); dp = 1 / (1 + p * p);
    if a <= EPS, a = EPS + 0 * p; dp = 0 * p; end
elseif m == 3
    a = atan(p + EPS); dp = 1 / (1 + (p + EPS) * (p + EPS));
elseif m == 4
    t = tanh(p); a = pi * t; dp = pi * (1 - t * t);
elseif m == 5
    s = 1 / (1 + exp(-p)); a = pi * s; dp = pi * s * (1 - s);
elseif m == 6
    a = p; dp = 1 + 0 * p;
elseif m == 7
    t = tanh(k * p); a = 2 * atan(t); dp = 2 * k * (1 - t * t) / (1 + t * t);
elseif m == 8                                    % atan2(q, w)
    a = atan2(q, w); r2 = w * w + q * q;
    if r2 > 0, dw = -q / r2; dq = w / r2; end
elseif m == 9                                    % atan2(w, q)
    a = atan2(w, q); r2 = w * w + q * q;
    if r2 > 0, dw = q / r2; dq = -w / r2; end
else                                             % atan2(w + q, x + q)
    yy = w + q; xx = x + q;
    a = atan2(yy, xx); r2 = yy * yy + xx * xx;
    if r2 > 0, dw = xx / r2; dx = -yy / r2; dq = dw + dx; end
end
end

function [h, ph, cR, cI] = fwdK(g1, g2, g3, g4, w1, w2, w3, w4, q1, q2, q3, q4, ...
        x1, x2, x3, x4, cRp, cIp, m, k, clip, gateEps, imagMode, carry, signMode, ...
        hEps, phMemEps, phMode, phOutEps, magFloor, xoff)
EPS = 1e-9;
if xoff > 0                                      % cstm_arch2 carries |c| and atan2(Im, Re + xoff)
    rp = sqrt(cRp * cRp + cIp * cIp); fp = atan2(cIp, cRp + xoff);
    cRp = rp * cos(fp); cIp = rp * sin(fp);
end
gf = min(max(g1, -clip), clip); gi = min(max(g2, -clip), clip);
gc = min(max(g3, -clip), clip); go = min(max(g4, -clip), clip);
p1 = min(max(w1 + q1, -clip), clip); p2 = min(max(w2 + q2, -clip), clip);
p3 = min(max(w3 + q3, -clip), clip); p4 = min(max(w4 + q4, -clip), clip);
f = 1 / (1 + exp(-gf)); ig = 1 / (1 + exp(-gi)); o = 1 / (1 + exp(-go)); gR = tanh(gc);
if gateEps
    f = max(f, EPS); ig = max(ig, EPS); o = max(o, EPS); gR = max(gR, EPS);
end
A1 = amapK(m, k, p1, w1, q1, x1); A2 = amapK(m, k, p2, w2, q2, x2);
A3 = amapK(m, k, p3, w3, q3, x3); A4 = amapK(m, k, p4, w4, q4, x4);
if imagMode == 1
    candR = gR; candI = A3;
elseif imagMode == 2
    candR = gR; candI = tanh(p3);
else
    candR = gR * cos(A3); candI = gR * sin(A3);
end
c1 = cos(A1); s1 = sin(A1); c2 = cos(A2); s2 = sin(A2);
if carry
    mR = f * (cRp * c1 - cIp * s1); mI = f * (cRp * s1 + cIp * c1);
else
    mR = 0 * cRp; mI = mR;
end
cR = mR + ig * (candR * c2 - candI * s2);
cI = mI + ig * (candR * s2 + candI * c2);
mag = sqrt(cR * cR + cI * cI);
if magFloor > 0, mag = max(mag, magFloor); end
if signMode == 0
    sg = 1 + 0 * cR;
elseif signMode == 1
    sg = sign(cR + cI);
else
    sg = sign(cR * cI);
end
h = o * tanh(sg * mag);
if hEps, h = max(h, EPS); end
phm = atan2(cI, cR + xoff);
if phMemEps, phm = max(phm, EPS); end
if phMode == 1
    ph = phm;
elseif phMode == 2
    ph = A4 * phm;
elseif phMode == 3
    ph = phm + A4;
else
    ph = phm + 1 / (1 + exp(-p4));
end
if phOutEps, ph = max(ph, EPS); end
end

function [dG1, dG2, dG3, dG4, dW1, dW2, dW3, dW4, dQ1, dQ2, dQ3, dQ4, dX1, dX2, dX3, dX4, dcRp, dcIp] = ...
        bwdK(g1, g2, g3, g4, w1, w2, w3, w4, q1, q2, q3, q4, x1, x2, x3, x4, cRp, cIp, ...
        dh, dph, dcR, dcI, m, k, clip, gateEps, imagMode, carry, signMode, ...
        hEps, phMemEps, phMode, phOutEps, magFloor, xoff)
EPS = 1e-9;
% ---- recompute the step (same as fwdK)
cRraw = cRp; cIraw = cIp;
rp = 0 * cRp; fp = rp;
if xoff > 0                                      % cstm_arch2 carries |c| and atan2(Im, Re + xoff)
    rp = sqrt(cRp * cRp + cIp * cIp); fp = atan2(cIp, cRp + xoff);
    cRp = rp * cos(fp); cIp = rp * sin(fp);
end
gf = min(max(g1, -clip), clip); gi = min(max(g2, -clip), clip);
gc = min(max(g3, -clip), clip); go = min(max(g4, -clip), clip);
p1 = min(max(w1 + q1, -clip), clip); p2 = min(max(w2 + q2, -clip), clip);
p3 = min(max(w3 + q3, -clip), clip); p4 = min(max(w4 + q4, -clip), clip);
sf = 1 / (1 + exp(-gf)); si = 1 / (1 + exp(-gi)); so = 1 / (1 + exp(-go)); tg = tanh(gc);
f = sf; ig = si; o = so; gR = tg;
if gateEps
    f = max(f, EPS); ig = max(ig, EPS); o = max(o, EPS); gR = max(gR, EPS);
end
[A1, dp1, dw1, dq1, dx1] = amapK(m, k, p1, w1, q1, x1);
[A2, dp2, dw2, dq2, dx2] = amapK(m, k, p2, w2, q2, x2);
[A3, dp3, dw3, dq3, dx3] = amapK(m, k, p3, w3, q3, x3);
[A4, dp4, dw4, dq4, dx4] = amapK(m, k, p4, w4, q4, x4);
tp3 = tanh(p3);
if imagMode == 1
    candR = gR; candI = A3;
elseif imagMode == 2
    candR = gR; candI = tp3;
else
    candR = gR * cos(A3); candI = gR * sin(A3);
end
c1 = cos(A1); s1 = sin(A1); c2 = cos(A2); s2 = sin(A2);
if carry
    mR = f * (cRp * c1 - cIp * s1); mI = f * (cRp * s1 + cIp * c1);
else
    mR = 0 * cRp; mI = mR;
end
uR = candR * c2 - candI * s2; uI = candR * s2 + candI * c2;
cR = mR + ig * uR; cI = mI + ig * uI;
rmag = sqrt(cR * cR + cI * cI);
mag = rmag;
if magFloor > 0, mag = max(mag, magFloor); end
if signMode == 0
    sg = 1 + 0 * cR;
elseif signMode == 1
    sg = sign(cR + cI);
else
    sg = sign(cR * cI);
end
tm = tanh(sg * mag);
hpre = o * tm;
xr = cR + xoff;
phraw = atan2(cI, xr);
phm = phraw;
if phMemEps, phm = max(phm, EPS); end
so4 = 1 / (1 + exp(-p4));
if phMode == 1
    ppre = phm;
elseif phMode == 2
    ppre = A4 * phm;
elseif phMode == 3
    ppre = phm + A4;
else
    ppre = phm + so4;
end
% ---- backward
if hEps && hpre <= EPS, dh = 0 * dh; end
dout = dh * tm;
dmag = dh * o * (1 - tm * tm) * sg;
if magFloor > 0 && rmag < magFloor, dmag = 0 * dmag; end
dpp = dph;
if phOutEps && ppre <= EPS, dpp = 0 * dpp; end
dA1 = 0 * dh; dA2 = dA1; dA3 = dA1; dA4 = dA1; dP3 = dA1; dP4 = dA1;
if phMode == 1
    dphm = dpp;
elseif phMode == 2
    dphm = dpp * A4; dA4 = dpp * phm;
elseif phMode == 3
    dphm = dpp; dA4 = dpp;
else
    dphm = dpp; dP4 = dpp * so4 * (1 - so4);
end
if phMemEps && phraw <= EPS, dphm = 0 * dphm; end
if rmag > 0
    dcR = dcR + dmag * cR / rmag; dcI = dcI + dmag * cI / rmag;
end
den = xr * xr + cI * cI;
if den > 0
    dcR = dcR - dphm * cI / den; dcI = dcI + dphm * xr / den;
end
dig = dcR * uR + dcI * uI;
dA2 = dA2 + ig * (dcI * uR - dcR * uI);
dcandR = ig * (dcR * c2 + dcI * s2); dcandI = ig * (dcI * c2 - dcR * s2);
if carry
    df = dcR * (cRp * c1 - cIp * s1) + dcI * (cRp * s1 + cIp * c1);
    dA1 = dA1 + dcI * mR - dcR * mI;
    dcRp = f * (dcR * c1 + dcI * s1); dcIp = f * (dcI * c1 - dcR * s1);
    if xoff > 0                                  % back through the polar carry
        drp = dcRp * cos(fp) + dcIp * sin(fp);
        dfp = rp * (dcIp * cos(fp) - dcRp * sin(fp));
        xr0 = cRraw + xoff; den0 = xr0 * xr0 + cIraw * cIraw;
        dcRp = 0 * dcRp; dcIp = dcRp;
        if rp > 0, dcRp = drp * cRraw / rp; dcIp = drp * cIraw / rp; end
        if den0 > 0, dcRp = dcRp - dfp * cIraw / den0; dcIp = dcIp + dfp * xr0 / den0; end
    end
else
    df = 0 * dh; dcRp = df; dcIp = df;
end
if imagMode == 1
    dgR = dcandR; dA3 = dA3 + dcandI;
elseif imagMode == 2
    dgR = dcandR; dP3 = dP3 + dcandI * (1 - tp3 * tp3);
else
    ca = cos(A3); sa = sin(A3);
    dgR = dcandR * ca + dcandI * sa;
    dA3 = dA3 + gR * (dcandI * ca - dcandR * sa);
end
if gateEps
    if sf <= EPS, df = 0 * df; end
    if si <= EPS, dig = 0 * dig; end
    if so <= EPS, dout = 0 * dout; end
    if tg <= EPS, dgR = 0 * dgR; end
end
dG1 = df * sf * (1 - sf);   if abs(g1) >= clip, dG1 = 0 * dG1; end
dG2 = dig * si * (1 - si);  if abs(g2) >= clip, dG2 = 0 * dG2; end
dG3 = dgR * (1 - tg * tg);  if abs(g3) >= clip, dG3 = 0 * dG3; end
dG4 = dout * so * (1 - so); if abs(g4) >= clip, dG4 = 0 * dG4; end
e1 = dA1 * dp1;       if abs(w1 + q1) >= clip, e1 = 0 * e1; end
e2 = dA2 * dp2;       if abs(w2 + q2) >= clip, e2 = 0 * e2; end
e3 = dP3 + dA3 * dp3; if abs(w3 + q3) >= clip, e3 = 0 * e3; end
e4 = dP4 + dA4 * dp4; if abs(w4 + q4) >= clip, e4 = 0 * e4; end
dW1 = e1 + dA1 * dw1; dQ1 = e1 + dA1 * dq1; dX1 = dA1 * dx1;
dW2 = e2 + dA2 * dw2; dQ2 = e2 + dA2 * dq2; dX2 = dA2 * dx2;
dW3 = e3 + dA3 * dw3; dQ3 = e3 + dA3 * dq3; dX3 = dA3 * dx3;
dW4 = e4 + dA4 * dw4; dQ4 = e4 + dA4 * dq4; dX4 = dA4 * dx4;
end
