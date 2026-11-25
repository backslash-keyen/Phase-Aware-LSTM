
function U = blockOrthogonal(nBlocks, H, gain)
% Returns [nBlocks*H, H] with each HxH block orthogonal, scaled by gain.
U = zeros(nBlocks*H, H, 'like', single(1));
for b = 1:nBlocks
    % QR on a random gaussian matrix
    A = randn(H, H, 'like', single(1));
    [Q,~] = qr(A,0);
    % fix sign (optional)
    d = diag(sign(diag(Q))); if any(diag(d)==0), d = eye(H,'like',Q); end
    Q = Q * d;
    U((b-1)*H + (1:H), :) = gain * Q;
end
end
