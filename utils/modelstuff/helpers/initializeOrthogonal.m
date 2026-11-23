function W = initializeOrthogonal(sz)
    [Q, ~] = qr(randn(sz));
    W = Q(:, 1:sz(2));
end