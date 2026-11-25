
function W = glorotUniform(sz)
% sz = [rows, cols]
fan_in  = sz(2);
fan_out = sz(1);
limit = sqrt(6/(fan_in + fan_out));
W = (rand(sz,'like',single(1)) * 2 - 1) * limit;
end