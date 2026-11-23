function W = initializeGlorot(sz)
    fanIn = sz(2);
    fanOut = sz(1);
    limit = sqrt(6 / (fanIn + fanOut));
    W = (rand(sz) * 2 - 1) * limit;
end
