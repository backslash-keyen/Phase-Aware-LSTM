function previousState = ensureReproducibility(i)
%ENSUREREPRODUCIBILITY
% Ensures deterministic behavior by setting the random seed for CPU and GPU.
% Used to maintain reproducible results across different runs or folds.
%
% Input:
%   i - Seed value (typically fold index or any integer)
%
% Output:
%   previousState - Placeholder for any future state-saving (currently unused)

% Set CPU random seed
rng(i);

% If GPU is available, set GPU random seed as well
if canUseGPU
    gpurng(i);  % Set GPU random number generator seed

    % For reproducibility: deterministic GPU behavior (commented by default)
    % Uncomment below lines only if you explicitly need deterministic GPU ops
    previousState = deep.gpu.deterministicAlgorithms(true);
    randStream = parallel.gpu.RandStream('CombRecursive', 'Seed', i);
    parallel.gpu.RandStream.setGlobalStream(randStream);

    previousState = [];  % Placeholder for future state tracking
else
    previousState = [];  % No GPU → nothing extra to track
end
end
