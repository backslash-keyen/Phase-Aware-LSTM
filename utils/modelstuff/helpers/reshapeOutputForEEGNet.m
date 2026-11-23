function Y = reshapeOutputForEEGNet(X)
    % Remove extraneous dimensions using stripdims
    X = stripdims(X);

    % Get the size of X
    sz = size(X);
if ndims(X) < 3

X = reshape(X,[sz(1),1,1,1]);
end
    % Check if X has 3 dimensions (e.g., 61x1x100) and add singleton dimensions if necessary
    if ndims(X) == 3
        % Reshape X to ensure it has 4 dimensions: [H, W, C, B]
        % Using `[]` allows MATLAB to infer the correct size for that dimension
        X = reshape(X, [sz(1), sz(2), 1, sz(3)]);
    end

    % Verify that X now has four dimensions, then permute to (61x100x1x1) or (H x W x C x B)
    if ndims(X) == 4
        X = permute(X, [1 4 3 2]); % Resulting in HxWxCxB (e.g., 61x100x1x1)
    else
        % error('reshapeOutputForEEGNet: Invalid dimensions after reshaping.');
    end

    % Reformat dlarray to match EEGNet expected format: (H, W, C, B)
    Y = dlarray(X, 'SSCB'); % S: Spatial, C: Channel, B: Batch
end
