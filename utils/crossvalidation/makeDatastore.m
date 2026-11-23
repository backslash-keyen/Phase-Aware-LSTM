function dataFeeder = makeDatastore(dataFeeder)
%MAKEDATASTORE
% Converts EEG data arrays into MATLAB datastores for training, validation, and test sets.
% Ensures compatibility with trainNetwork by wrapping both features and labels into datastores.
%
% Input:
%   dataFeeder - Struct with the following fields:
%       xtrain, xval, xtest : [channels x time x N] input arrays
%       ytrain, yval, ytest : [1 x N] label arrays (numeric or categorical)
%
% Output:
%   dataFeeder - Same struct with three additional fields:
%       dsTrain, dsVal, dsTest : Combined and shuffled datastores

%--------------------------------------------
% Set batch size for training
% (Optional: make this configurable later)
%--------------------------------------------
batchSize = 1;

%--------------------------------------------
% Create feature datastores
% Iterate along the 3rd dimension (trials)
%--------------------------------------------
x1 = arrayDatastore(dataFeeder.xtrain, 'IterationDimension', 3, 'ReadSize', batchSize);
x2 = arrayDatastore(dataFeeder.xval,   'IterationDimension', 3, 'ReadSize', batchSize);
x3 = arrayDatastore(dataFeeder.xtest,  'IterationDimension', 3, 'ReadSize', batchSize);

%--------------------------------------------
% Ensure labels are categorical and share the same category set
% This allows compatibility with classification workflows
%--------------------------------------------
if ~iscategorical(dataFeeder.ytrain)
    % Get union of all label values to ensure consistent categories across splits    
    dataFeeder.ytrain = categorical(dataFeeder.ytrain, unique([dataFeeder.ytrain]));
    dataFeeder.yval   = categorical(dataFeeder.yval,   unique([dataFeeder.yval]));
    dataFeeder.ytest  = categorical(dataFeeder.ytest,  unique([dataFeeder.ytest]));
end

%--------------------------------------------
% Create label datastores
% Labels are 1D vectors, so iterate along dimension 1
%--------------------------------------------
y1 = arrayDatastore(dataFeeder.ytrain, 'IterationDimension', 1);
y2 = arrayDatastore(dataFeeder.yval,   'IterationDimension', 1);
y3 = arrayDatastore(dataFeeder.ytest,  'IterationDimension', 1);

%--------------------------------------------
% Combine feature and label datastores into paired datastores
%--------------------------------------------
ds1 = combine(x1, y1);
ds2 = combine(x2, y2);
ds3 = combine(x3, y3);

%--------------------------------------------
% Shuffle all three datastores (reproducibly)
%--------------------------------------------
rng(1);  % Fixed seed for consistent shuffling across runs
dataFeeder.dsTrain = shuffle(ds1);
dataFeeder.dsVal   = shuffle(ds2);
dataFeeder.dsTest  = shuffle(ds3);

end
