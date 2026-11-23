function perfold_printevaluation(i, evaluation, model)
%PRINTFOLDEVALUATION
% Prints training, validation, and test accuracy, along with training time.
%
% Inputs:
%   i           - Fold index (1 to k)
%   evaluation  - Struct with fields:
%                   .accuracyTrain, .accuracyVal, .accuracyTest
%   model       - Trained model struct with field:
%                   .trainingTime (in seconds)

%-------------------------------
% Display formatted evaluation
%-------------------------------
fprintf('Fold %d completed:\n', i);
fprintf('   ▸ Train Accuracy  : %.2f%%\n', evaluation.accuracyTrain * 100);
fprintf('   ▸ Val Accuracy    : %.2f%%\n', evaluation.accuracyVal * 100);
fprintf('   ▸ Test Accuracy   : %.2f%%\n', evaluation.accuracyTest * 100);

%-------------------------------
% Display training duration
%-------------------------------
fprintf('   ⏱  Training Time   : %.2f seconds\n', model.trainingTime);


end
