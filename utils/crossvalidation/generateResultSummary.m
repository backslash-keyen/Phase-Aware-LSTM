function [meanTrainAcc, meanValAcc, meanTestAcc, meanTrainTime] = generateResultSummary(config,evals) 
% Calculate average metrics across folds
    accTrain = zeros(1, config.ncval);
    accVal = zeros(1, config.ncval);
    accTest = zeros(1, config.ncval);
    trainTimes = zeros(1, config.ncval);

    for i = 1:config.ncval
        accTrain(i) = evals{i}.accuracyTrain;
        accVal(i) = evals{i}.accuracyVal;
        accTest(i) = evals{i}.accuracyTest;
        trainTimes(i) = evals{i}.trainingTime;
    end

    meanTrainAcc = mean(accTrain);
    meanValAcc = mean(accVal);
    meanTestAcc = mean(accTest);
    meanTrainTime = mean(trainTimes);
end

