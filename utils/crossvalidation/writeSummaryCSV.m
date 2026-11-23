function writeSummaryCSV(config,meanTrainAcc, meanValAcc, meanTestAcc, meanTrainTime,timestamp)
% Prepare summary line
    % Define output directory
    outputDir = fullfile(config.projectLoc, 'cvalresults', config.datasetType, config.typeNet);

    % Create path to the summary CSV file
    summaryFileName = fullfile(outputDir, 'projectionModule_baseline.csv');
    
    summaryLine = sprintf('%s,%s,%s,%.4f,%.4f,%.4f,%.2f,%s\n', ...
        config.typeNet, config.domain, config.layerName, ...
        meanTrainAcc, meanValAcc, meanTestAcc, meanTrainTime, timestamp);

    % Write header if file does not exist
    if ~isfile(summaryFileName)
        fid = fopen(summaryFileName, 'w');
        fprintf(fid, 'TypeNet,Domain,LayerName,TrainAcc,ValAcc,TestAcc,TrainTime,Timestamp\n');
        fclose(fid);
    end

    % Append new summary
    fid = fopen(summaryFileName, 'a');
    fprintf(fid, summaryLine);
    fclose(fid);

    fprintf('📝 Summary appended: %s\n', summaryFileName);
end

