function [accuracy, confusionMat] = compute_accuracy(trueLabels, predictedLabels)
    % Compute accuracy and confusion matrix
    % trueLabels - cell array of true labels
    % predictedLabels - cell array of predicted labels
    
    accuracy = mean(strcmp(trueLabels, predictedLabels));
    
    uniqueLabels = unique([trueLabels; predictedLabels]);
    numLabels = length(uniqueLabels);
    confusionMat = zeros(numLabels, numLabels);
    
    for i = 1:numLabels
        for j = 1:numLabels
            confusionMat(i, j) = sum(strcmp(trueLabels, uniqueLabels{i}) & strcmp(predictedLabels, uniqueLabels{j}));
        end
    end
end
