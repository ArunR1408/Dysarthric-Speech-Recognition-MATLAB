% Main script for dysarthric speech recognition

clear;
clc;

% Define paths (update these with your actual paths)
dataDir = 'path/to/your/data/';
modelDir = 'path/to/save/models/';

% Load dysarthric speech data
[trainData, trainLabels] = load_dysarthric_data(dataDir, 'train');
[testData, testLabels] = load_dysarthric_data(dataDir, 'test');

% Extract features
trainFeatures = extract_features(trainData);
testFeatures = extract_features(testData);

% Train acoustic model
acousticModel = train_acoustic_model(trainFeatures, trainLabels);

% Save the trained model
save(fullfile(modelDir, 'acoustic_model.mat'), 'acousticModel');

% Test the model
uniqueLabels = unique(trainLabels);
predictedLabels = test_acoustic_model(acousticModel, testFeatures, uniqueLabels);

% Evaluate performance
[accuracy, confusionMat] = compute_accuracy(testLabels, predictedLabels);
fprintf('Recognition accuracy: %.2f%%\n', accuracy * 100);

% Plot confusion matrix
figure;
confusionchart(confusionMat, uniqueLabels);
title('Confusion Matrix');

% Additional analysis
% 1. Per-class accuracy
perClassAccuracy = diag(confusionMat) ./ sum(confusionMat, 2);
figure;
bar(perClassAccuracy);
set(gca, 'XTickLabel', uniqueLabels);
xlabel('Classes');
ylabel('Accuracy');
title('Per-class Accuracy');

% 2. Error analysis
errors = find(~strcmp(testLabels, predictedLabels));
fprintf('\nError Analysis:\n');
for i = 1:min(10, length(errors))
    fprintf('Sample %d: True label = %s, Predicted label = %s\n', ...
        errors(i), testLabels{errors(i)}, predictedLabels{errors(i)});
end

% 3. Learning curve
trainSizes = round(linspace(100, length(trainFeatures), 10));
accuracies = zeros(length(trainSizes), 1);

for i = 1:length(trainSizes)
    subsetIdx = randperm(length(trainFeatures), trainSizes(i));
    subsetFeatures = trainFeatures(subsetIdx);
    subsetLabels = trainLabels(subsetIdx);
    
    subsetModel = train_acoustic_model(subsetFeatures, subsetLabels);
    subsetPredictions = test_acoustic_model(subsetModel, testFeatures, uniqueLabels);
    accuracies(i) = mean(strcmp(testLabels, subsetPredictions));
end

figure;
plot(trainSizes, accuracies * 100, '-o');
xlabel('Training Set Size');
ylabel('Accuracy (%)');
title('Learning Curve');
