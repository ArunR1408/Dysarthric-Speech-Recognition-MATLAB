% CNN for Dysarthria Detection

% Load required toolboxes
% Make sure you have Signal Processing Toolbox, Audio Toolbox, and Deep Learning Toolbox installed

% Load data
% Replace this with actual data loading. 'data' should be a table with columns:
% 'filepath' (string), 'condition' (categorical)
load('dysarthria_data.mat', 'data');

% Extract MFCC features
numCoeff = 128;
numRows = 16;
numCols = 8;
X = zeros(height(data), numRows, numCols);
y = zeros(height(data), 1);

for i = 1:height(data)
    [x, fs] = audioread(data.filepath{i});
    mfccs = mfcc(x, fs, 'NumCoeffs', numCoeff);
    mfccs = reshape(mfccs(1:numRows*numCols), [numRows, numCols]);
    X(i, :, :) = mfccs;
    
    if strcmp(data.condition{i}, 'with_dysarthria')
        y(i) = 1;
    else
        y(i) = 0;
    end
end

% Prepare data for deep learning
X = reshape(X, [size(X, 1), size(X, 2), size(X, 3), 1]);

% Split data
cv = cvpartition(size(X, 1), 'HoldOut', 0.2);
XTrain = X(training(cv), :, :, :);
yTrain = y(training(cv));
XTest = X(test(cv), :, :, :);
yTest = y(test(cv));

% Define CNN architecture
layers = [
    imageInputLayer([numRows numCols 1])
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    fullyConnectedLayer(128)
    dropoutLayer(0.5)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
];

% Training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 32, ...
    'ValidationData', {XTest, categorical(yTest)}, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
net = trainNetwork(XTrain, categorical(yTrain), layers, options);

% Evaluate the trained network
predictedLabels = classify(net, XTest);
accuracy = mean(predictedLabels == categorical(yTest));
fprintf('CNN Recognition accuracy: %.2f%%\n', accuracy * 100);

% Confusion matrix
confusionchart(categorical(yTest), predictedLabels);
title('Confusion Matrix - CNN');

% Additional analysis
% Per-class accuracy
[confMat, order] = confusionmat(categorical(yTest), predictedLabels);
perClassAccuracy = diag(confMat) ./ sum(confMat, 2);
figure;
bar(perClassAccuracy);
set(gca, 'XTickLabel', order);
xlabel('Classes');
ylabel('Accuracy');
title('Per-class Accuracy - CNN');

% Error analysis
errors = find(predictedLabels ~= categorical(yTest));
fprintf('\nError Analysis (CNN):\n');
for i = 1:min(10, length(errors))
    fprintf('Sample %d: True label = %s, Predicted label = %s\n', ...
        errors(i), string(categorical(yTest(errors(i)))), string(predictedLabels(errors(i))));
end

% Learning curve
trainSizes = round(linspace(100, size(XTrain, 1), 10));
accuracies = zeros(length(trainSizes), 1);

for i = 1:length(trainSizes)
    subsetIdx = randperm(size(XTrain, 1), trainSizes(i));
    subsetXTrain = XTrain(subsetIdx, :, :, :);
    subsetYTrain = yTrain(subsetIdx);
    
    subsetNet = trainNetwork(subsetXTrain, categorical(subsetYTrain), layers, options);
    subsetPredictions = classify(subsetNet, XTest);
    accuracies(i) = mean(categorical(yTest) == subsetPredictions);
end

figure;
plot(trainSizes, accuracies * 100, '-o');
xlabel('Training Set Size');
ylabel('Accuracy (%)');
title('Learning Curve - CNN');
