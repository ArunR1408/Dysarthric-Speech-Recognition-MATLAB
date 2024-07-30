% MATLAB implementation of dysarthria detection using CNN on MFCC features

clear;
clc;

% Load data
rootDir = 'D:\Amrita\OneDrive - Amrita university\Amrita\S6\Lab\19EAC386 Speech Processing\Project\Code';
[data, labels, speaker_info, mic_type] = load_speech_data(rootDir);

% Extract MFCC features
features = extract_features(data);

% Prepare data for deep learning
numCoeff = size(features{1}, 2);
numFrames = 50;  % Fixed number of frames (you may need to adjust this)

X = zeros(length(features), numFrames, numCoeff);
y = zeros(length(labels), 1);

for i = 1:length(features)
    if size(features{i}, 1) >= numFrames
        X(i, :, :) = features{i}(1:numFrames, :);
    else
        X(i, 1:size(features{i}, 1), :) = features{i};
        X(i, size(features{i}, 1)+1:end, :) = repmat(features{i}(end, :), numFrames - size(features{i}, 1), 1);
    end
    
    if strcmp(labels{i}, 'dysarthric')
        y(i) = 1;
    else
        y(i) = 0;
    end
end

% Reshape X for CNN input
X = reshape(X, [size(X, 1), size(X, 2), size(X, 3), 1]);

% Convert labels to categorical
y = categorical(y);

% Split data ensuring speakers are not mixed between train and test sets
unique_speakers = unique(speaker_info);
cv = cvpartition(length(unique_speakers), 'HoldOut', 0.2);

train_speakers = unique_speakers(cv.training);
test_speakers = unique_speakers(cv.test);

train_idx = ismember(speaker_info, train_speakers);
test_idx = ismember(speaker_info, test_speakers);

XTrain = X(train_idx, :, :, :);
yTrain = y(train_idx);
XTest = X(test_idx, :, :, :);
yTest = y(test_idx);

% The rest of the code (CNN architecture, training, and evaluation) remains the same as in the previous version.

% ...

% After training and evaluating, you can add some analysis based on microphone type:
unique_mics = unique(mic_type);
for i = 1:length(unique_mics)
    mic_idx = strcmp(mic_type(test_idx), unique_mics{i});
    mic_accuracy = mean(predictedLabels(mic_idx) == yTest(mic_idx));
    fprintf('Accuracy for %s: %.2f%%\n', unique_mics{i}, mic_accuracy * 100);
end