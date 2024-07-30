function model = train_acoustic_model(features, labels)
    % Train a simple GMM-HMM acoustic model
    % features - cell array of feature vectors
    % labels - cell array of labels
    
    uniqueLabels = unique(labels);
    numClasses = length(uniqueLabels);
    model = cell(numClasses, 1);
    
    for i = 1:numClasses
        classFeatures = features(strcmp(labels, uniqueLabels{i}));
        model{i} = trainGMM(classFeatures, 5, 2);  % 5 states, 2 mixtures per state
    end
end
