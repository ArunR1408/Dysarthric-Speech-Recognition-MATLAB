function predictedLabels = test_acoustic_model(model, features, uniqueLabels)
    % Test the acoustic model
    % model - trained acoustic model
    % features - cell array of feature vectors
    % uniqueLabels - cell array of unique labels
    
    numSamples = length(features);
    predictedLabels = cell(numSamples, 1);
    
    for i = 1:numSamples
        logLikelihoods = zeros(length(model), 1);
        for j = 1:length(model)
            logLikelihoods(j) = sum(log(pdf(model{j}, features{i})));
        end
        [~, idx] = max(logLikelihoods);
        predictedLabels{i} = uniqueLabels{idx};
    end
end
