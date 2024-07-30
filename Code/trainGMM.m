function gmm = trainGMM(features, numStates, numMixtures)
    % Train a Gaussian Mixture Model
    % features - cell array of feature vectors
    % numStates - number of states in the GMM
    % numMixtures - number of mixtures per state
    
    X = cell2mat(features);
    gmm = fitgmdist(X, numStates * numMixtures, 'RegularizationValue', 0.01, 'CovarianceType', 'diagonal');
end
