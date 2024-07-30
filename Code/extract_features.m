function features = extract_features(data)
    % Extract MFCC features from the audio data
    % data - cell array of audio data
    
    numCoeffs = 13;
    features = cell(length(data), 1);
    
    for i = 1:length(data)
        [coeffs, delta, deltaDelta] = mfcc(data{i}, 16000, 'NumCoeffs', numCoeffs);
        features{i} = [coeffs; delta; deltaDelta]';  % Use coefficients and their derivatives
    end
end