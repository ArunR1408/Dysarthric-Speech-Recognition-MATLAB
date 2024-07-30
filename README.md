# Dysarthric Speech Recognition

## Project Overview
This project aims to classify dysarthric and non-dysarthric speech using deep learning techniques. The dataset consists of audio samples from dysarthric and non-dysarthric individuals, with the goal of developing a model that can accurately distinguish between the two classes.

## Features
- Speech waveplot visualization
- Spectrogram analysis
- Zero-crossing rate (ZCR) calculation and visualization
- Feature extraction using MFCC
- Deep learning model implementation using CNN

## Data Information
The dataset contains 2000 audio samples divided into four categories:
- Dysarthric females: 500 samples
- Dysarthric males: 500 samples
- Non-dysarthric females: 500 samples
- Non-dysarthric males: 500 samples

### Data CSV Information
The `data.csv` file contains the following columns:
- `filename`: Path to the audio file
- `is_dysarthria`: Indicates if the sample is from a dysarthric individual (dysarthria or non_dysarthria)
- `gender`: Gender of the speaker (male or female)

## Usage
1. Clone the repository.
2. Ensure the dataset is located in the specified directory.
3. Run `main.m` to load data, visualize waveplots, spectrograms, and ZCRs, and train the CNN model.

## Work in Progress
This project is a work in progress, and additional features and improvements are being actively developed.

## License
This project is licensed under the MIT License.
