function [data, labels, speaker_info, mic_type] = load_speech_data(rootDir)
    % Load speech data from the given root directory
    % rootDir - root directory containing 'F_Con' and (eventually) 'F_Dys' subdirectories
    
    data = {};
    labels = {};
    speaker_info = {};
    mic_type = {};
    
    % Load control (non-dysarthric) data
    conDir = fullfile(rootDir, 'F_Con');
    [conData, conLabels, conSpeakers, conMics] = load_directory_data(conDir, 'control');
    data = [data; conData];
    labels = [labels; conLabels];
    speaker_info = [speaker_info; conSpeakers];
    mic_type = [mic_type; conMics];
    
    % Load dysarthric data (when available)
    dysDir = fullfile(rootDir, 'F_Dys');
    if exist(dysDir, 'dir')
        [dysData, dysLabels, dysSpeakers, dysMics] = load_directory_data(dysDir, 'dysarthric');
        data = [data; dysData];
        labels = [labels; dysLabels];
        speaker_info = [speaker_info; dysSpeakers];
        mic_type = [mic_type; dysMics];
    else
        warning('Dysarthric data directory not found. Only using control data.');
    end
end

function [data, labels, speakers, mics] = load_directory_data(directory, label)
    data = {};
    labels = {};
    speakers = {};
    mics = {};
    
    subdirs = dir(fullfile(directory, 'wav_*'));
    for i = 1:length(subdirs)
        if subdirs(i).isdir
            subdir_name = subdirs(i).name;
            subdir_path = fullfile(directory, subdir_name);
            
            % Extract speaker and session info
            parts = strsplit(subdir_name, '_');
            current_mic = parts{2};
            current_speaker = parts{3}(1:4);  % e.g., 'FC01'
            current_session = parts{3}(5:end);  % e.g., 'S01'
            
            wav_files = dir(fullfile(subdir_path, '*.wav'));
            for j = 1:length(wav_files)
                file_path = fullfile(subdir_path, wav_files(j).name);
                [audio, fs] = audioread(file_path);
                data{end+1} = audio;
                labels{end+1} = label;
                speakers{end+1} = strcat(current_speaker, '_', current_session);
                mics{end+1} = current_mic;
            end
        end
    end
end