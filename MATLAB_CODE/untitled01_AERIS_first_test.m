%% =========================================================
%  AERIS - ADAPTIVE NOISE CANCELLATION
%  FIRST DATASET TEST
%  No MATLAB audio playback - saves files instead
% ==========================================================

clear;
clc;
close all;

fprintf('\n');
fprintf('===============================================\n');
fprintf('       AERIS ANC - DATASET TEST\n');
fprintf('===============================================\n\n');

%% =========================================================
% 1. DATASET PATHS
% ==========================================================

cleanFolder = 'C:\Users\nuppe\Documents\MATLAB\AERIS_SIH\DATASET\CLEAN';
noiseFolder = 'C:\Users\nuppe\Documents\MATLAB\AERIS_SIH\DATASET\NOISE';

outputFolder = 'C:\Users\nuppe\Documents\MATLAB\AERIS_SIH\OUTPUT';

%% =========================================================
% 2. CHECK OUTPUT FOLDER
% ==========================================================

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

fprintf('Clean folder:\n%s\n\n', cleanFolder);
fprintf('Noise folder:\n%s\n\n', noiseFolder);
fprintf('Output folder:\n%s\n\n', outputFolder);

%% =========================================================
% 3. FIND CLEAN SPEECH FILES
% ==========================================================

cleanFiles = dir(fullfile(cleanFolder, '**', '*.flac'));

fprintf('Clean speech files found : %d\n', numel(cleanFiles));

if isempty(cleanFiles)
    error('No FLAC clean speech files found.');
end

%% =========================================================
% 4. FIND NOISE FILES
% ==========================================================

noiseFiles = dir(fullfile(noiseFolder, '**', '*.wav'));

fprintf('Noise files found        : %d\n\n', numel(noiseFiles));

if isempty(noiseFiles)
    error('No WAV noise files found.');
end

%% =========================================================
% 5. SELECT ONE CLEAN SPEECH FILE
% ==========================================================

cleanPath = fullfile(cleanFiles(1).folder, cleanFiles(1).name);

fprintf('Selected clean file:\n%s\n\n', cleanPath);

[clean, FsClean] = audioread(cleanPath);

%% =========================================================
% 6. SELECT ONE NOISE FILE
% ==========================================================

noisePath = fullfile(noiseFiles(1).folder, noiseFiles(1).name);

fprintf('Selected noise file:\n%s\n\n', noisePath);

[noise, FsNoise] = audioread(noisePath);

%% =========================================================
% 7. CONVERT STEREO TO MONO
% ==========================================================

if size(clean,2) > 1
    clean = mean(clean,2);
end

if size(noise,2) > 1
    noise = mean(noise,2);
end

%% =========================================================
% 8. CHECK SAMPLING RATES
% ==========================================================

fprintf('Clean sampling rate : %d Hz\n', FsClean);
fprintf('Noise sampling rate : %d Hz\n\n', FsNoise);

if FsClean ~= FsNoise
    fprintf('Resampling noise to %d Hz...\n', FsClean);

    noise = resample(noise, FsClean, FsNoise);

    FsNoise = FsClean;
end

Fs = FsClean;

%% =========================================================
% 9. MATCH NOISE LENGTH TO CLEAN SPEECH
% ==========================================================

N = length(clean);

if length(noise) < N

    fprintf('Noise is shorter than speech.\n');
    fprintf('Repeating noise to match speech length...\n\n');

    repetitions = ceil(N / length(noise));

    noise = repmat(noise, repetitions, 1);

end

noise = noise(1:N);

clean = clean(:);
noise = noise(:);

%% =========================================================
% 10. REMOVE DC OFFSET
% ==========================================================

clean = clean - mean(clean);
noise = noise - mean(noise);

%% =========================================================
% 11. TARGET SNR
% ==========================================================

targetSNR = -10;

%% =========================================================
% 12. CALCULATE RMS VALUES
% ==========================================================

cleanRMS = rms(clean);
noiseRMS = rms(noise);

fprintf('Clean RMS : %.6f\n', cleanRMS);
fprintf('Noise RMS : %.6f\n\n', noiseRMS);

%% =========================================================
% 13. SCALE NOISE FOR EXACT TARGET SNR
% ==========================================================

desiredNoiseRMS = cleanRMS / (10^(targetSNR/20));

noise = noise * (desiredNoiseRMS / noiseRMS);

%% =========================================================
% 14. CREATE NOISY SPEECH
% ==========================================================

noisy = clean + noise;

%% =========================================================
% 15. CALCULATE ACTUAL SNR
% ==========================================================

actualSNR = 20 * log10(rms(clean) / rms(noise));

fprintf('Target SNR : %.2f dB\n', targetSNR);
fprintf('Actual SNR : %.2f dB\n\n', actualSNR);

%% =========================================================
% 16. PREVENT CLIPPING
% ==========================================================

maxValue = max(abs(noisy));

if maxValue > 0.99

    scaleFactor = 0.99 / maxValue;

    clean = clean * scaleFactor;
    noise = noise * scaleFactor;
    noisy = noisy * scaleFactor;

end

%% =========================================================
% 17. SAVE CLEAN SPEECH
% ==========================================================

cleanOutput = fullfile(outputFolder, 'clean_test.wav');

audiowrite(cleanOutput, clean, Fs);

fprintf('Clean speech saved:\n%s\n\n', cleanOutput);

%% =========================================================
% 18. SAVE NOISY SPEECH
% ==========================================================

noisyOutput = fullfile(outputFolder, 'noisy_test.wav');

audiowrite(noisyOutput, noisy, Fs);

fprintf('Noisy speech saved:\n%s\n\n', noisyOutput);

%% =========================================================
% 19. SAVE NOISE
% ==========================================================

noiseOutput = fullfile(outputFolder, 'scaled_noise.wav');

audiowrite(noiseOutput, noise, Fs);

fprintf('Scaled noise saved:\n%s\n\n', noiseOutput);

%% =========================================================
% 20. PLOT CLEAN SPEECH
% ==========================================================

t = (0:length(clean)-1) / Fs;

figure;

plot(t, clean);

xlabel('Time (seconds)');
ylabel('Amplitude');

title('Clean Speech Signal');

grid on;

saveas(gcf, fullfile(outputFolder, 'clean_speech_plot.png'));

%% =========================================================
% 21. PLOT NOISY SPEECH
% ==========================================================

figure;

plot(t, noisy);

xlabel('Time (seconds)');
ylabel('Amplitude');

title(sprintf('Noisy Speech - SNR = %.2f dB', actualSNR));

grid on;

saveas(gcf, fullfile(outputFolder, 'noisy_speech_plot.png'));

%% =========================================================
% 22. FFT OF CLEAN SIGNAL
% ==========================================================

L = length(clean);

Yclean = fft(clean);

P2clean = abs(Yclean/L);

P1clean = P2clean(1:floor(L/2)+1);

P1clean(2:end-1) = 2*P1clean(2:end-1);

f = Fs*(0:floor(L/2))/L;

figure;

plot(f, P1clean);

xlabel('Frequency (Hz)');
ylabel('Magnitude');

title('Clean Speech Frequency Spectrum');

xlim([0 8000]);

grid on;

saveas(gcf, fullfile(outputFolder, 'clean_fft.png'));

%% =========================================================
% 23. FFT OF NOISY SIGNAL
% ==========================================================

Ynoisy = fft(noisy);

P2noisy = abs(Ynoisy/L);

P1noisy = P2noisy(1:floor(L/2)+1);

P1noisy(2:end-1) = 2*P1noisy(2:end-1);

figure;

plot(f, P1noisy);

xlabel('Frequency (Hz)');
ylabel('Magnitude');

title('Noisy Speech Frequency Spectrum');

xlim([0 8000]);

grid on;

saveas(gcf, fullfile(outputFolder, 'noisy_fft.png'));

%% =========================================================
% 24. FINAL INFORMATION
% ==========================================================

fprintf('\n');
fprintf('===============================================\n');
fprintf('          AERIS TEST COMPLETE\n');
fprintf('===============================================\n\n');

fprintf('Clean files available : %d\n', numel(cleanFiles));
fprintf('Noise files available : %d\n', numel(noiseFiles));
fprintf('Sampling rate         : %d Hz\n', Fs);
fprintf('Target SNR             : %.2f dB\n', targetSNR);
fprintf('Actual SNR             : %.2f dB\n\n', actualSNR);

fprintf('Generated files:\n');
fprintf('1. clean_test.wav\n');
fprintf('2. noisy_test.wav\n');
fprintf('3. scaled_noise.wav\n');
fprintf('4. clean_speech_plot.png\n');
fprintf('5. noisy_speech_plot.png\n');
fprintf('6. clean_fft.png\n');
fprintf('7. noisy_fft.png\n\n');

fprintf('All files are saved inside:\n');
fprintf('%s\n\n', outputFolder);

fprintf('NEXT STEP: Adaptive Noise Cancellation\n');

fprintf('\n===============================================\n');