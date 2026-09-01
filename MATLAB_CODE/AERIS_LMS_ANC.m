%% =========================================================
% AERIS - LMS ADAPTIVE NOISE CANCELLATION
% ==========================================================

clear;
clc;
close all;

fprintf('\n');
fprintf('===============================================\n');
fprintf('       AERIS - LMS ADAPTIVE NOISE CANCELLER\n');
fprintf('===============================================\n\n');

%% ---------------------------------------------------------
% 1. PATHS
% ----------------------------------------------------------

projectFolder = 'C:\Users\nuppe\Documents\MATLAB\AERIS_SIH';

outputFolder = fullfile(projectFolder,'OUTPUT');

cleanFile = fullfile(outputFolder,'clean_test.wav');
noisyFile = fullfile(outputFolder,'noisy_test.wav');
noiseFile = fullfile(outputFolder,'scaled_noise.wav');

%% ---------------------------------------------------------
% 2. LOAD AUDIO
% ----------------------------------------------------------

[clean, Fs1] = audioread(cleanFile);
[noisy, Fs2] = audioread(noisyFile);
[referenceNoise, Fs3] = audioread(noiseFile);

%% ---------------------------------------------------------
% 3. CONVERT TO MONO
% ----------------------------------------------------------

if size(clean,2) > 1
    clean = mean(clean,2);
end

if size(noisy,2) > 1
    noisy = mean(noisy,2);
end

if size(referenceNoise,2) > 1
    referenceNoise = mean(referenceNoise,2);
end

%% ---------------------------------------------------------
% 4. CHECK SAMPLING RATE
% ----------------------------------------------------------

Fs = Fs1;

fprintf('Sampling rate : %d Hz\n',Fs);

%% ---------------------------------------------------------
% 5. MATCH LENGTHS
% ----------------------------------------------------------

N = min([length(clean), length(noisy), length(referenceNoise)]);

clean = clean(1:N);
noisy = noisy(1:N);
referenceNoise = referenceNoise(1:N);

%% ---------------------------------------------------------
% 6. LMS PARAMETERS
% ----------------------------------------------------------

filterOrder = 32;

mu = 0.005;

fprintf('LMS filter order : %d\n',filterOrder);
fprintf('Step size        : %.4f\n\n',mu);

%% ---------------------------------------------------------
% 7. INITIALIZE LMS FILTER
% ----------------------------------------------------------

weights = zeros(filterOrder,1);

estimatedNoise = zeros(N,1);

enhanced = zeros(N,1);

xBuffer = zeros(filterOrder,1);

%% ---------------------------------------------------------
% 8. LMS ADAPTIVE FILTER
% ----------------------------------------------------------

fprintf('Running LMS adaptive noise cancellation...\n');

for n = 1:N

    % Update reference noise buffer

    xBuffer = [referenceNoise(n); xBuffer(1:end-1)];

    % Estimate noise

    estimatedNoise(n) = weights' * xBuffer;

    % Error signal

    enhanced(n) = noisy(n) - estimatedNoise(n);

    % Update filter weights

    weights = weights + mu * enhanced(n) * xBuffer;

end

fprintf('LMS processing complete.\n\n');

%% ---------------------------------------------------------
% 9. NORMALIZE OUTPUT
% ----------------------------------------------------------

maxValue = max(abs(enhanced));

if maxValue > 0.99
    enhanced = enhanced * (0.99/maxValue);
end

%% ---------------------------------------------------------
% 10. CALCULATE INPUT SNR
% ----------------------------------------------------------

inputNoise = noisy - clean;

inputSNR = 10 * log10( ...
    sum(clean.^2) / sum(inputNoise.^2) );

%% ---------------------------------------------------------
% 11. CALCULATE OUTPUT SNR
% ----------------------------------------------------------

outputNoise = enhanced - clean;

outputSNR = 10 * log10( ...
    sum(clean.^2) / sum(outputNoise.^2) );

%% ---------------------------------------------------------
% 12. SNR IMPROVEMENT
% ----------------------------------------------------------

snrImprovement = outputSNR - inputSNR;

%% ---------------------------------------------------------
% 13. DISPLAY RESULTS
% ----------------------------------------------------------

fprintf('===============================================\n');
fprintf('              AERIS RESULTS\n');
fprintf('===============================================\n\n');

fprintf('Input SNR       : %.2f dB\n',inputSNR);
fprintf('Output SNR      : %.2f dB\n',outputSNR);
fprintf('SNR Improvement : %.2f dB\n\n',snrImprovement);

%% ---------------------------------------------------------
% 14. SAVE ENHANCED SPEECH
% ----------------------------------------------------------

enhancedFile = fullfile(outputFolder,'enhanced_speech.wav');

audiowrite(enhancedFile,enhanced,Fs);

fprintf('Enhanced speech saved:\n');
fprintf('%s\n\n',enhancedFile);

%% ---------------------------------------------------------
% 15. SAVE ESTIMATED NOISE
% ----------------------------------------------------------

estimatedNoiseFile = fullfile(outputFolder,'estimated_noise.wav');

audiowrite(estimatedNoiseFile,estimatedNoise,Fs);

fprintf('Estimated noise saved:\n');
fprintf('%s\n\n',estimatedNoiseFile);

%% ---------------------------------------------------------
% 16. TIME AXIS
% ----------------------------------------------------------

t = (0:N-1)/Fs;

%% ---------------------------------------------------------
% 17. CLEAN SPEECH
% ----------------------------------------------------------

figure;

plot(t,clean);

xlabel('Time (seconds)');
ylabel('Amplitude');

title('Original Clean Speech');

grid on;

saveas(gcf, ...
    fullfile(outputFolder,'01_clean_speech.png'));

%% ---------------------------------------------------------
% 18. NOISY SPEECH
% ----------------------------------------------------------

figure;

plot(t,noisy);

xlabel('Time (seconds)');
ylabel('Amplitude');

title(sprintf('Noisy Speech - Input SNR %.2f dB',inputSNR));

grid on;

saveas(gcf, ...
    fullfile(outputFolder,'02_noisy_speech.png'));

%% ---------------------------------------------------------
% 19. ESTIMATED NOISE
% ----------------------------------------------------------

figure;

plot(t,estimatedNoise);

xlabel('Time (seconds)');
ylabel('Amplitude');

title('LMS Estimated Noise');

grid on;

saveas(gcf, ...
    fullfile(outputFolder,'03_estimated_noise.png'));

%% ---------------------------------------------------------
% 20. ENHANCED SPEECH
% ----------------------------------------------------------

figure;

plot(t,enhanced);

xlabel('Time (seconds)');
ylabel('Amplitude');

title(sprintf('Enhanced Speech - Output SNR %.2f dB',outputSNR));

grid on;

saveas(gcf, ...
    fullfile(outputFolder,'04_enhanced_speech.png'));

%% ---------------------------------------------------------
% 21. COMPARISON
% ----------------------------------------------------------

figure;

plot(t,noisy);
hold on;
plot(t,enhanced);

xlabel('Time (seconds)');
ylabel('Amplitude');

title('AERIS ANC: Noisy vs Enhanced');

legend('Noisy Speech','Enhanced Speech');

grid on;

saveas(gcf, ...
    fullfile(outputFolder,'05_noisy_vs_enhanced.png'));

%% ---------------------------------------------------------
% 22. FINAL
% ----------------------------------------------------------

fprintf('===============================================\n');
fprintf('        AERIS LMS ANC TEST COMPLETE\n');
fprintf('===============================================\n\n');

fprintf('Input SNR       : %.2f dB\n',inputSNR);
fprintf('Output SNR      : %.2f dB\n',outputSNR);
fprintf('Improvement     : %.2f dB\n\n',snrImprovement);

fprintf('Next stage: NLMS / RLS adaptive filtering\n');

fprintf('===============================================\n');