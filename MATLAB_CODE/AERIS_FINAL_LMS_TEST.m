%% =========================================================
% AERIS - FINAL OPTIMIZED LMS TEST
% SIH Adaptive Noise Cancellation
% =========================================================

clear;
clc;
close all;

fprintf('\n');
fprintf('=================================================\n');
fprintf('        AERIS FINAL OPTIMIZED LMS TEST\n');
fprintf('=================================================\n\n');

%% PATHS

projectFolder = ...
    'C:\Users\nuppe\Documents\MATLAB\AERIS_SIH';

outputFolder = fullfile(projectFolder,'OUTPUT');

cleanFile = fullfile(outputFolder,'clean_test.wav');
noiseFile = fullfile(outputFolder,'scaled_noise.wav');

%% OPTIMIZED LMS PARAMETERS

filterOrder = 16;
mu = 0.010;

snrLevels = [5 0 -5 -10 -15 -20];

%% LOAD AUDIO

[clean,Fs] = audioread(cleanFile);
[noise,FsNoise] = audioread(noiseFile);

%% MONO

if size(clean,2) > 1
    clean = mean(clean,2);
end

if size(noise,2) > 1
    noise = mean(noise,2);
end

%% MATCH LENGTH

N = min(length(clean),length(noise));

clean = clean(1:N);
noise = noise(1:N);

clean = clean(:);
noise = noise(:);

fprintf('Sampling rate : %d Hz\n',Fs);
fprintf('Samples       : %d\n',N);
fprintf('Duration      : %.2f seconds\n\n',N/Fs);

fprintf('Optimized LMS filter order : %d\n',filterOrder);
fprintf('Optimized step size        : %.4f\n\n',mu);

%% RESULT ARRAYS

inputSNR = zeros(length(snrLevels),1);
outputSNR = zeros(length(snrLevels),1);
improvement = zeros(length(snrLevels),1);
outputMSE = zeros(length(snrLevels),1);

%% =========================================================
% PROCESS EACH SNR
% =========================================================

for k = 1:length(snrLevels)

    targetSNR = snrLevels(k);

    fprintf('---------------------------------------------\n');
    fprintf('Testing SNR = %d dB\n',targetSNR);
    fprintf('---------------------------------------------\n');

    %% SCALE NOISE

    cleanPower = mean(clean.^2);
    noisePower = mean(noise.^2);

    requiredNoisePower = ...
        cleanPower / (10^(targetSNR/10));

    scaleFactor = ...
        sqrt(requiredNoisePower/noisePower);

    scaledNoise = noise * scaleFactor;

    %% CREATE NOISY SIGNAL

    noisy = clean + scaledNoise;

    %% INPUT SNR

    inputNoise = noisy - clean;

    inputSNR(k) = ...
        10*log10( ...
        sum(clean.^2)/sum(inputNoise.^2));

    %% LMS

    w = zeros(filterOrder,1);
    xBuffer = zeros(filterOrder,1);

    enhanced = zeros(N,1);

    for n = 1:N

        xBuffer = ...
            [scaledNoise(n); ...
             xBuffer(1:end-1)];

        estimatedNoise = w' * xBuffer;

        enhanced(n) = ...
            noisy(n) - estimatedNoise;

        errorSignal = enhanced(n);

        w = ...
            w + mu*errorSignal*xBuffer;

    end

    %% OUTPUT SNR

    residualNoise = enhanced - clean;

    outputSNR(k) = ...
        10*log10( ...
        sum(clean.^2)/sum(residualNoise.^2));

    %% IMPROVEMENT

    improvement(k) = ...
        outputSNR(k)-inputSNR(k);

    %% MSE

    outputMSE(k) = ...
        mean(residualNoise.^2);

    %% DISPLAY

    fprintf('Input SNR       : %.2f dB\n',inputSNR(k));
    fprintf('Output SNR      : %.2f dB\n',outputSNR(k));
    fprintf('Improvement     : %.2f dB\n',improvement(k));
    fprintf('Output MSE      : %.8f\n\n',outputMSE(k));

    %% SAVE IMPORTANT AUDIO

    if targetSNR == -10 || targetSNR == -20

        saveAudio = enhanced;

        saveAudio = ...
            saveAudio/(max(abs(saveAudio))+eps);

        filename = fullfile( ...
            outputFolder, ...
            sprintf('AERIS_FINAL_%ddB.wav',targetSNR));

        audiowrite(filename,saveAudio,Fs);

    end

end

%% =========================================================
% FINAL TABLE
% =========================================================

fprintf('\n');
fprintf('=================================================\n');
fprintf('           AERIS FINAL RESULTS\n');
fprintf('=================================================\n\n');

fprintf('%10s %15s %15s %15s\n', ...
    'SNR','Input SNR','Output SNR','Improvement');

fprintf('-------------------------------------------------\n');

for k = 1:length(snrLevels)

    fprintf('%10d %15.2f %15.2f %15.2f\n', ...
        snrLevels(k), ...
        inputSNR(k), ...
        outputSNR(k), ...
        improvement(k));

end

%% =========================================================
% AVERAGE PERFORMANCE
% =========================================================

averageImprovement = mean(improvement);
minimumImprovement = min(improvement);
maximumImprovement = max(improvement);

fprintf('\n');
fprintf('Average improvement : %.2f dB\n',averageImprovement);
fprintf('Minimum improvement : %.2f dB\n',minimumImprovement);
fprintf('Maximum improvement : %.2f dB\n',maximumImprovement);

%% =========================================================
% SAVE CSV
% =========================================================

resultsTable = table( ...
    snrLevels(:), ...
    inputSNR, ...
    outputSNR, ...
    improvement, ...
    outputMSE, ...
    'VariableNames', ...
    {'TargetSNR_dB', ...
     'InputSNR_dB', ...
     'OutputSNR_dB', ...
     'Improvement_dB', ...
     'OutputMSE'});

csvFile = fullfile( ...
    outputFolder, ...
    'AERIS_FINAL_RESULTS.csv');

writetable(resultsTable,csvFile);

%% =========================================================
% GRAPH 1
% =========================================================

figure;

plot(snrLevels,inputSNR,'-o','LineWidth',2);
hold on;
plot(snrLevels,outputSNR,'-o','LineWidth',2);

xlabel('Input SNR (dB)');
ylabel('SNR (dB)');

title('AERIS - Input vs Output SNR');

legend('Input SNR','Output SNR','Location','best');

grid on;

saveas(gcf, ...
    fullfile(outputFolder,...
    'AERIS_FINAL_SNR_COMPARISON.png'));

%% =========================================================
% GRAPH 2
% =========================================================

figure;

bar(snrLevels,improvement);

xlabel('Input SNR (dB)');
ylabel('SNR Improvement (dB)');

title('AERIS - Noise Cancellation Improvement');

grid on;

saveas(gcf, ...
    fullfile(outputFolder,...
    'AERIS_FINAL_IMPROVEMENT.png'));

%% =========================================================
% FINISH
% =========================================================

fprintf('\n');
fprintf('=================================================\n');
fprintf('       AERIS FINAL LMS TEST COMPLETE\n');
fprintf('=================================================\n\n');

fprintf('Optimized filter order : %d\n',filterOrder);
fprintf('Optimized step size    : %.4f\n',mu);

fprintf('\nFiles generated:\n');
fprintf('AERIS_FINAL_RESULTS.csv\n');
fprintf('AERIS_FINAL_SNR_COMPARISON.png\n');
fprintf('AERIS_FINAL_IMPROVEMENT.png\n');
fprintf('AERIS_FINAL_-10dB.wav\n');
fprintf('AERIS_FINAL_-20dB.wav\n');

fprintf('\n=================================================\n');
