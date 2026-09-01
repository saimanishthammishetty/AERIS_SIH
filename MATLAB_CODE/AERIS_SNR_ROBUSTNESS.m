%% =========================================================
% AERIS - LOW SNR ROBUSTNESS TEST
% LMS Adaptive Noise Cancellation
% =========================================================

clear;
clc;
close all;

fprintf('\n');
fprintf('=================================================\n');
fprintf('       AERIS LOW-SNR ROBUSTNESS TEST\n');
fprintf('=================================================\n\n');

%% PATHS

projectFolder = ...
    'C:\Users\nuppe\Documents\MATLAB\AERIS_SIH';

outputFolder = fullfile(projectFolder,'OUTPUT');

cleanFile = fullfile(outputFolder,'clean_test.wav');
noiseFile = fullfile(outputFolder,'scaled_noise.wav');

%% LOAD AUDIO

[clean,Fs] = audioread(cleanFile);
[noise,FsNoise] = audioread(noiseFile);

%% CONVERT TO MONO

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

%% SNR LEVELS TO TEST

snrLevels = [5 0 -5 -10 -15 -20];

numTests = length(snrLevels);

inputSNR = zeros(numTests,1);
outputSNR = zeros(numTests,1);
improvement = zeros(numTests,1);
mseOutput = zeros(numTests,1);

%% LMS PARAMETERS

filterOrder = 32;
mu = 0.005;

fprintf('LMS filter order : %d\n',filterOrder);
fprintf('LMS step size    : %.4f\n\n',mu);

%% =========================================================
% RUN ALL SNR TESTS
% =========================================================

for test = 1:numTests

    targetSNR = snrLevels(test);

    fprintf('---------------------------------------------\n');
    fprintf('Testing SNR = %d dB\n',targetSNR);
    fprintf('---------------------------------------------\n');

    %% CALCULATE REQUIRED NOISE POWER

    cleanPower = mean(clean.^2);
    noisePower = mean(noise.^2);

    requiredNoisePower = ...
        cleanPower / (10^(targetSNR/10));

    scaleFactor = ...
        sqrt(requiredNoisePower / noisePower);

    %% SCALE NOISE

    scaledNoise = noise * scaleFactor;

    %% CREATE NOISY SPEECH

    noisy = clean + scaledNoise;

    %% CALCULATE ACTUAL INPUT SNR

    actualNoise = noisy - clean;

    inputSNR(test) = ...
        10 * log10( ...
        sum(clean.^2) / sum(actualNoise.^2));

    %% =====================================================
    % LMS ADAPTIVE NOISE CANCELLATION
    % ======================================================

    w = zeros(filterOrder,1);

    xBuffer = zeros(filterOrder,1);

    estimatedNoise = zeros(N,1);

    enhanced = zeros(N,1);

    for n = 1:N

        xBuffer = ...
            [noise(n); xBuffer(1:end-1)];

        estimatedNoise(n) = ...
            w' * xBuffer;

        enhanced(n) = ...
            noisy(n) - estimatedNoise(n);

        errorSignal = enhanced(n);

        w = ...
            w + mu * errorSignal * xBuffer;

    end

    %% CALCULATE OUTPUT SNR

    residualNoise = enhanced - clean;

    outputSNR(test) = ...
        10 * log10( ...
        sum(clean.^2) / sum(residualNoise.^2));

    %% SNR IMPROVEMENT

    improvement(test) = ...
        outputSNR(test) - inputSNR(test);

    %% OUTPUT MSE

    mseOutput(test) = ...
        mean(residualNoise.^2);

    %% DISPLAY RESULTS

    fprintf('Input SNR       : %.2f dB\n',inputSNR(test));
    fprintf('Output SNR      : %.2f dB\n',outputSNR(test));
    fprintf('Improvement     : %.2f dB\n',improvement(test));
    fprintf('Output MSE      : %.8f\n\n',mseOutput(test));

    %% SAVE -10 dB RESULT

    if targetSNR == -10

        audiowrite( ...
            fullfile(outputFolder,...
            'AERIS_LMS_minus10dB.wav'), ...
            enhanced,Fs);

    end

    %% SAVE -20 dB RESULT

    if targetSNR == -20

        audiowrite( ...
            fullfile(outputFolder,...
            'AERIS_LMS_minus20dB.wav'), ...
            enhanced,Fs);

    end

end

%% =========================================================
% FINAL RESULTS TABLE
% =========================================================

fprintf('\n');
fprintf('=================================================\n');
fprintf('              AERIS ROBUSTNESS RESULTS\n');
fprintf('=================================================\n\n');

fprintf('%10s %15s %15s %15s\n', ...
    'SNR','Input SNR','Output SNR','Improvement');

fprintf('-------------------------------------------------\n');

for k = 1:numTests

    fprintf('%10d %15.2f %15.2f %15.2f\n', ...
        snrLevels(k), ...
        inputSNR(k), ...
        outputSNR(k), ...
        improvement(k));

end

fprintf('\n');

%% BEST RESULT

[bestImprovement,bestIndex] = max(improvement);

fprintf('Best improvement : %.2f dB\n',bestImprovement);
fprintf('At input SNR     : %.0f dB\n\n',snrLevels(bestIndex));

%% =========================================================
% GRAPH 1 - INPUT VS OUTPUT SNR
% =========================================================

figure;

plot(snrLevels,inputSNR,'-o','LineWidth',2);

hold on;

plot(snrLevels,outputSNR,'-o','LineWidth',2);

xlabel('Input SNR (dB)');
ylabel('SNR (dB)');

title('AERIS LMS - Input vs Output SNR');

legend('Input SNR','Output SNR','Location','best');

grid on;

saveas(gcf, ...
    fullfile(outputFolder,...
    'AERIS_Input_vs_Output_SNR.png'));

%% =========================================================
% GRAPH 2 - SNR IMPROVEMENT
% =========================================================

figure;

bar(snrLevels,improvement);

xlabel('Input SNR (dB)');
ylabel('SNR Improvement (dB)');

title('AERIS LMS - SNR Improvement');

grid on;

saveas(gcf, ...
    fullfile(outputFolder,...
    'AERIS_SNR_Improvement.png'));

%% =========================================================
% SAVE RESULTS TO CSV
% =========================================================

resultsTable = table( ...
    snrLevels(:), ...
    inputSNR, ...
    outputSNR, ...
    improvement, ...
    mseOutput, ...
    'VariableNames', ...
    {'TargetSNR_dB', ...
     'InputSNR_dB', ...
     'OutputSNR_dB', ...
     'Improvement_dB', ...
     'OutputMSE'});

writetable(resultsTable, ...
    fullfile(outputFolder,...
    'AERIS_SNR_Results.csv'));

%% =========================================================
% FINAL MESSAGE
% =========================================================

fprintf('=================================================\n');
fprintf('        AERIS LOW-SNR TEST COMPLETE\n');
fprintf('=================================================\n\n');

fprintf('Results saved inside:\n');
fprintf('%s\n\n',outputFolder);

fprintf('Generated files:\n');
fprintf('1. AERIS_Input_vs_Output_SNR.png\n');
fprintf('2. AERIS_SNR_Improvement.png\n');
fprintf('3. AERIS_SNR_Results.csv\n');
fprintf('4. AERIS_LMS_minus10dB.wav\n');
fprintf('5. AERIS_LMS_minus20dB.wav\n');

fprintf('\n=================================================\n');