%% =========================================================
% AERIS - LMS PARAMETER OPTIMIZATION
% AI/ML Adaptive Noise Cancellation - SIH
% =========================================================

clear;
clc;
close all;

fprintf('\n');
fprintf('=================================================\n');
fprintf('       AERIS LMS PARAMETER OPTIMIZATION\n');
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

%% PARAMETERS TO TEST

filterOrders = [16 32 64 128];

stepSizes = [0.001 0.003 0.005 0.008 0.01];

snrLevels = [5 0 -5 -10 -15 -20];

numOrders = length(filterOrders);
numSteps = length(stepSizes);
numSNR = length(snrLevels);

%% RESULT STORAGE

results = [];

resultIndex = 1;

%% =========================================================
% OPTIMIZATION
% =========================================================

fprintf('Testing LMS configurations...\n\n');

for orderIndex = 1:numOrders

    filterOrder = filterOrders(orderIndex);

    for stepIndex = 1:numSteps

        mu = stepSizes(stepIndex);

        improvementValues = zeros(numSNR,1);

        fprintf('Filter Order = %d | Mu = %.4f\n', ...
            filterOrder,mu);

        %% TEST EACH SNR

        for snrIndex = 1:numSNR

            targetSNR = snrLevels(snrIndex);

            %% CREATE REQUIRED NOISE LEVEL

            cleanPower = mean(clean.^2);
            noisePower = mean(noise.^2);

            requiredNoisePower = ...
                cleanPower / (10^(targetSNR/10));

            scaleFactor = ...
                sqrt(requiredNoisePower / noisePower);

            scaledNoise = noise * scaleFactor;

            %% CREATE NOISY SPEECH

            noisy = clean + scaledNoise;

            %% INPUT SNR

            inputNoise = noisy - clean;

            actualInputSNR = ...
                10 * log10( ...
                sum(clean.^2) / ...
                sum(inputNoise.^2));

            %% LMS INITIALIZATION

            w = zeros(filterOrder,1);

            xBuffer = zeros(filterOrder,1);

            enhanced = zeros(N,1);

            %% LMS PROCESSING

            for n = 1:N

                xBuffer = ...
                    [scaledNoise(n); ...
                     xBuffer(1:end-1)];

                estimatedNoise = ...
                    w' * xBuffer;

                enhanced(n) = ...
                    noisy(n) - estimatedNoise;

                errorSignal = enhanced(n);

                w = ...
                    w + mu * errorSignal * xBuffer;

            end

            %% OUTPUT SNR

            residualNoise = enhanced - clean;

            outputSNR = ...
                10 * log10( ...
                sum(clean.^2) / ...
                sum(residualNoise.^2));

            %% IMPROVEMENT

            improvement = ...
                outputSNR - actualInputSNR;

            improvementValues(snrIndex) = improvement;

        end

        %% AVERAGE PERFORMANCE

        averageImprovement = ...
            mean(improvementValues);

        minimumImprovement = ...
            min(improvementValues);

        maximumImprovement = ...
            max(improvementValues);

        %% STORE RESULT

        results(resultIndex,:) = ...
            [filterOrder ...
             mu ...
             averageImprovement ...
             minimumImprovement ...
             maximumImprovement];

        resultIndex = resultIndex + 1;

        fprintf('Average improvement : %.2f dB\n', ...
            averageImprovement);

        fprintf('Minimum improvement : %.2f dB\n', ...
            minimumImprovement);

        fprintf('Maximum improvement : %.2f dB\n\n', ...
            maximumImprovement);

    end
end

%% =========================================================
% FIND BEST CONFIGURATION
% =========================================================

[~,bestIndex] = max(results(:,3));

bestOrder = results(bestIndex,1);
bestMu = results(bestIndex,2);
bestAverage = results(bestIndex,3);
bestMinimum = results(bestIndex,4);
bestMaximum = results(bestIndex,5);

%% =========================================================
% DISPLAY FINAL RESULT
% =========================================================

fprintf('\n');
fprintf('=================================================\n');
fprintf('             AERIS OPTIMIZATION RESULT\n');
fprintf('=================================================\n\n');

fprintf('BEST LMS FILTER ORDER : %d\n',bestOrder);
fprintf('BEST STEP SIZE        : %.4f\n',bestMu);
fprintf('AVERAGE IMPROVEMENT   : %.2f dB\n',bestAverage);
fprintf('MINIMUM IMPROVEMENT   : %.2f dB\n',bestMinimum);
fprintf('MAXIMUM IMPROVEMENT   : %.2f dB\n\n',bestMaximum);

fprintf('=================================================\n');

%% =========================================================
% CREATE RESULTS TABLE
% =========================================================

resultsTable = table( ...
    results(:,1), ...
    results(:,2), ...
    results(:,3), ...
    results(:,4), ...
    results(:,5), ...
    'VariableNames', ...
    {'FilterOrder', ...
     'StepSize', ...
     'AverageImprovement_dB', ...
     'MinimumImprovement_dB', ...
     'MaximumImprovement_dB'});

%% SAVE CSV

csvFile = fullfile( ...
    outputFolder, ...
    'AERIS_LMS_Optimization_Results.csv');

writetable(resultsTable,csvFile);

%% =========================================================
% SAVE BEST CONFIGURATION
% =========================================================

configFile = fullfile( ...
    outputFolder, ...
    'AERIS_BEST_LMS_CONFIGURATION.txt');

fid = fopen(configFile,'w');

fprintf(fid,'AERIS BEST LMS CONFIGURATION\n');
fprintf(fid,'============================\n\n');

fprintf(fid,'Filter Order : %d\n',bestOrder);
fprintf(fid,'Step Size    : %.4f\n',bestMu);
fprintf(fid,'Average SNR Improvement : %.2f dB\n',bestAverage);
fprintf(fid,'Minimum SNR Improvement : %.2f dB\n',bestMinimum);
fprintf(fid,'Maximum SNR Improvement : %.2f dB\n',bestMaximum);

fclose(fid);

%% =========================================================
% PLOT OPTIMIZATION RESULTS
% =========================================================

figure;

plot(results(:,1), ...
     results(:,3), ...
     'o','LineWidth',2);

xlabel('LMS Filter Order');
ylabel('Average SNR Improvement (dB)');

title('AERIS LMS Parameter Optimization');

grid on;

saveas(gcf, ...
    fullfile(outputFolder, ...
    'AERIS_LMS_Optimization.png'));

%% =========================================================
% FINAL MESSAGE
% =========================================================

fprintf('\n');
fprintf('Optimization complete.\n\n');

fprintf('Generated files:\n');
fprintf('1. AERIS_LMS_Optimization_Results.csv\n');
fprintf('2. AERIS_BEST_LMS_CONFIGURATION.txt\n');
fprintf('3. AERIS_LMS_Optimization.png\n\n');

fprintf('Best configuration:\n');
fprintf('Filter Order = %d\n',bestOrder);
fprintf('Step Size    = %.4f\n',bestMu);

fprintf('\n');
fprintf('=================================================\n');
fprintf('              AERIS OPTIMIZATION DONE\n');
fprintf('=================================================\n');