%% ================================================================
%                         AERIS MASTER
%              ADAPTIVE NOISE CANCELLATION SYSTEM
%
%  LMS + NLMS
%  Low-SNR robustness testing
%  Stable parameter handling
%  Automatic SNR calculation
%  WAV + CSV + TXT + PNG outputs
%
%  MATLAB R2026a
%% ================================================================

clear;
clc;
close all;

fprintf('\n');
fprintf('================================================================\n');
fprintf('                    AERIS MASTER\n');
fprintf('       ADAPTIVE NOISE CANCELLATION SYSTEM\n');
fprintf('================================================================\n\n');

%% ================================================================
% PATHS
%% ================================================================

baseFolder = 'C:\Users\nuppe\Documents\MATLAB\AERIS_SIH';

cleanFolder = fullfile(baseFolder,'DATASET','CLEAN');
noiseFolder = fullfile(baseFolder,'DATASET','NOISE');
outputFolder = fullfile(baseFolder,'OUTPUT');

if ~exist(outputFolder,'dir')
    mkdir(outputFolder);
end

fprintf('Base folder:\n%s\n\n',baseFolder);
fprintf('Clean dataset:\n%s\n\n',cleanFolder);
fprintf('Noise dataset:\n%s\n\n',noiseFolder);
fprintf('Output folder:\n%s\n\n',outputFolder);

%% ================================================================
% FIND DATASET FILES
%% ================================================================

cleanFiles = dir(fullfile(cleanFolder,'**','*.flac'));
noiseWavFiles = dir(fullfile(noiseFolder,'**','*.wav'));

% Also allow WAV noise files
noiseFiles = noiseWavFiles;

fprintf('Clean speech files found : %d\n',numel(cleanFiles));
fprintf('Noise files found        : %d\n\n',numel(noiseFiles));

if isempty(cleanFiles)
    error('No clean speech files found.');
end

if isempty(noiseFiles)
    error('No noise files found.');
end

%% ================================================================
% SELECT TEST FILES
%% ================================================================

cleanIndex = 1;
noiseIndex = 1;

cleanPath = fullfile(cleanFiles(cleanIndex).folder,...
                     cleanFiles(cleanIndex).name);

noisePath = fullfile(noiseFiles(noiseIndex).folder,...
                     noiseFiles(noiseIndex).name);

fprintf('Selected clean speech:\n%s\n\n',cleanPath);
fprintf('Selected noise:\n%s\n\n',noisePath);

%% ================================================================
% READ AUDIO
%% ================================================================

[clean,FsClean] = audioread(cleanPath);
[noise,FsNoise] = audioread(noisePath);

% Convert stereo to mono
if size(clean,2) > 1
    clean = mean(clean,2);
end

if size(noise,2) > 1
    noise = mean(noise,2);
end

fprintf('Sampling rate:\n');
fprintf('Clean : %d Hz\n',FsClean);
fprintf('Noise : %d Hz\n\n',FsNoise);

%% ================================================================
% RESAMPLE IF REQUIRED
%% ================================================================

Fs = 16000;

if FsClean ~= Fs
    clean = resample(clean,Fs,FsClean);
end

if FsNoise ~= Fs
    noise = resample(noise,Fs,FsNoise);
end

%% ================================================================
% LIMIT TEST DURATION
%% ================================================================

testDuration = 10;

N = min([length(clean),length(noise),round(testDuration*Fs)]);

clean = clean(1:N);
noise = noise(1:N);

% Remove DC
clean = clean - mean(clean);
noise = noise - mean(noise);

fprintf('Test duration : %.2f seconds\n',N/Fs);
fprintf('Samples       : %d\n\n',N);

%% ================================================================
% SYSTEM PARAMETERS
%% ================================================================

LMS_ORDER = 16;

% Base LMS parameter.
% The actual value is automatically reduced if necessary
% to guarantee stability.
LMS_MU_BASE = 0.010;

NLMS_ORDER = 32;
NLMS_MU = 0.50;
NLMS_EPS = 1e-8;

% Safety leakage for LMS
LMS_LEAK = 1e-6;

% Maximum allowed output magnitude during processing
MAX_SIGNAL = 20;

SNR_POINTS = [5 0 -5 -10 -15 -20];

fprintf('================================================================\n');
fprintf('                     SYSTEM PARAMETERS\n');
fprintf('================================================================\n\n');

fprintf('Sampling rate        : %d Hz\n',Fs);
fprintf('LMS filter order     : %d\n',LMS_ORDER);
fprintf('LMS requested mu     : %.4f\n',LMS_MU_BASE);
fprintf('NLMS filter order    : %d\n',NLMS_ORDER);
fprintf('NLMS step size       : %.2f\n',NLMS_MU);
fprintf('NLMS epsilon         : %.1e\n',NLMS_EPS);
fprintf('SNR test points      : ');

fprintf('%g ',SNR_POINTS);

fprintf('dB\n\n');

%% ================================================================
% RESULT ARRAYS
%% ================================================================

numTests = length(SNR_POINTS);

InputSNR = nan(numTests,1);

LMS_OutputSNR = nan(numTests,1);
LMS_Improvement = nan(numTests,1);
LMS_MSE = nan(numTests,1);
LMS_Stable = false(numTests,1);

NLMS_OutputSNR = nan(numTests,1);
NLMS_Improvement = nan(numTests,1);
NLMS_MSE = nan(numTests,1);
NLMS_Stable = false(numTests,1);

LMS_MuUsed = nan(numTests,1);

%% ================================================================
% STORAGE FOR -10 dB RESULTS
%% ================================================================

cleanMinus10 = [];
noisyMinus10 = [];
lmsMinus10 = [];
nlmsMinus10 = [];

%% ================================================================
% MAIN EXPERIMENT
%% ================================================================

fprintf('\n');
fprintf('================================================================\n');
fprintf('                  RUNNING AERIS EXPERIMENTS\n');
fprintf('================================================================\n');

for k = 1:numTests

    targetSNR = SNR_POINTS(k);

    fprintf('\n');
    fprintf('------------------------------------------------------------\n');
    fprintf('Testing target SNR = %g dB\n',targetSNR);
    fprintf('------------------------------------------------------------\n');

    %% ------------------------------------------------------------
    % CREATE EXACT-SNR NOISY SIGNAL
    %% ------------------------------------------------------------

    cleanPower = mean(clean.^2);
    noisePower = mean(noise.^2);

    requiredNoisePower = cleanPower / (10^(targetSNR/10));

    noiseScale = sqrt(requiredNoisePower/(noisePower + eps));

    scaledNoise = noise * noiseScale;

    noisy = clean + scaledNoise;

    % Exact measured input SNR
    actualInputSNR = calculateSNR(clean,noisy-clean);

    InputSNR(k) = actualInputSNR;

    fprintf('Target input SNR : %8.2f dB\n',targetSNR);
    fprintf('Actual input SNR : %8.2f dB\n\n',actualInputSNR);

    %% ------------------------------------------------------------
    % STABLE LMS STEP SIZE
    %% ------------------------------------------------------------

    refPower = mean(scaledNoise.^2);

    % Classical LMS stability condition:
    %
    %       0 < mu < 2/(M*Px)
    %
    % Use only a conservative fraction of this bound.

    theoreticalMaxMu = 2 / (LMS_ORDER * refPower + eps);

    safeMu = min(LMS_MU_BASE,0.10*theoreticalMaxMu);

    % Additional lower bound protection
    safeMu = max(safeMu,1e-7);

    LMS_MuUsed(k) = safeMu;

    fprintf('Stable LMS mu     : %.8f\n\n',safeMu);

    %% ------------------------------------------------------------
    % LMS
    %% ------------------------------------------------------------

    fprintf('Running LMS...\n');

    [lmsOutput,lmsNoise,stableLMS] = ...
        runStableLMS(noisy,scaledNoise,LMS_ORDER,...
                     safeMu,LMS_LEAK,MAX_SIGNAL);

    LMS_Stable(k) = stableLMS;

    if stableLMS

        lmsError = clean - lmsOutput;

        lmsMSE = mean(lmsError.^2);

        lmsSNR = calculateSNR(clean,lmsError);

        lmsImprovement = lmsSNR - actualInputSNR;

        LMS_OutputSNR(k) = lmsSNR;
        LMS_Improvement(k) = lmsImprovement;
        LMS_MSE(k) = lmsMSE;

    else

        LMS_OutputSNR(k) = NaN;
        LMS_Improvement(k) = NaN;
        LMS_MSE(k) = NaN;

        fprintf('LMS marked unstable.\n');

    end

    %% ------------------------------------------------------------
    % NLMS
    %% ------------------------------------------------------------

    fprintf('Running NLMS...\n');

    [nlmsOutput,nlmsNoise,stableNLMS] = ...
        runNLMS(noisy,scaledNoise,NLMS_ORDER,...
                NLMS_MU,NLMS_EPS,MAX_SIGNAL);

    NLMS_Stable(k) = stableNLMS;

    if stableNLMS

        nlmsError = clean - nlmsOutput;

        nlmsMSE = mean(nlmsError.^2);

        nlmsSNR = calculateSNR(clean,nlmsError);

        nlmsImprovement = nlmsSNR - actualInputSNR;

        NLMS_OutputSNR(k) = nlmsSNR;
        NLMS_Improvement(k) = nlmsImprovement;
        NLMS_MSE(k) = nlmsMSE;

    else

        NLMS_OutputSNR(k) = NaN;
        NLMS_Improvement(k) = NaN;
        NLMS_MSE(k) = NaN;

        fprintf('NLMS marked unstable.\n');

    end

    %% ------------------------------------------------------------
    % DISPLAY RESULTS
    %% ------------------------------------------------------------

    fprintf('\n');
    fprintf('LMS RESULTS\n');
    fprintf('Input SNR       : %8.2f dB\n',actualInputSNR);
    fprintf('Output SNR      : %8.2f dB\n',LMS_OutputSNR(k));
    fprintf('Improvement     : %8.2f dB\n',LMS_Improvement(k));
    fprintf('MSE             : %.8f\n',LMS_MSE(k));

    fprintf('\n');
    fprintf('NLMS RESULTS\n');
    fprintf('Input SNR       : %8.2f dB\n',actualInputSNR);
    fprintf('Output SNR      : %8.2f dB\n',NLMS_OutputSNR(k));
    fprintf('Improvement     : %8.2f dB\n',NLMS_Improvement(k));
    fprintf('MSE             : %.8f\n',NLMS_MSE(k));

    %% ------------------------------------------------------------
    % SAVE -10 dB SIGNALS
    %% ------------------------------------------------------------

    if targetSNR == -10

        cleanMinus10 = clean;
        noisyMinus10 = noisy;
        lmsMinus10 = lmsOutput;
        nlmsMinus10 = nlmsOutput;

        writeSafeAudio( ...
            fullfile(outputFolder,'AERIS_CLEAN_-10dB.wav'),...
            cleanMinus10,Fs);

        writeSafeAudio( ...
            fullfile(outputFolder,'AERIS_NOISY_-10dB.wav'),...
            noisyMinus10,Fs);

        writeSafeAudio( ...
            fullfile(outputFolder,'AERIS_FINAL_LMS_-10dB.wav'),...
            lmsMinus10,Fs);

        writeSafeAudio( ...
            fullfile(outputFolder,'AERIS_FINAL_NLMS_-10dB.wav'),...
            nlmsMinus10,Fs);

    end

    %% ------------------------------------------------------------
    % SAVE -20 dB SIGNALS
    %% ------------------------------------------------------------

    if targetSNR == -20

        writeSafeAudio( ...
            fullfile(outputFolder,'AERIS_FINAL_LMS_-20dB.wav'),...
            lmsOutput,Fs);

        writeSafeAudio( ...
            fullfile(outputFolder,'AERIS_FINAL_NLMS_-20dB.wav'),...
            nlmsOutput,Fs);

    end

end

%% ================================================================
% FINAL RESULTS TABLE
%% ================================================================

fprintf('\n');
fprintf('================================================================\n');
fprintf('                       AERIS FINAL RESULTS\n');
fprintf('================================================================\n\n');

Results = table( ...
    SNR_POINTS(:), ...
    InputSNR, ...
    LMS_OutputSNR, ...
    LMS_Improvement, ...
    LMS_MSE, ...
    NLMS_OutputSNR, ...
    NLMS_Improvement, ...
    NLMS_MSE, ...
    LMS_MuUsed, ...
    LMS_Stable, ...
    NLMS_Stable, ...
    'VariableNames',{ ...
    'TargetSNR',...
    'InputSNR',...
    'LMS_OutputSNR',...
    'LMS_Improvement',...
    'LMS_MSE',...
    'NLMS_OutputSNR',...
    'NLMS_Improvement',...
    'NLMS_MSE',...
    'LMS_MuUsed',...
    'LMS_Stable',...
    'NLMS_Stable'});

disp(Results);

%% ================================================================
% VALID RESULTS
%% ================================================================

validLMS = isfinite(LMS_Improvement);

validNLMS = isfinite(NLMS_Improvement);

%% ================================================================
% LMS STATISTICS
%% ================================================================

if any(validLMS)

    avgLMS = mean(LMS_Improvement(validLMS));
    minLMS = min(LMS_Improvement(validLMS));
    maxLMS = max(LMS_Improvement(validLMS));

    [bestLMS,bestLMSIndex] = ...
        max(LMS_Improvement(validLMS));

    validIndices = find(validLMS);
    bestLMSRealIndex = validIndices(bestLMSIndex);

    bestLMSSNR = SNR_POINTS(bestLMSRealIndex);

else

    avgLMS = NaN;
    minLMS = NaN;
    maxLMS = NaN;
    bestLMS = NaN;
    bestLMSSNR = NaN;

end

%% ================================================================
% NLMS STATISTICS
%% ================================================================

if any(validNLMS)

    avgNLMS = mean(NLMS_Improvement(validNLMS));
    minNLMS = min(NLMS_Improvement(validNLMS));
    maxNLMS = max(NLMS_Improvement(validNLMS));

    [bestNLMS,bestNLMSIndex] = ...
        max(NLMS_Improvement(validNLMS));

    validIndices = find(validNLMS);
    bestNLMSRealIndex = validIndices(bestNLMSIndex);

    bestNLMSSNR = SNR_POINTS(bestNLMSRealIndex);

else

    avgNLMS = NaN;
    minNLMS = NaN;
    maxNLMS = NaN;
    bestNLMS = NaN;
    bestNLMSSNR = NaN;

end

%% ================================================================
% SELECT BEST ALGORITHM
%
% Use average improvement as the main criterion.
%% ================================================================

if isfinite(avgLMS) && isfinite(avgNLMS)

    if avgLMS >= avgNLMS
        bestAlgorithm = 'LMS';
        bestAverage = avgLMS;
    else
        bestAlgorithm = 'NLMS';
        bestAverage = avgNLMS;
    end

elseif isfinite(avgLMS)

    bestAlgorithm = 'LMS';
    bestAverage = avgLMS;

elseif isfinite(avgNLMS)

    bestAlgorithm = 'NLMS';
    bestAverage = avgNLMS;

else

    bestAlgorithm = 'NONE';
    bestAverage = NaN;

end

%% ================================================================
% DISPLAY BEST RESULTS
%% ================================================================

fprintf('\n');
fprintf('================================================================\n');
fprintf('                         BEST RESULTS\n');
fprintf('================================================================\n\n');

fprintf('LMS\n');
fprintf('Average improvement : %.2f dB\n',avgLMS);
fprintf('Minimum improvement : %.2f dB\n',minLMS);
fprintf('Maximum improvement : %.2f dB\n',maxLMS);
fprintf('Best improvement    : %.2f dB\n',bestLMS);
fprintf('Best test SNR       : %g dB\n\n',bestLMSSNR);

fprintf('NLMS\n');
fprintf('Average improvement : %.2f dB\n',avgNLMS);
fprintf('Minimum improvement : %.2f dB\n',minNLMS);
fprintf('Maximum improvement : %.2f dB\n',maxNLMS);
fprintf('Best improvement    : %.2f dB\n',bestNLMS);
fprintf('Best test SNR       : %g dB\n\n',bestNLMSSNR);

fprintf('BEST OVERALL ALGORITHM : %s\n',bestAlgorithm);
fprintf('Best average improvement : %.2f dB\n',bestAverage);

%% ================================================================
% SAVE CSV
%% ================================================================

csvPath = fullfile(outputFolder,'AERIS_MASTER_RESULTS.csv');

writetable(Results,csvPath);

%% ================================================================
% SNR COMPARISON PLOT
%% ================================================================

figure('Name','AERIS SNR Comparison');

plot(SNR_POINTS,InputSNR,'o-','LineWidth',1.5);
hold on;

plot(SNR_POINTS,LMS_OutputSNR,'s-','LineWidth',1.5);
plot(SNR_POINTS,NLMS_OutputSNR,'^-','LineWidth',1.5);

grid on;

xlabel('Input / Target SNR (dB)');
ylabel('SNR (dB)');

title('AERIS - Input vs Output SNR');

legend('Input SNR','LMS Output','NLMS Output',...
       'Location','best');

saveas(gcf,...
    fullfile(outputFolder,'AERIS_MASTER_SNR_COMPARISON.png'));

%% ================================================================
% IMPROVEMENT PLOT
%% ================================================================

figure('Name','AERIS SNR Improvement');

plot(SNR_POINTS,LMS_Improvement,'s-','LineWidth',1.5);
hold on;

plot(SNR_POINTS,NLMS_Improvement,'^-','LineWidth',1.5);

yline(0,'--');

grid on;

xlabel('Input SNR (dB)');
ylabel('SNR Improvement (dB)');

title('AERIS - SNR Improvement');

legend('LMS','NLMS','Zero Improvement',...
       'Location','best');

saveas(gcf,...
    fullfile(outputFolder,'AERIS_MASTER_IMPROVEMENT.png'));

%% ================================================================
% MSE PLOT
%% ================================================================

figure('Name','AERIS MSE');

semilogy(SNR_POINTS,LMS_MSE,'s-','LineWidth',1.5);
hold on;

semilogy(SNR_POINTS,NLMS_MSE,'^-','LineWidth',1.5);

grid on;

xlabel('Input SNR (dB)');
ylabel('MSE');

title('AERIS - Mean Squared Error');

legend('LMS','NLMS','Location','best');

saveas(gcf,...
    fullfile(outputFolder,'AERIS_MASTER_MSE.png'));

%% ================================================================
% -10 dB WAVEFORM
%% ================================================================

if ~isempty(cleanMinus10)

    t = (0:length(cleanMinus10)-1)/Fs;

    figure('Name','AERIS -10 dB Waveform');

    plot(t,cleanMinus10);
    hold on;

    plot(t,noisyMinus10);
    plot(t,lmsMinus10);
    plot(t,nlmsMinus10);

    grid on;

    xlabel('Time (seconds)');
    ylabel('Amplitude');

    title('AERIS -10 dB ANC Comparison');

    legend('Clean','Noisy','LMS','NLMS',...
           'Location','best');

    saveas(gcf,...
        fullfile(outputFolder,'AERIS_-10dB_WAVEFORM.png'));

end

%% ================================================================
% -10 dB SPECTROGRAMS
%% ================================================================

if ~isempty(cleanMinus10)

    figure('Name','AERIS -10 dB Noisy Spectrogram');

    spectrogram(noisyMinus10,512,384,1024,Fs,'yaxis');
    title('AERIS -10 dB Noisy Speech');

    saveas(gcf,...
        fullfile(outputFolder,'AERIS_-10dB_NOISY_SPECTROGRAM.png'));

    figure('Name','AERIS -10 dB LMS Spectrogram');

    spectrogram(lmsMinus10,512,384,1024,Fs,'yaxis');
    title('AERIS -10 dB LMS Enhanced Speech');

    saveas(gcf,...
        fullfile(outputFolder,'AERIS_-10dB_LMS_SPECTROGRAM.png'));

    figure('Name','AERIS -10 dB NLMS Spectrogram');

    spectrogram(nlmsMinus10,512,384,1024,Fs,'yaxis');
    title('AERIS -10 dB NLMS Enhanced Speech');

    saveas(gcf,...
        fullfile(outputFolder,'AERIS_-10dB_NLMS_SPECTROGRAM.png'));

end

%% ================================================================
% TEXT REPORT
%% ================================================================

reportPath = fullfile(outputFolder,'AERIS_MASTER_REPORT.txt');

fid = fopen(reportPath,'w');

fprintf(fid,'============================================================\n');
fprintf(fid,'                    AERIS MASTER REPORT\n');
fprintf(fid,'============================================================\n\n');

fprintf(fid,'Base folder:\n%s\n\n',baseFolder);

fprintf(fid,'Clean files : %d\n',numel(cleanFiles));
fprintf(fid,'Noise files : %d\n',numel(noiseFiles));

fprintf(fid,'Sampling rate : %d Hz\n',Fs);
fprintf(fid,'Samples       : %d\n',N);
fprintf(fid,'Duration      : %.2f seconds\n\n',N/Fs);

fprintf(fid,'LMS filter order : %d\n',LMS_ORDER);
fprintf(fid,'LMS requested mu : %.6f\n',LMS_MU_BASE);

fprintf(fid,'NLMS filter order : %d\n',NLMS_ORDER);
fprintf(fid,'NLMS mu          : %.6f\n',NLMS_MU);
fprintf(fid,'NLMS epsilon     : %.2e\n\n',NLMS_EPS);

fprintf(fid,'RESULTS\n');
fprintf(fid,'------------------------------------------------------------\n');

for k = 1:numTests

    fprintf(fid,...
        'Target=%6.1f dB | Input=%8.2f | LMS=%8.2f | LMS Imp=%8.2f | NLMS=%8.2f | NLMS Imp=%8.2f\n',...
        SNR_POINTS(k),...
        InputSNR(k),...
        LMS_OutputSNR(k),...
        LMS_Improvement(k),...
        NLMS_OutputSNR(k),...
        NLMS_Improvement(k));

end

fprintf(fid,'\n');
fprintf(fid,'LMS average improvement  : %.2f dB\n',avgLMS);
fprintf(fid,'LMS minimum improvement  : %.2f dB\n',minLMS);
fprintf(fid,'LMS maximum improvement  : %.2f dB\n',maxLMS);

fprintf(fid,'\n');

fprintf(fid,'NLMS average improvement : %.2f dB\n',avgNLMS);
fprintf(fid,'NLMS minimum improvement : %.2f dB\n',minNLMS);
fprintf(fid,'NLMS maximum improvement : %.2f dB\n',maxNLMS);

fprintf(fid,'\n');
fprintf(fid,'BEST ALGORITHM : %s\n',bestAlgorithm);
fprintf(fid,'BEST AVERAGE IMPROVEMENT : %.2f dB\n',bestAverage);

fprintf(fid,'\n============================================================\n');

fclose(fid);

%% ================================================================
% FINISH
%% ================================================================

fprintf('\n');
fprintf('================================================================\n');
fprintf('                  AERIS MASTER COMPLETE\n');
fprintf('================================================================\n\n');

fprintf('BEST ALGORITHM : %s\n',bestAlgorithm);
fprintf('BEST AVERAGE IMPROVEMENT : %.2f dB\n\n',bestAverage);

fprintf('Results saved to:\n%s\n\n',outputFolder);

fprintf('Generated files include:\n');
fprintf('1. AERIS_MASTER_RESULTS.csv\n');
fprintf('2. AERIS_MASTER_REPORT.txt\n');
fprintf('3. AERIS_MASTER_SNR_COMPARISON.png\n');
fprintf('4. AERIS_MASTER_IMPROVEMENT.png\n');
fprintf('5. AERIS_MASTER_MSE.png\n');
fprintf('6. AERIS_-10dB_WAVEFORM.png\n');
fprintf('7. AERIS_-10dB_NOISY_SPECTROGRAM.png\n');
fprintf('8. AERIS_-10dB_LMS_SPECTROGRAM.png\n');
fprintf('9. AERIS_-10dB_NLMS_SPECTROGRAM.png\n');
fprintf('10. AERIS_FINAL_LMS_-10dB.wav\n');
fprintf('11. AERIS_FINAL_NLMS_-10dB.wav\n');
fprintf('12. AERIS_FINAL_LMS_-20dB.wav\n');
fprintf('13. AERIS_FINAL_NLMS_-20dB.wav\n');

fprintf('\n');
fprintf('================================================================\n');
fprintf('                AERIS MASTER TEST COMPLETE\n');
fprintf('================================================================\n');

%% ================================================================
% LOCAL FUNCTION: SNR
%% ================================================================

function snrDB = calculateSNR(cleanSignal,errorSignal)

    cleanPower = mean(cleanSignal.^2);
    errorPower = mean(errorSignal.^2);

    if errorPower <= 0 || ~isfinite(errorPower)

        snrDB = NaN;

    else

        snrDB = 10*log10( ...
            (cleanPower + eps) / ...
            (errorPower + eps));

    end

end

%% ================================================================
% LOCAL FUNCTION: STABLE LMS
%% ================================================================

function [output,estimatedNoise,isStable] = ...
    runStableLMS(desired,reference,M,mu,leak,maxSignal)

    N = length(desired);

    weights = zeros(M,1);

    output = zeros(N,1);
    estimatedNoise = zeros(N,1);

    isStable = true;

    for n = M:N

        x = reference(n:-1:n-M+1);

        % Estimated noise
        y = weights' * x;

        % ANC error / enhanced speech
        e = desired(n) - y;

        % Leaky LMS update
        weights = (1-leak)*weights + mu*e*x;

        output(n) = e;
        estimatedNoise(n) = y;

        % Numerical stability protection
        if any(~isfinite(weights)) || ...
           ~isfinite(e) || ...
           abs(e) > maxSignal

            isStable = false;

            output(n:end) = NaN;
            estimatedNoise(n:end) = NaN;

            return;
        end

    end

end

%% ================================================================
% LOCAL FUNCTION: NLMS
%% ================================================================

function [output,estimatedNoise,isStable] = ...
    runNLMS(desired,reference,M,mu,epsilon,maxSignal)

    N = length(desired);

    weights = zeros(M,1);

    output = zeros(N,1);
    estimatedNoise = zeros(N,1);

    isStable = true;

    for n = M:N

        x = reference(n:-1:n-M+1);

        % Adaptive noise estimate
        y = weights' * x;

        % ANC output
        e = desired(n) - y;

        % Normalized LMS update
        normalization = epsilon + (x' * x);

        weights = weights + ...
                  (mu/normalization) * e * x;

        output(n) = e;
        estimatedNoise(n) = y;

        % Stability protection
        if any(~isfinite(weights)) || ...
           ~isfinite(e) || ...
           abs(e) > maxSignal

            isStable = false;

            output(n:end) = NaN;
            estimatedNoise(n:end) = NaN;

            return;
        end

    end

end

%% ================================================================
% LOCAL FUNCTION: SAFE AUDIO WRITE
%% ================================================================

function writeSafeAudio(filename,x,Fs)

    x = x(:);

    valid = isfinite(x);

    if ~any(valid)
        warning('Audio signal is invalid: %s',filename);
        return;
    end

    x(~valid) = 0;

    peak = max(abs(x));

    if peak > 0.99

        x = 0.99*x/peak;

    end

    audiowrite(filename,x,Fs);

end