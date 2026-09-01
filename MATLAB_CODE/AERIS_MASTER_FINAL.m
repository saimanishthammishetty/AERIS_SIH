%% ================================================================
%                    AERIS MASTER FINAL
%       ADAPTIVE NOISE CANCELLATION + AI-READY DATA GENERATION
% ================================================================
%
% Final MATLAB pipeline for AERIS SIH project.
%
% Pipeline:
% Dataset -> Controlled Noise -> LMS/NLMS ANC -> Evaluation
%         -> WAV/Graphs/CSV -> AI-ready feature dataset
%
% ================================================================

clear;
clc;
close all;

%% ================================================================
% 1. PATHS
% ================================================================

BASE = 'C:\Users\nuppe\Documents\MATLAB\AERIS_SIH';

CLEAN_FOLDER = fullfile(BASE,'DATASET','CLEAN');
NOISE_FOLDER = fullfile(BASE,'DATASET','NOISE');
OUTPUT_FOLDER = fullfile(BASE,'OUTPUT');
AI_FOLDER = fullfile(OUTPUT_FOLDER,'AERIS_AI_READY');

if ~exist(OUTPUT_FOLDER,'dir')
    mkdir(OUTPUT_FOLDER);
end

if ~exist(AI_FOLDER,'dir')
    mkdir(AI_FOLDER);
end

fprintf('\n');
fprintf('================================================================\n');
fprintf('                    AERIS MASTER FINAL\n');
fprintf('       ADAPTIVE NOISE CANCELLATION SYSTEM\n');
fprintf('================================================================\n\n');

fprintf('Base folder:\n%s\n\n',BASE);

%% ================================================================
% 2. FIND DATASET FILES
% ================================================================

cleanFiles = [
    dir(fullfile(CLEAN_FOLDER,'**','*.flac'));
    dir(fullfile(CLEAN_FOLDER,'**','*.wav'))
];

noiseFiles = [
    dir(fullfile(NOISE_FOLDER,'**','*.wav'));
    dir(fullfile(NOISE_FOLDER,'**','*.flac'))
];

fprintf('Clean speech files found : %d\n',numel(cleanFiles));
fprintf('Noise files found        : %d\n\n',numel(noiseFiles));

if isempty(cleanFiles)
    error('No clean speech files found.');
end

if isempty(noiseFiles)
    error('No noise files found.');
end

%% ================================================================
% 3. SELECT FILES
% ================================================================

cleanPath = fullfile(cleanFiles(1).folder,cleanFiles(1).name);
noisePath = fullfile(noiseFiles(1).folder,noiseFiles(1).name);

fprintf('Selected clean speech:\n%s\n\n',cleanPath);
fprintf('Selected noise:\n%s\n\n',noisePath);

%% ================================================================
% 4. READ AUDIO
% ================================================================

[clean,FsClean] = audioread(cleanPath);
[noise,FsNoise] = audioread(noisePath);

% Convert stereo to mono
if size(clean,2) > 1
    clean = mean(clean,2);
end

if size(noise,2) > 1
    noise = mean(noise,2);
end

% Target sample rate
Fs = 16000;

if FsClean ~= Fs
    clean = resample(clean,Fs,FsClean);
end

if FsNoise ~= Fs
    noise = resample(noise,Fs,FsNoise);
end

clean = clean(:);
noise = noise(:);

fprintf('Sampling rate:\n');
fprintf('Clean : %d Hz\n',Fs);
fprintf('Noise : %d Hz\n\n',Fs);

%% ================================================================
% 5. NORMALIZE DATA
% ================================================================

clean = clean ./ (max(abs(clean)) + eps);
noise = noise ./ (max(abs(noise)) + eps);

% Use approximately 6 seconds maximum
N = min([length(clean), length(noise), round(6*Fs)]);

clean = clean(1:N);
noise = noise(1:N);

duration = N/Fs;

fprintf('Test duration : %.2f seconds\n',duration);
fprintf('Samples       : %d\n\n',N);

%% ================================================================
% 6. ANC PARAMETERS
% ================================================================

% LMS
LMS_ORDER = 16;

% Conservative base LMS step size
LMS_MU = 0.005;

% NLMS
NLMS_ORDER = 32;

% NLMS normalized step
NLMS_MU = 0.15;

% Numerical protection
EPS = 1e-8;

% SNR points
SNR_LIST = [5 0 -5 -10 -15 -20];

fprintf('================================================================\n');
fprintf('                     SYSTEM PARAMETERS\n');
fprintf('================================================================\n\n');

fprintf('Sampling rate       : %d Hz\n',Fs);
fprintf('LMS filter order    : %d\n',LMS_ORDER);
fprintf('LMS step size       : %.4f\n',LMS_MU);
fprintf('NLMS filter order   : %d\n',NLMS_ORDER);
fprintf('NLMS step size      : %.4f\n',NLMS_MU);
fprintf('NLMS epsilon        : %.1e\n',EPS);
fprintf('SNR test points     : ');

fprintf('%d ',SNR_LIST);
fprintf('dB\n\n');

%% ================================================================
% 7. PREALLOCATE RESULTS
% ================================================================

numTests = length(SNR_LIST);

TargetSNR = SNR_LIST(:);

InputSNR = zeros(numTests,1);

LMS_OutputSNR = NaN(numTests,1);
LMS_Improvement = NaN(numTests,1);
LMS_MSE = NaN(numTests,1);
LMS_Stable = false(numTests,1);

NLMS_OutputSNR = NaN(numTests,1);
NLMS_Improvement = NaN(numTests,1);
NLMS_MSE = NaN(numTests,1);
NLMS_Stable = false(numTests,1);

% Store important outputs
savedLMS10 = [];
savedNLMS10 = [];
savedNoisy10 = [];

savedLMS20 = [];
savedNLMS20 = [];

%% ================================================================
% 8. MAIN EXPERIMENT
% ================================================================

fprintf('\n');
fprintf('================================================================\n');
fprintf('                  RUNNING AERIS EXPERIMENTS\n');
fprintf('================================================================\n');

for k = 1:numTests

    targetSNR = SNR_LIST(k);

    fprintf('\n------------------------------------------------------------\n');
    fprintf('Testing target SNR = %d dB\n',targetSNR);
    fprintf('------------------------------------------------------------\n');

    %% ------------------------------------------------------------
    % CREATE CONTROLLED NOISE
    % -------------------------------------------------------------

    cleanPower = mean(clean.^2);
    noisePower = mean(noise.^2);

    requiredNoisePower = cleanPower / (10^(targetSNR/10));

    scale = sqrt(requiredNoisePower/(noisePower + EPS));

    referenceNoise = noise * scale;

    noisy = clean + referenceNoise;

    % Safety normalization only for playback/storage.
    % Internal calculations use the unmodified noisy signal.
    inputSNR = 10*log10( ...
        mean(clean.^2) / ...
        (mean(referenceNoise.^2) + EPS) );

    InputSNR(k) = inputSNR;

    fprintf('Target input SNR : %8.2f dB\n',targetSNR);
    fprintf('Actual input SNR : %8.2f dB\n',inputSNR);

    %% ------------------------------------------------------------
    % LMS STEP-SIZE SAFETY
    % -------------------------------------------------------------

    % Estimate signal power.
    Px = mean(referenceNoise.^2);

    % Conservative upper bound for LMS.
    muMax = 0.5/(LMS_ORDER*(Px + EPS));

    stableMu = min(LMS_MU,0.25*muMax);

    % Never allow an impractically tiny value.
    stableMu = max(stableMu,1e-5);

    fprintf('\nStable LMS mu : %.8f\n',stableMu);

    %% ------------------------------------------------------------
    % LMS
    % -------------------------------------------------------------

    fprintf('\nRunning LMS...\n');

    [lmsEnhanced,lmsEstimated,lmsStable] = ...
        aerisLMS(noisy,referenceNoise,LMS_ORDER,stableMu,clean);

    LMS_Stable(k) = lmsStable;

    if lmsStable

        lmsError = lmsEnhanced - clean;

        lmsMSE = mean(lmsError.^2);

        lmsOutputSNR = 10*log10( ...
            mean(clean.^2)/(lmsMSE + EPS));

        LMS_OutputSNR(k) = lmsOutputSNR;
        LMS_Improvement(k) = lmsOutputSNR-inputSNR;
        LMS_MSE(k) = lmsMSE;

    end

    %% ------------------------------------------------------------
    % NLMS
    % -------------------------------------------------------------

    fprintf('Running NLMS...\n');

    [nlmsEnhanced,nlmsEstimated,nlmsStable] = ...
        aerisNLMS(noisy,referenceNoise,NLMS_ORDER,NLMS_MU,EPS,clean);

    NLMS_Stable(k) = nlmsStable;

    if nlmsStable

        nlmsError = nlmsEnhanced-clean;

        nlmsMSE = mean(nlmsError.^2);

        nlmsOutputSNR = 10*log10( ...
            mean(clean.^2)/(nlmsMSE + EPS));

        NLMS_OutputSNR(k) = nlmsOutputSNR;
        NLMS_Improvement(k) = nlmsOutputSNR-inputSNR;
        NLMS_MSE(k) = nlmsMSE;

    end

    %% ------------------------------------------------------------
    % PRINT RESULTS
    % -------------------------------------------------------------

    fprintf('\nLMS RESULTS\n');

    if LMS_Stable(k)
        fprintf('Input SNR       : %8.2f dB\n',inputSNR);
        fprintf('Output SNR      : %8.2f dB\n',LMS_OutputSNR(k));
        fprintf('Improvement     : %8.2f dB\n',LMS_Improvement(k));
        fprintf('MSE             : %.8f\n',LMS_MSE(k));
    else
        fprintf('LMS unstable at this SNR.\n');
    end

    fprintf('\nNLMS RESULTS\n');

    if NLMS_Stable(k)
        fprintf('Input SNR       : %8.2f dB\n',inputSNR);
        fprintf('Output SNR      : %8.2f dB\n',NLMS_OutputSNR(k));
        fprintf('Improvement     : %8.2f dB\n',NLMS_Improvement(k));
        fprintf('MSE             : %.8f\n',NLMS_MSE(k));
    else
        fprintf('NLMS unstable at this SNR.\n');
    end

    %% ------------------------------------------------------------
    % SAVE -10 dB
    % -------------------------------------------------------------

    if targetSNR == -10

        savedNoisy10 = noisy;
        savedLMS10 = lmsEnhanced;
        savedNLMS10 = nlmsEnhanced;

        audiowrite(fullfile(AI_FOLDER, ...
            'AERIS_CLEAN_-10dB.wav'),clean,Fs);

        audiowrite(fullfile(AI_FOLDER, ...
            'AERIS_NOISY_-10dB.wav'), ...
            safeAudio(noisy),Fs);

        audiowrite(fullfile(AI_FOLDER, ...
            'AERIS_LMS_-10dB.wav'), ...
            safeAudio(lmsEnhanced),Fs);

        audiowrite(fullfile(AI_FOLDER, ...
            'AERIS_NLMS_-10dB.wav'), ...
            safeAudio(nlmsEnhanced),Fs);
    end

    %% ------------------------------------------------------------
    % SAVE -20 dB
    % -------------------------------------------------------------

    if targetSNR == -20

        savedLMS20 = lmsEnhanced;
        savedNLMS20 = nlmsEnhanced;

        audiowrite(fullfile(AI_FOLDER, ...
            'AERIS_LMS_-20dB.wav'), ...
            safeAudio(lmsEnhanced),Fs);

        audiowrite(fullfile(AI_FOLDER, ...
            'AERIS_NLMS_-20dB.wav'), ...
            safeAudio(nlmsEnhanced),Fs);
    end

end

%% ================================================================
% 9. RESULTS TABLE
% ================================================================

Results = table( ...
    TargetSNR, ...
    InputSNR, ...
    LMS_OutputSNR, ...
    LMS_Improvement, ...
    LMS_MSE, ...
    NLMS_OutputSNR, ...
    NLMS_Improvement, ...
    NLMS_MSE, ...
    LMS_Stable, ...
    NLMS_Stable);

writetable(Results, ...
    fullfile(OUTPUT_FOLDER,'AERIS_MASTER_FINAL_RESULTS.csv'));

%% ================================================================
% 10. SUMMARY STATISTICS
% ================================================================

validLMS = isfinite(LMS_Improvement);
validNLMS = isfinite(NLMS_Improvement);

if any(validLMS)

    avgLMS = mean(LMS_Improvement(validLMS));
    minLMS = min(LMS_Improvement(validLMS));
    maxLMS = max(LMS_Improvement(validLMS));

    [bestLMS,bestLMSidx] = max(LMS_Improvement);

    bestLMSSNR = TargetSNR(bestLMSidx);

else

    avgLMS = NaN;
    minLMS = NaN;
    maxLMS = NaN;
    bestLMS = NaN;
    bestLMSSNR = NaN;

end

if any(validNLMS)

    avgNLMS = mean(NLMS_Improvement(validNLMS));
    minNLMS = min(NLMS_Improvement(validNLMS));
    maxNLMS = max(NLMS_Improvement(validNLMS));

    [bestNLMS,bestNLMSidx] = max(NLMS_Improvement);

    bestNLMSSNR = TargetSNR(bestNLMSidx);

else

    avgNLMS = NaN;
    minNLMS = NaN;
    maxNLMS = NaN;
    bestNLMS = NaN;
    bestNLMSSNR = NaN;

end

%% ================================================================
% 11. SELECT BEST ALGORITHM
% ================================================================

if avgLMS >= avgNLMS
    bestAlgorithm = 'LMS';
    bestAverage = avgLMS;
else
    bestAlgorithm = 'NLMS';
    bestAverage = avgNLMS;
end

%% ================================================================
% 12. PRINT FINAL RESULTS
% ================================================================

fprintf('\n');
fprintf('================================================================\n');
fprintf('                       AERIS FINAL RESULTS\n');
fprintf('================================================================\n\n');

disp(Results);

fprintf('\n');
fprintf('================================================================\n');
fprintf('                         BEST RESULTS\n');
fprintf('================================================================\n\n');

fprintf('LMS\n');
fprintf('Average improvement : %.2f dB\n',avgLMS);
fprintf('Minimum improvement : %.2f dB\n',minLMS);
fprintf('Maximum improvement : %.2f dB\n',maxLMS);
fprintf('Best improvement    : %.2f dB\n',bestLMS);
fprintf('Best test SNR       : %.0f dB\n\n',bestLMSSNR);

fprintf('NLMS\n');
fprintf('Average improvement : %.2f dB\n',avgNLMS);
fprintf('Minimum improvement : %.2f dB\n',minNLMS);
fprintf('Maximum improvement : %.2f dB\n',maxNLMS);
fprintf('Best improvement    : %.2f dB\n',bestNLMS);
fprintf('Best test SNR       : %.0f dB\n\n',bestNLMSSNR);

fprintf('BEST OVERALL ALGORITHM : %s\n',bestAlgorithm);
fprintf('BEST AVERAGE IMPROVEMENT : %.2f dB\n',bestAverage);

%% ================================================================
% 13. PLOT 1 - INPUT VS OUTPUT SNR
% ================================================================

figure('Visible','off');

plot(TargetSNR,InputSNR,'o-','LineWidth',1.5);
hold on;
plot(TargetSNR,LMS_OutputSNR,'s-','LineWidth',1.5);
plot(TargetSNR,NLMS_OutputSNR,'^-','LineWidth',1.5);

grid on;

xlabel('Target SNR (dB)');
ylabel('SNR (dB)');

title('AERIS - Input vs Output SNR');

legend('Input','LMS Output','NLMS Output', ...
    'Location','best');

saveas(gcf,fullfile(OUTPUT_FOLDER, ...
    'AERIS_MASTER_FINAL_SNR_COMPARISON.png'));

close;

%% ================================================================
% 14. PLOT 2 - IMPROVEMENT
% ================================================================

figure('Visible','off');

plot(TargetSNR,LMS_Improvement,'s-','LineWidth',1.5);
hold on;
plot(TargetSNR,NLMS_Improvement,'^-','LineWidth',1.5);

grid on;

xlabel('Target SNR (dB)');
ylabel('SNR Improvement (dB)');

title('AERIS - SNR Improvement');

legend('LMS','NLMS','Location','best');

saveas(gcf,fullfile(OUTPUT_FOLDER, ...
    'AERIS_MASTER_FINAL_IMPROVEMENT.png'));

close;

%% ================================================================
% 15. PLOT 3 - MSE
% ================================================================

figure('Visible','off');

semilogy(TargetSNR,LMS_MSE,'s-','LineWidth',1.5);
hold on;
semilogy(TargetSNR,NLMS_MSE,'^-','LineWidth',1.5);

grid on;

xlabel('Target SNR (dB)');
ylabel('MSE');

title('AERIS - Mean Squared Error');

legend('LMS','NLMS','Location','best');

saveas(gcf,fullfile(OUTPUT_FOLDER, ...
    'AERIS_MASTER_FINAL_MSE.png'));

close;

%% ================================================================
% 16. -10 dB WAVEFORM
% ================================================================

if ~isempty(savedNoisy10)

    t = (0:length(clean)-1)/Fs;

    figure('Visible','off');

    plot(t,clean,'LineWidth',0.8);
    hold on;
    plot(t,savedNoisy10,'LineWidth',0.5);
    plot(t,savedLMS10,'LineWidth',0.8);
    plot(t,savedNLMS10,'LineWidth',0.8);

    grid on;

    xlabel('Time (seconds)');
    ylabel('Amplitude');

    title('AERIS -10 dB ANC Waveform');

    legend('Clean','Noisy','LMS Enhanced','NLMS Enhanced');

    saveas(gcf,fullfile(OUTPUT_FOLDER, ...
        'AERIS_-10dB_WAVEFORM_FINAL.png'));

    close;

end

%% ================================================================
% 17. SPECTROGRAMS
% ================================================================

if ~isempty(savedNoisy10)

    createSpectrogram(clean,Fs, ...
        fullfile(OUTPUT_FOLDER, ...
        'AERIS_-10dB_CLEAN_SPECTROGRAM.png'), ...
        'Clean Speech');

    createSpectrogram(savedNoisy10,Fs, ...
        fullfile(OUTPUT_FOLDER, ...
        'AERIS_-10dB_NOISY_SPECTROGRAM.png'), ...
        'Noisy Speech');

    createSpectrogram(savedLMS10,Fs, ...
        fullfile(OUTPUT_FOLDER, ...
        'AERIS_-10dB_LMS_SPECTROGRAM.png'), ...
        'LMS Enhanced Speech');

    createSpectrogram(savedNLMS10,Fs, ...
        fullfile(OUTPUT_FOLDER, ...
        'AERIS_-10dB_NLMS_SPECTROGRAM.png'), ...
        'NLMS Enhanced Speech');

end

%% ================================================================
% 18. CREATE AI-READY FEATURE DATASET
% ================================================================

fprintf('\n');
fprintf('================================================================\n');
fprintf('                 CREATING AI-READY DATASET\n');
fprintf('================================================================\n');

aiClean = clean;

if ~isempty(savedNoisy10)
    aiNoisy = savedNoisy10;
else
    aiNoisy = clean;
end

if ~isempty(savedLMS10)
    aiLMS = savedLMS10;
else
    aiLMS = clean;
end

if ~isempty(savedNLMS10)
    aiNLMS = savedNLMS10;
else
    aiNLMS = clean;
end

% Feature extraction
featureNames = { ...
    'RMS', ...
    'Energy', ...
    'ZeroCrossingRate', ...
    'MeanAbs', ...
    'StdDev', ...
    'PeakAmplitude', ...
    'CrestFactor', ...
    'SpectralCentroid', ...
    'SpectralBandwidth', ...
    'SpectralRolloff'};

cleanFeatures = extractAERISFeatures(aiClean,Fs);
noisyFeatures = extractAERISFeatures(aiNoisy,Fs);
lmsFeatures = extractAERISFeatures(aiLMS,Fs);
nlmsFeatures = extractAERISFeatures(aiNLMS,Fs);

FeatureTable = array2table( ...
    [cleanFeatures; ...
     noisyFeatures; ...
     lmsFeatures; ...
     nlmsFeatures], ...
    'VariableNames',featureNames);

Label = categorical({ ...
    'CLEAN'; ...
    'NOISY'; ...
    'LMS_ENHANCED'; ...
    'NLMS_ENHANCED'});

FeatureTable.Label = Label;

writetable(FeatureTable, ...
    fullfile(AI_FOLDER,'AERIS_AI_FEATURES.csv'));

%% ================================================================
% 19. COPY IMPORTANT AI AUDIO
% ================================================================

audiowrite(fullfile(AI_FOLDER,'clean.wav'), ...
    safeAudio(aiClean),Fs);

audiowrite(fullfile(AI_FOLDER,'noisy.wav'), ...
    safeAudio(aiNoisy),Fs);

audiowrite(fullfile(AI_FOLDER,'lms_enhanced.wav'), ...
    safeAudio(aiLMS),Fs);

audiowrite(fullfile(AI_FOLDER,'nlms_enhanced.wav'), ...
    safeAudio(aiNLMS),Fs);

%% ================================================================
% 20. CREATE AI DATASET INFORMATION FILE
% ================================================================

fid = fopen(fullfile(AI_FOLDER,'README_AI_DATASET.txt'),'w');

fprintf(fid,'AERIS AI-READY DATASET\n');
fprintf(fid,'======================\n\n');

fprintf(fid,'Sampling rate: %d Hz\n',Fs);
fprintf(fid,'Duration: %.2f seconds\n',duration);
fprintf(fid,'Samples: %d\n\n',N);

fprintf(fid,'Audio files:\n');
fprintf(fid,'1. clean.wav\n');
fprintf(fid,'2. noisy.wav\n');
fprintf(fid,'3. lms_enhanced.wav\n');
fprintf(fid,'4. nlms_enhanced.wav\n\n');

fprintf(fid,'Feature file:\n');
fprintf(fid,'AERIS_AI_FEATURES.csv\n\n');

fprintf(fid,'Features:\n');

for i = 1:length(featureNames)
    fprintf(fid,'%d. %s\n',i,featureNames{i});
end

fprintf(fid,'\nPurpose:\n');
fprintf(fid,'AI noise classification and adaptive ANC control.\n');

fclose(fid);

%% ================================================================
% 21. CREATE MASTER REPORT
% ================================================================

reportPath = fullfile(OUTPUT_FOLDER, ...
    'AERIS_MASTER_FINAL_REPORT.txt');

fid = fopen(reportPath,'w');

fprintf(fid,'AERIS MASTER FINAL REPORT\n');
fprintf(fid,'=========================\n\n');

fprintf(fid,'Base folder:\n%s\n\n',BASE);

fprintf(fid,'Clean files: %d\n',numel(cleanFiles));
fprintf(fid,'Noise files: %d\n',numel(noiseFiles));
fprintf(fid,'Sampling rate: %d Hz\n',Fs);
fprintf(fid,'Samples: %d\n',N);
fprintf(fid,'Duration: %.2f seconds\n\n',duration);

fprintf(fid,'LMS order: %d\n',LMS_ORDER);
fprintf(fid,'LMS requested step size: %.6f\n',LMS_MU);

fprintf(fid,'NLMS order: %d\n',NLMS_ORDER);
fprintf(fid,'NLMS step size: %.6f\n\n',NLMS_MU);

fprintf(fid,'RESULTS\n');
fprintf(fid,'=======\n\n');

for k = 1:numTests

    fprintf(fid, ...
        'Target SNR: %6.1f dB | Input: %7.2f dB | ', ...
        TargetSNR(k),InputSNR(k));

    if LMS_Stable(k)
        fprintf(fid,'LMS: %7.2f dB | Improvement: %7.2f dB | ', ...
            LMS_OutputSNR(k),LMS_Improvement(k));
    else
        fprintf(fid,'LMS: UNSTABLE | ');
    end

    if NLMS_Stable(k)
        fprintf(fid,'NLMS: %7.2f dB | Improvement: %7.2f dB\n', ...
            NLMS_OutputSNR(k),NLMS_Improvement(k));
    else
        fprintf(fid,'NLMS: UNSTABLE\n');
    end

end

fprintf(fid,'\nSUMMARY\n');
fprintf(fid,'=======\n');

fprintf(fid,'LMS average improvement: %.2f dB\n',avgLMS);
fprintf(fid,'LMS minimum improvement: %.2f dB\n',minLMS);
fprintf(fid,'LMS maximum improvement: %.2f dB\n',maxLMS);

fprintf(fid,'\nNLMS average improvement: %.2f dB\n',avgNLMS);
fprintf(fid,'NLMS minimum improvement: %.2f dB\n',minNLMS);
fprintf(fid,'NLMS maximum improvement: %.2f dB\n',maxNLMS);

fprintf(fid,'\nBEST ALGORITHM: %s\n',bestAlgorithm);
fprintf(fid,'BEST AVERAGE IMPROVEMENT: %.2f dB\n',bestAverage);

fprintf(fid,'\nAI STAGE READY\n');
fprintf(fid,'===============\n');
fprintf(fid,'AI-ready data folder:\n%s\n',AI_FOLDER);

fclose(fid);

%% ================================================================
% 22. FINAL MESSAGE
% ================================================================

fprintf('\n');
fprintf('================================================================\n');
fprintf('                  AERIS MASTER COMPLETE\n');
fprintf('================================================================\n\n');

fprintf('BEST ALGORITHM : %s\n',bestAlgorithm);
fprintf('BEST AVERAGE IMPROVEMENT : %.2f dB\n\n',bestAverage);

fprintf('Results saved to:\n%s\n\n',OUTPUT_FOLDER);

fprintf('AI-ready dataset saved to:\n%s\n\n',AI_FOLDER);

fprintf('Important AI files:\n');
fprintf('1. AERIS_AI_FEATURES.csv\n');
fprintf('2. clean.wav\n');
fprintf('3. noisy.wav\n');
fprintf('4. lms_enhanced.wav\n');
fprintf('5. nlms_enhanced.wav\n');
fprintf('6. README_AI_DATASET.txt\n');

fprintf('\n');
fprintf('================================================================\n');
fprintf('       MATLAB ANC STAGE COMPLETE - READY FOR AI\n');
fprintf('================================================================\n\n');

fprintf('NEXT STAGE:\n');
fprintf('AI-based noise classification.\n');
fprintf('================================================================\n');


%% ================================================================
% LOCAL FUNCTION: LMS
% ================================================================

function [enhanced,estimated,stable] = ...
    aerisLMS(d,reference,M,mu,clean)

N = length(d);

w = zeros(M,1);

estimated = zeros(N,1);
enhanced = zeros(N,1);

stable = true;

% Maximum allowed internal amplitude
MAX_VALUE = 10;

for n = M:N

    x = reference(n:-1:n-M+1);

    y = w' * x;

    e = d(n)-y;

    estimated(n) = y;
    enhanced(n) = e;

    w = w + mu*e*x;

    % Divergence protection
    if any(~isfinite(w)) || ...
       any(~isfinite(e)) || ...
       abs(e) > MAX_VALUE || ...
       max(abs(w)) > MAX_VALUE

        stable = false;

        enhanced(:) = NaN;
        estimated(:) = NaN;

        return;
    end
end

% Initial samples
enhanced(1:M-1) = d(1:M-1);

% Final numerical check
if any(~isfinite(enhanced))

    stable = false;

end

end


%% ================================================================
% LOCAL FUNCTION: NLMS
% ================================================================

function [enhanced,estimated,stable] = ...
    aerisNLMS(d,reference,M,mu,epsValue,clean)

N = length(d);

w = zeros(M,1);

estimated = zeros(N,1);
enhanced = zeros(N,1);

stable = true;

MAX_VALUE = 10;

for n = M:N

    x = reference(n:-1:n-M+1);

    powerX = x'*x;

    y = w'*x;

    e = d(n)-y;

    estimated(n) = y;
    enhanced(n) = e;

    denominator = powerX + epsValue;

    w = w + (mu/denominator)*e*x;

    % Numerical protection
    if any(~isfinite(w)) || ...
       ~isfinite(e) || ...
       abs(e) > MAX_VALUE || ...
       max(abs(w)) > MAX_VALUE

        stable = false;

        enhanced(:) = NaN;
        estimated(:) = NaN;

        return;
    end

end

enhanced(1:M-1) = d(1:M-1);

if any(~isfinite(enhanced))

    stable = false;

end

end


%% ================================================================
% LOCAL FUNCTION: SAFE AUDIO
% ================================================================

function y = safeAudio(x)

y = x;

y(~isfinite(y)) = 0;

peak = max(abs(y));

if peak > 0.999

    y = y ./ peak .* 0.999;

end

end


%% ================================================================
% LOCAL FUNCTION: SPECTROGRAM
% ================================================================

function createSpectrogram(x,Fs,fileName,plotTitle)

x = safeAudio(x);

figure('Visible','off');

spectrogram(x,512,384,1024,Fs,'yaxis');

title(plotTitle);
colorbar;

saveas(gcf,fileName);

close;

end


%% ================================================================
% LOCAL FUNCTION: FEATURE EXTRACTION
% ================================================================

function features = extractAERISFeatures(x,Fs)

x = x(:);

x(~isfinite(x)) = 0;

% RMS
rmsValue = sqrt(mean(x.^2));

% Energy
energyValue = sum(x.^2);

% Zero crossing rate
zcr = sum(abs(diff(x > 0))) / max(length(x)-1,1);

% Mean absolute amplitude
meanAbsValue = mean(abs(x));

% Standard deviation
stdValue = std(x);

% Peak
peakValue = max(abs(x));

% Crest factor
crestFactor = peakValue/(rmsValue + eps);

% FFT
N = length(x);

X = abs(fft(x));

X = X(1:floor(N/2)+1);

f = (0:length(X)-1)' * Fs/N;

powerSpectrum = X.^2;

totalPower = sum(powerSpectrum) + eps;

% Spectral centroid
spectralCentroid = sum(f.*powerSpectrum)/totalPower;

% Spectral bandwidth
spectralBandwidth = sqrt( ...
    sum(((f-spectralCentroid).^2).*powerSpectrum) ...
    /totalPower);

% Spectral rolloff 85%
cumulativePower = cumsum(powerSpectrum);

rolloffIndex = find( ...
    cumulativePower >= 0.85*totalPower,1,'first');

if isempty(rolloffIndex)
    rolloffIndex = length(f);
end

spectralRolloff = f(rolloffIndex);

features = [ ...
    rmsValue, ...
    energyValue, ...
    zcr, ...
    meanAbsValue, ...
    stdValue, ...
    peakValue, ...
    crestFactor, ...
    spectralCentroid, ...
    spectralBandwidth, ...
    spectralRolloff];

end