%% =========================================================
% AERIS - LMS vs NLMS COMPARISON
% ==========================================================

clear;
clc;
close all;

fprintf('\n');
fprintf('=================================================\n');
fprintf('          AERIS LMS vs NLMS COMPARISON\n');
fprintf('=================================================\n\n');

%% PATHS

projectFolder = ...
    'C:\Users\nuppe\Documents\MATLAB\AERIS_SIH';

outputFolder = fullfile(projectFolder,'OUTPUT');

cleanFile = fullfile(outputFolder,'clean_test.wav');
noisyFile = fullfile(outputFolder,'noisy_test.wav');
noiseFile = fullfile(outputFolder,'scaled_noise.wav');

%% LOAD SIGNALS

[clean,Fs] = audioread(cleanFile);
[noisy,Fs2] = audioread(noisyFile);
[referenceNoise,Fs3] = audioread(noiseFile);

%% MONO

if size(clean,2) > 1
    clean = mean(clean,2);
end

if size(noisy,2) > 1
    noisy = mean(noisy,2);
end

if size(referenceNoise,2) > 1
    referenceNoise = mean(referenceNoise,2);
end

%% LENGTH

N = min([length(clean), ...
         length(noisy), ...
         length(referenceNoise)]);

clean = clean(1:N);
noisy = noisy(1:N);
referenceNoise = referenceNoise(1:N);

clean = clean(:);
noisy = noisy(:);
referenceNoise = referenceNoise(:);

fprintf('Sampling rate : %d Hz\n',Fs);
fprintf('Samples       : %d\n',N);
fprintf('Duration      : %.2f seconds\n\n',N/Fs);

%% INPUT SNR

inputError = noisy - clean;

inputSNR = ...
    10*log10(sum(clean.^2)/sum(inputError.^2));

fprintf('Input SNR : %.2f dB\n\n',inputSNR);

%% =========================================================
% LMS
% ==========================================================

fprintf('Running LMS...\n');

filterOrder = 32;
muLMS = 0.005;

wLMS = zeros(filterOrder,1);
xBuffer = zeros(filterOrder,1);

estimatedLMS = zeros(N,1);
enhancedLMS = zeros(N,1);

for n = 1:N

    xBuffer = ...
        [referenceNoise(n); xBuffer(1:end-1)];

    estimatedLMS(n) = ...
        wLMS' * xBuffer;

    enhancedLMS(n) = ...
        noisy(n) - estimatedLMS(n);

    wLMS = ...
        wLMS + ...
        muLMS * enhancedLMS(n) * xBuffer;

end

%% LMS SNR

lmsError = enhancedLMS - clean;

lmsSNR = ...
    10*log10(sum(clean.^2)/sum(lmsError.^2));

lmsImprovement = lmsSNR - inputSNR;

lmsMSE = mean(lmsError.^2);

fprintf('LMS complete.\n');
fprintf('LMS Output SNR       : %.2f dB\n',lmsSNR);
fprintf('LMS Improvement      : %.2f dB\n',lmsImprovement);
fprintf('LMS MSE              : %.8f\n\n',lmsMSE);

%% =========================================================
% NLMS
% ==========================================================

fprintf('Running NLMS...\n');

muNLMS = 0.5;
epsilon = 1e-6;

wNLMS = zeros(filterOrder,1);
xBuffer = zeros(filterOrder,1);

estimatedNLMS = zeros(N,1);
enhancedNLMS = zeros(N,1);

for n = 1:N

    xBuffer = ...
        [referenceNoise(n); xBuffer(1:end-1)];

    estimatedNLMS(n) = ...
        wNLMS' * xBuffer;

    enhancedNLMS(n) = ...
        noisy(n) - estimatedNLMS(n);

    inputPower = ...
        xBuffer' * xBuffer;

    wNLMS = ...
        wNLMS + ...
        (muNLMS/(epsilon+inputPower)) ...
        * enhancedNLMS(n) * xBuffer;

end

%% NLMS SNR

nlmsError = enhancedNLMS - clean;

nlmsSNR = ...
    10*log10(sum(clean.^2)/sum(nlmsError.^2));

nlmsImprovement = nlmsSNR - inputSNR;

nlmsMSE = mean(nlmsError.^2);

fprintf('NLMS complete.\n');
fprintf('NLMS Output SNR       : %.2f dB\n',nlmsSNR);
fprintf('NLMS Improvement      : %.2f dB\n',nlmsImprovement);
fprintf('NLMS MSE              : %.8f\n\n',nlmsMSE);

%% =========================================================
% RESULTS
% ==========================================================

fprintf('=================================================\n');
fprintf('                  FINAL RESULTS\n');
fprintf('=================================================\n\n');

fprintf('%-15s %-12s %-12s %-12s\n', ...
    'Method','Input SNR','Output SNR','Improvement');

fprintf('%-15s %-12.2f %-12.2f %-12.2f\n', ...
    'LMS',inputSNR,lmsSNR,lmsImprovement);

fprintf('%-15s %-12.2f %-12.2f %-12.2f\n', ...
    'NLMS',inputSNR,nlmsSNR,nlmsImprovement);

fprintf('\n');

%% =========================================================
% BEST METHOD
% ==========================================================

if lmsSNR > nlmsSNR

    fprintf('BEST METHOD: LMS\n');
    fprintf('Best output SNR: %.2f dB\n',lmsSNR);

else

    fprintf('BEST METHOD: NLMS\n');
    fprintf('Best output SNR: %.2f dB\n',nlmsSNR);

end

%% =========================================================
% SAVE AUDIO
% ==========================================================

audiowrite( ...
    fullfile(outputFolder,'comparison_LMS.wav'), ...
    enhancedLMS,Fs);

audiowrite( ...
    fullfile(outputFolder,'comparison_NLMS.wav'), ...
    enhancedNLMS,Fs);

%% =========================================================
% COMPARISON BAR GRAPH
% ==========================================================

figure;

methods = categorical({'Noisy','LMS','NLMS'});

snrValues = ...
    [inputSNR,lmsSNR,nlmsSNR];

bar(methods,snrValues);

ylabel('SNR (dB)');

title('AERIS ANC Performance Comparison');

grid on;

saveas(gcf, ...
    fullfile(outputFolder,'LMS_vs_NLMS_SNR.png'));

%% =========================================================
% WAVEFORM COMPARISON
% ==========================================================

t = (0:N-1)/Fs;

figure;

plot(t,noisy);

hold on;

plot(t,enhancedLMS);

plot(t,enhancedNLMS);

xlabel('Time (seconds)');
ylabel('Amplitude');

title('AERIS - Noisy vs LMS vs NLMS');

legend('Noisy','LMS','NLMS');

grid on;

saveas(gcf, ...
    fullfile(outputFolder,'LMS_vs_NLMS_waveform.png'));

%% =========================================================
% SAVE RESULTS
% ==========================================================

fid = fopen( ...
    fullfile(outputFolder,'LMS_vs_NLMS_results.txt'),'w');

fprintf(fid,'AERIS LMS vs NLMS RESULTS\n');
fprintf(fid,'=========================\n\n');

fprintf(fid,'Sampling Rate : %d Hz\n',Fs);

fprintf(fid,'Input SNR     : %.4f dB\n\n',inputSNR);

fprintf(fid,'LMS Output SNR : %.4f dB\n',lmsSNR);
fprintf(fid,'LMS Improvement: %.4f dB\n',lmsImprovement);
fprintf(fid,'LMS MSE        : %.10f\n\n',lmsMSE);

fprintf(fid,'NLMS Output SNR : %.4f dB\n',nlmsSNR);
fprintf(fid,'NLMS Improvement: %.4f dB\n',nlmsImprovement);
fprintf(fid,'NLMS MSE        : %.10f\n',nlmsMSE);

fclose(fid);

fprintf('\n=================================================\n');
fprintf('       COMPARISON COMPLETE\n');
fprintf('=================================================\n');