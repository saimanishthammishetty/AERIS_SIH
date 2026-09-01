"""
=============================================================================
PROJECT AERIS — ACOUSTIC FEATURE EXTRACTION PIPELINE
Module: AI/features/extract_features.py
=============================================================================
Provides modular, production-grade time-domain, frequency-domain, and cepstral
acoustic feature extraction for the AERIS noise classification engine.
"""

import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack
import scipy.stats as stats

TARGET_SR = 16000

# Precompute standard 26-band Mel Filterbank
def compute_mel_filterbank(num_filters=26, nfft=4096, sr=16000, low_freq=0, high_freq=8000):
    low_mel = 2595 * np.log10(1 + low_freq / 700.0)
    high_mel = 2595 * np.log10(1 + high_freq / 700.0)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = 700 * (10**(mel_points / 2595.0) - 1)
    bin_points = np.floor((nfft + 1) * hz_points / sr).astype(int)
    
    num_bins = nfft // 2 + 1
    fbank = np.zeros((num_filters, num_bins))
    for m in range(1, num_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m - 1, k] = (k - bin_points[m - 1]) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m - 1, k] = (bin_points[m + 1] - k) / (f_m_plus - f_m)
    return fbank

MEL_FBANK_DEFAULT = compute_mel_filterbank(num_filters=26, nfft=4096, sr=TARGET_SR)

def extract_audio_features(x: np.ndarray, sr: int = TARGET_SR, prev_spectrum: np.ndarray = None) -> tuple:
    """
    Extracts comprehensive acoustic features from an audio frame:
    
    TIME DOMAIN:
    - RMS, Peak Amplitude, Zero Crossing Rate, Crest Factor, Variance, Kurtosis, Skewness, Mean Absolute Amplitude
    
    FREQUENCY DOMAIN:
    - Dominant Frequency, Spectral Centroid, Spectral Bandwidth, Spectral Rolloff (85%),
      Spectral Flatness, Spectral Energy, Spectral Flux, Sub-band energies (Low, Mid, High)
      
    CEPSTRAL DOMAIN:
    - 13 MFCC Coefficients (MFCC_1 to MFCC_13)
    - Delta MFCC approximations
    
    Returns: (features_dict, magnitude_spectrum)
    """
    N = len(x)
    if N == 0:
        return {}, None
        
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 1. TIME-DOMAIN & STATISTICAL FEATURES
    rms = float(np.sqrt(np.mean(x**2)))
    peak_amp = float(np.max(np.abs(x)))
    mean_abs = float(np.mean(np.abs(x)))
    variance = float(np.var(x))
    std_dev = float(np.std(x, ddof=1)) if N > 1 else 0.0
    crest_factor = float(peak_amp / (rms + 1e-12))
    zcr = float(np.mean(np.abs(np.diff(x >= 0).astype(float)))) if N > 1 else 0.0
    kurt = float(stats.kurtosis(x)) if N > 3 else 0.0
    skewness = float(stats.skew(x)) if N > 2 else 0.0
    
    # 2. FREQUENCY-DOMAIN SPECTRAL FEATURES
    window = np.hanning(N)
    x_win = x * window
    nfft = 4096
    X_complex = np.fft.rfft(x_win, n=nfft)
    X = np.abs(X_complex)
    power_spectrum = X**2
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    sum_X = np.sum(X)
    spectral_energy = float(np.sum(power_spectrum))
    
    max_idx = np.argmax(X)
    dominant_freq = float(freqs[max_idx]) if len(freqs) > max_idx else 0.0
    
    if sum_X > 1e-12:
        centroid = float(np.sum(freqs * X) / sum_X)
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid)**2) * X) / sum_X))
        cum_energy = np.cumsum(power_spectrum)
        total_energy = cum_energy[-1]
        rolloff_idx = np.where(cum_energy >= 0.85 * total_energy)[0]
        rolloff = float(freqs[rolloff_idx[0]]) if len(rolloff_idx) > 0 else 0.0
    else:
        centroid = 0.0
        bandwidth = 0.0
        rolloff = 0.0
        
    # Spectral Flux
    if prev_spectrum is not None and len(prev_spectrum) == len(X):
        flux = float(np.sqrt(np.sum((X - prev_spectrum)**2)))
    else:
        flux = float(np.sqrt(np.sum(np.diff(X)**2)))
        
    # Spectral Flatness
    valid_power = power_spectrum + 1e-12
    geom_mean = float(np.exp(np.mean(np.log(valid_power))))
    arith_mean = float(np.mean(valid_power))
    flatness = float(geom_mean / (arith_mean + 1e-12))
    
    # Sub-band Energies
    low_band_energy = float(np.sum(power_spectrum[freqs <= 500]))
    mid_band_energy = float(np.sum(power_spectrum[(freqs > 500) & (freqs <= 2500)]))
    high_band_energy = float(np.sum(power_spectrum[freqs > 2500]))
    
    # 3. MFCC FEATURES (13 Coefficients)
    mel_energies = np.dot(MEL_FBANK_DEFAULT, power_spectrum)
    mel_energies = np.where(mel_energies > 1e-12, mel_energies, 1e-12)
    log_mel = np.log(mel_energies)
    mfcc_all = fftpack.dct(log_mel, type=2, norm='ortho')
    mfcc_13 = mfcc_all[:13]
    
    # Assemble Feature Dictionary
    features = {
        'RMS': rms,
        'PeakAmplitude': peak_amp,
        'MeanAbs': mean_abs,
        'Variance': variance,
        'StdDev': std_dev,
        'CrestFactor': crest_factor,
        'ZeroCrossingRate': zcr,
        'Kurtosis': kurt,
        'Skewness': skewness,
        'DominantFrequency': dominant_freq,
        'SpectralCentroid': centroid,
        'SpectralBandwidth': bandwidth,
        'SpectralRolloff': rolloff,
        'SpectralFlatness': flatness,
        'SpectralEnergy': spectral_energy,
        'SpectralFlux': flux,
        'BandEnergy_Low': low_band_energy,
        'BandEnergy_Mid': mid_band_energy,
        'BandEnergy_High': high_band_energy
    }
    
    for i in range(13):
        features[f'MFCC_{i+1}'] = float(mfcc_13[i])
        
    return features, X
